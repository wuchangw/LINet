from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from data import encode_points, extract_physical_patch, normalize_modalities
from model import build_model


def _center_from_round(round_prompt):
    for key in ("pos", "neg", "pos_scribble", "neg_scribble"):
        points = round_prompt.get(key, [])
        if points:
            return tuple(float(v) for v in points[0])
    if "bbox" in round_prompt:
        box = round_prompt["bbox"]
        return tuple((float(box[i]) + float(box[i + 3])) / 2.0 for i in range(3))
    raise ValueError("Each prompt round must contain a positive point, positive scribble, or bbox")


def _round_centers(round_prompt):
    """Every positive or negative prompt of ONE interaction anchors a local
    refinement ROI. Each interaction is refined exactly once."""
    centers = []
    for key in ("pos", "neg"):
        centers.extend(tuple(float(v) for v in point) for point in round_prompt.get(key, []))
    for key in ("pos_scribble", "neg_scribble"):
        scribble = round_prompt.get(key, [])
        if scribble:
            centers.append(tuple(float(v) for v in np.asarray(scribble, dtype=np.float64).mean(axis=0)))
    if "bbox" in round_prompt:
        box = round_prompt["bbox"]
        centers.append(tuple((float(box[i]) + float(box[i + 3])) / 2.0 for i in range(3)))
    if not centers:
        centers.append(_center_from_round(round_prompt))
    return list(dict.fromkeys(centers))


def _source_geometry(center, source_spacing, target_spacing, patch_size):
    size = [max(2, int(round(p * t / s))) for p, t, s in zip(patch_size, target_spacing, source_spacing)]
    starts = [int(round(c - n / 2)) for c, n in zip(center, size)]
    return size, starts


def _points_to_patch(points, starts, source_size, patch_size):
    converted = []
    for point in points:
        value = [int(round((float(p) - start) * dst / src))
                 for p, start, dst, src in zip(point, starts, patch_size, source_size)]
        if all(0 <= p < n for p, n in zip(value, patch_size)):
            converted.append(tuple(value))
    return converted


def _bbox_channels(box, starts, source_size, patch_size):
    inside = np.zeros(tuple(patch_size), dtype=np.float32)
    if box is None:
        return np.stack([inside, inside])
    lo = _points_to_patch([box[:3]], starts, source_size, patch_size)
    hi = _points_to_patch([[float(v) - 1 for v in box[3:]]], starts, source_size, patch_size)
    if not lo or not hi:
        return np.stack([inside, inside])
    lo, hi = np.asarray(lo[0]), np.asarray(hi[0]) + 1
    inside[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] = 1.0
    signed = ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(1.0 - inside)
    signed /= max(float(np.abs(signed).max()), 1.0)
    return np.stack([inside, signed.astype(np.float32)])


def _restore_patch(array, original_shape, center, source_size, order=1):
    starts = [int(round(c - n / 2)) for c, n in zip(center, source_size)]
    restored = ndimage.zoom(array, [src / dst for src, dst in zip(source_size, array.shape)], order=order)
    fixed = np.zeros(tuple(source_size), dtype=np.float32)
    common = tuple(slice(0, min(a, b)) for a, b in zip(restored.shape, source_size)); fixed[common] = restored[common]
    src_slices, dst_slices = [], []
    for start, size, total in zip(starts, source_size, original_shape):
        dst0, dst1 = max(0, start), min(total, start + size)
        src0, src1 = dst0 - start, dst1 - start
        src_slices.append(slice(src0, src1)); dst_slices.append(slice(dst0, dst1))
    return fixed[tuple(src_slices)], tuple(dst_slices)


def paste_probability(probability, original_shape, center, source_size):
    restored, dst_slices = _restore_patch(probability, original_shape, center, source_size)
    output = np.zeros(tuple(original_shape), dtype=np.float32)
    output[dst_slices] = restored
    return output


def _empty_prompt_tensors(patch_size, device):
    prompt = torch.zeros((1, 2, *patch_size), device=device)
    previous = torch.zeros((1, 1, *patch_size), device=device)
    route = torch.zeros((1, 3), device=device)
    return prompt, previous, route


def _axis_starts(length, window, overlap):
    if length <= window:
        return [int(round((length - window) / 2.0))]
    step = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, length - window + 1, step))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _blend_window(shape):
    axes = []
    for length in shape:
        values = np.hanning(length).astype(np.float32) if length > 2 else np.ones(length, dtype=np.float32)
        axes.append(np.maximum(values, 0.05))
    window = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (window / float(window.max())).astype(np.float32)


@torch.no_grad()
def predict_auto_volume(model, ct, pet, spacing, target_spacing, patch_size,
                        plan, config, amp_dtype, overlap=None):
    """Run the pure nnU-Net automatic path over the complete scan.

    One forward pass per window with the dedicated ``forward_auto`` entry
    point: no prompts, no feedback, no fusion. The composed volume is the
    complete initial segmentation used for every later prompt update.
    """
    overlap = float(config.get("auto_sliding_window_overlap", 0.5) if overlap is None else overlap)
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Sliding-window overlap must be in [0, 1)")
    source_size, _ = _source_geometry((0.0, 0.0, 0.0), spacing, target_spacing, patch_size)
    starts = [_axis_starts(n, w, overlap) for n, w in zip(ct.shape, source_size)]
    accumulator = np.zeros(ct.shape, dtype=np.float32)
    weights = np.zeros(ct.shape, dtype=np.float32)
    importance = _blend_window(patch_size)
    amp_enabled = str(config["device"]).startswith("cuda")
    window_count = 0
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                center = tuple(start + size / 2.0 for start, size in zip((z, y, x), source_size))
                ct_p, pet_p = extract_physical_patch(
                    [ct, pet], center, spacing, target_spacing, patch_size, (1, 1), (-1000.0, 0.0),
                )
                image = torch.from_numpy(normalize_modalities(ct_p, pet_p, plan))[None].to(config["device"])
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model.forward_auto(image)
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                local_probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
                restored_probability, dst = _restore_patch(local_probability, ct.shape, center, source_size)
                restored_weight, weight_dst = _restore_patch(importance, ct.shape, center, source_size)
                if dst != weight_dst:
                    raise RuntimeError("Probability and blending windows were restored to different regions")
                accumulator[dst] += restored_probability * restored_weight
                weights[dst] += restored_weight
                window_count += 1
    return accumulator / np.maximum(weights, 1e-6), window_count


def _prompt_influence_mask(round_prompt, starts, source_size, patch_size, target_spacing,
                           initial_probability, config):
    seed = np.zeros(tuple(patch_size), dtype=bool)
    influence = np.zeros(tuple(patch_size), dtype=bool)
    for key in ("pos", "neg", "pos_scribble", "neg_scribble"):
        local = _points_to_patch(round_prompt.get(key, []), starts, source_size, patch_size)
        for point in local:
            seed[point] = True
    if "bbox" in round_prompt:
        inside = _bbox_channels(round_prompt["bbox"], starts, source_size, patch_size)[0] > 0
        if inside.any():
            margin = float(config.get("prompt_bbox_margin_mm", 8.0))
            influence |= ndimage.distance_transform_edt(~inside, sampling=target_spacing) <= margin
    if seed.any():
        radius = float(config.get("prompt_refinement_radius_mm", 48.0))
        influence |= ndimage.distance_transform_edt(~seed, sampling=target_spacing) <= radius
    # A negative click on an automatic false positive should be able to remove
    # the whole connected component, even when it extends beyond the radius.
    components, _ = ndimage.label(
        initial_probability >= float(config.get("auto_threshold", 0.5)),
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    touched = np.unique(components[seed | influence])
    touched = touched[touched > 0]
    if touched.size:
        influence |= np.isin(components, touched)
    return influence


@torch.no_grad()
def predict_roi(model, ct, pet, center, spacing, target_spacing, patch_size,
                round_prompt, plan, config, amp_dtype, initial_probability=None):
    """ONE prompt interaction refined by exactly ONE forward pass.

    The frozen global automatic probability enters as ``prev_prob``; the model
    returns a local increment for the prompted region (add / delete / boundary
    fix / fill-in). The caller composites the result only inside the influence
    mask, so every unprompted part of the automatic segmentation is preserved.
    """
    source_size, starts = _source_geometry(center, spacing, target_spacing, patch_size)
    ct_p, pet_p = extract_physical_patch(
        [ct, pet], center, spacing, target_spacing, patch_size, (1, 1), (-1000.0, 0.0),
    )
    image = torch.from_numpy(normalize_modalities(ct_p, pet_p, plan))[None].to(config["device"])
    if initial_probability is None:
        initial_patch = np.zeros(tuple(patch_size), dtype=np.float32)
    else:
        initial_patch = extract_physical_patch(
            [initial_probability], center, spacing, target_spacing, patch_size, (1,), (0.0,),
        )[0].astype(np.float32)
    previous = torch.from_numpy(initial_patch[None, None]).to(config["device"])
    local_pos = _points_to_patch(round_prompt.get("pos", []), starts, source_size, patch_size)
    local_neg = _points_to_patch(round_prompt.get("neg", []), starts, source_size, patch_size)
    local_pos_s = _points_to_patch(round_prompt.get("pos_scribble", []), starts, source_size, patch_size)
    local_neg_s = _points_to_patch(round_prompt.get("neg_scribble", []), starts, source_size, patch_size)
    box = round_prompt.get("bbox")
    click = np.stack([encode_points(local_pos, patch_size, config["click_sigma"]),
                      encode_points(local_neg, patch_size, config["click_sigma"])])
    scribble = np.stack([encode_points(local_pos_s, patch_size, config["click_sigma"]),
                         encode_points(local_neg_s, patch_size, config["click_sigma"])])
    bbox = _bbox_channels(box, starts, source_size, patch_size)
    route = torch.tensor([[float(bool(local_pos or local_neg)), float(np.any(bbox[0])),
                           float(bool(local_pos_s or local_neg_s))]], device=config["device"])
    tensors = [torch.from_numpy(x[None]).to(config["device"]) for x in (click, bbox, scribble)]
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=str(config["device"]).startswith("cuda")):
        outputs = model.forward_refine(image, tensors[0], tensors[1], tensors[2], previous, route)
    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    refined_patch = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    influence_patch = _prompt_influence_mask(
        round_prompt, starts, source_size, patch_size, target_spacing, initial_patch, config,
    )
    refined, dst = _restore_patch(refined_patch, ct.shape, center, source_size)
    influence, influence_dst = _restore_patch(
        influence_patch.astype(np.float32), ct.shape, center, source_size, order=0,
    )
    if dst != influence_dst:
        raise RuntimeError("Refinement and influence masks were restored to different regions")
    return refined, influence >= 0.5, dst


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Whole-volume automatic segmentation with local prompt refinement")
    parser.add_argument("--ct", required=True); parser.add_argument("--pet", required=True)
    parser.add_argument("--prompts-json", help="Optional list of single-update prompt interactions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--plan"); parser.add_argument("--device")
    parser.add_argument("--overlap", type=float, help="Automatic sliding-window overlap")
    parser.add_argument("--threshold", type=float, help="Final probability threshold")
    parser.add_argument("--auto-output", help="Optional path for the unrefined automatic mask")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); config = checkpoint["config"]
    if args.device: config["device"] = args.device
    plan_path = args.plan or config["plan_path"]; plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    rounds = json.loads(Path(args.prompts_json).read_text(encoding="utf-8")) if args.prompts_json else []
    if not isinstance(rounds, list): raise ValueError("Prompt JSON must contain a list of prompt interactions")
    if args.prompts_json and not rounds: raise ValueError("Prompt JSON must contain at least one round")
    ct_img, pet_img = nib.load(args.ct), nib.load(args.pet)
    ct = np.asarray(np.asanyarray(ct_img.dataobj), dtype=np.float32)
    pet = np.asarray(np.asanyarray(pet_img.dataobj), dtype=np.float32)
    spacing = tuple(float(v) for v in ct_img.header.get_zooms()[:3]); target_spacing = tuple(plan["target_spacing"])
    patch_size = tuple(config["patch_size"])
    model = build_model(config).to(config["device"]); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    amp_dtype = torch.float16 if config["amp_dtype"] == "float16" else torch.bfloat16
    # Stage 1 - global automatic segmentation: one pure nnU-Net forward per
    # sliding window finds every suspected lesion in the whole volume.
    probability, window_count = predict_auto_volume(
        model, ct, pet, spacing, target_spacing, patch_size, plan, config, amp_dtype, args.overlap,
    )
    threshold = float(config.get("auto_threshold", 0.5) if args.threshold is None else args.threshold)
    if args.auto_output:
        automatic_mask = (probability >= threshold).astype(np.uint8)
        nib.save(nib.Nifti1Image(automatic_mask, ct_img.affine, ct_img.header), args.auto_output)
    # Stage 2 - local incremental updates: each prompt interaction is refined
    # exactly once and composited only inside its influence mask. Everything
    # outside the prompted region keeps the automatic result unchanged.
    roi_count = 0
    for round_prompt in rounds:
        for center in _round_centers(round_prompt):
            refined, influence, dst = predict_roi(
                model, ct, pet, center, spacing, target_spacing, patch_size, round_prompt,
                plan, config, amp_dtype, initial_probability=probability,
            )
            current = probability[dst]
            current[influence] = refined[influence]
            roi_count += 1
    prediction = (probability >= threshold).astype(np.uint8)
    nib.save(nib.Nifti1Image(prediction, ct_img.affine, ct_img.header), args.output)
    probability_path = str(Path(args.output).with_name(Path(args.output).name.replace(".nii.gz", "_prob.nii.gz")))
    nib.save(nib.Nifti1Image(probability.astype(np.float32), ct_img.affine, ct_img.header), probability_path)
    print(f"Processed {window_count} automatic window(s) and {roi_count} single-update prompt ROI(s)\n"
          f"Saved mask: {args.output}\nSaved probability: {probability_path}")


if __name__ == "__main__":
    main()
