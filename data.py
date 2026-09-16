from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage


STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)
PREPROCESS_VERSION = 2
PROMPT_ROUTE = {
    "automatic": (0.0, 0.0, 0.0),
    "click": (1.0, 0.0, 0.0),
    "bbox": (0.0, 1.0, 0.0),
    "scribble": (0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    ct_path: str
    pet_path: str
    label_path: str


@dataclass(frozen=True)
class LesionInfo:
    component_id: int
    center_zyx: tuple[float, float, float]
    voxels: int
    volume_mm3: float


def _load_split(path: str, key: int, stage: str) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(obj[key][stage])


def build_registry(config: dict) -> dict[str, list[CaseRecord]]:
    registry: dict[str, list[CaseRecord]] = {}
    missing: list[dict] = []
    for stage in ("train", "val"):
        records = []
        for case_id in _load_split(config["split_json"], config.get("split_key", 0), stage):
            paths = {
                "ct": Path(config["images_dir"]) / f"{case_id}_{config['ct_suffix']}.nii.gz",
                "pet": Path(config["images_dir"]) / f"{case_id}_{config['pet_suffix']}.nii.gz",
                "label": Path(config["labels_dir"]) / f"{case_id}.nii.gz",
            }
            absent = [name for name, path in paths.items() if not path.exists()]
            if absent:
                missing.append({"stage": stage, "case_id": case_id, "missing": absent})
                continue
            records.append(CaseRecord(case_id, str(paths["ct"]), str(paths["pet"]), str(paths["label"])))
        registry[stage] = records

    manifest = {
        "train_cases": [r.case_id for r in registry["train"]],
        "val_cases": [r.case_id for r in registry["val"]],
        "missing": missing,
    }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if missing and not config.get("allow_missing", False):
        raise FileNotFoundError(f"{len(missing)} split entries are incomplete; see data_manifest.json")
    if not registry["train"] or not registry["val"]:
        raise RuntimeError("Both train and val must contain at least one usable case")
    return registry


def create_preprocess_plan(records: Sequence[CaseRecord], path: str) -> dict:
    spacings, ct_samples, pet_samples, lesion_pet_samples = [], [], [], []
    for record in records:
        ct_img, pet_img = nib.load(record.ct_path), nib.load(record.pet_path)
        spacings.append(tuple(float(v) for v in pet_img.header.get_zooms()[:3]))
        ct = np.asanyarray(ct_img.dataobj)
        pet = np.asanyarray(pet_img.dataobj)
        stride = tuple(max(1, int(math.ceil(s / 64))) for s in pet.shape)
        ct_sample = np.asarray(ct[::stride[0], ::stride[1], ::stride[2]], dtype=np.float32)
        sample = np.asarray(pet[::stride[0], ::stride[1], ::stride[2]], dtype=np.float32)
        ct_sample = ct_sample[np.isfinite(ct_sample) & (ct_sample > -950.0)]
        sample = sample[np.isfinite(sample)]
        if ct_sample.size:
            ct_samples.append(ct_sample)
        if sample.size:
            pet_samples.append(sample)
        label = np.asarray(np.asanyarray(nib.load(record.label_path).dataobj) > 0)
        lesion_values = np.asarray(pet[label & np.isfinite(pet)], dtype=np.float32)
        if lesion_values.size:
            lesion_pet_samples.append(lesion_values)
    spacing = np.median(np.asarray(spacings, dtype=np.float64), axis=0)
    ct_values = np.concatenate(ct_samples) if ct_samples else np.asarray([-1000.0, 0.0, 2000.0], dtype=np.float32)
    ct_clipped = np.clip(ct_values, -1000.0, 2000.0)
    values = np.concatenate(pet_samples) if pet_samples else np.asarray([0.0, 1.0], dtype=np.float32)
    lesion_values = np.concatenate(lesion_pet_samples) if lesion_pet_samples else values
    lo, hi = np.percentile(values, [0.5, 99.5])
    clipped = np.clip(values, lo, hi)
    suv_cap = max(float(np.percentile(values, 99.9)), float(np.percentile(lesion_values, 99.5)), 1e-6)
    plan = {
        "version": PREPROCESS_VERSION,
        "target_spacing": [float(v) for v in spacing],
        "ct_clip": [-1000.0, 2000.0],
        "ct_mean": float(ct_clipped.mean()),
        "ct_std": float(ct_clipped.std() + 1e-6),
        "pet_clip": [float(lo), float(hi)],
        "pet_mean": float(clipped.mean()),
        "pet_std": float(clipped.std() + 1e-6),
        "suv_log_cap": float(np.log1p(suv_cap)),
        "suv_raw_cap": float(suv_cap),
        "source": "training split only",
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def load_or_create_plan(records: Sequence[CaseRecord], path: str) -> dict:
    plan_path = Path(path)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if int(plan.get("version", 0)) == PREPROCESS_VERSION:
            return plan
    return create_preprocess_plan(records, path)


@lru_cache(maxsize=4)
def _load_case(ct_path: str, pet_path: str, label_path: str):
    ct_img, pet_img, label_img = nib.load(ct_path), nib.load(pet_path), nib.load(label_path)
    ct = np.asarray(np.asanyarray(ct_img.dataobj), dtype=np.float32)
    pet = np.asarray(np.asanyarray(pet_img.dataobj), dtype=np.float32)
    label = np.asarray(np.asanyarray(label_img.dataobj) > 0, dtype=np.uint8)
    spacing = tuple(float(v) for v in ct_img.header.get_zooms()[:3])
    return ct, pet, label, spacing


def connected_lesions(label: np.ndarray, spacing: Sequence[float]) -> tuple[np.ndarray, list[LesionInfo]]:
    components, count = ndimage.label(label.astype(bool), structure=STRUCTURE_26)
    voxel_mm3 = float(np.prod(spacing))
    lesions = []
    for component_id in range(1, count + 1):
        points = np.argwhere(components == component_id)
        if not len(points):
            continue
        lesions.append(LesionInfo(
            component_id=component_id,
            center_zyx=tuple(float(v) for v in points.mean(axis=0)),
            voxels=int(len(points)),
            volume_mm3=float(len(points) * voxel_mm3),
        ))
    return components, lesions


@lru_cache(maxsize=64)
def _component_catalog(label_path: str, spacing: tuple[float, float, float]):
    label = np.asarray(np.asanyarray(nib.load(label_path).dataobj) > 0, dtype=np.uint8)
    components, lesions = connected_lesions(label, spacing)
    coordinates = {lesion.component_id: np.argwhere(components == lesion.component_id).astype(np.int32)
                   for lesion in lesions}
    return tuple(lesions), coordinates


def _crop_pad(array: np.ndarray, center: Sequence[float], size: Sequence[int], value: float) -> np.ndarray:
    starts = [int(round(c - s / 2)) for c, s in zip(center, size)]
    ends = [start + int(s) for start, s in zip(starts, size)]
    src = tuple(slice(max(0, a), min(n, b)) for a, b, n in zip(starts, ends, array.shape))
    crop = array[src]
    pads = [(max(0, -a), max(0, b - n)) for a, b, n in zip(starts, ends, array.shape)]
    return np.pad(crop, pads, mode="constant", constant_values=value)


def _resize_exact(array: np.ndarray, shape: Sequence[int], order: int) -> np.ndarray:
    zoom = [float(dst) / float(src) for src, dst in zip(array.shape, shape)]
    result = ndimage.zoom(array, zoom, order=order, mode="nearest", prefilter=order > 1)
    fixed = np.zeros(tuple(shape), dtype=result.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(result.shape, shape))
    fixed[slices] = result[slices]
    return fixed


def extract_physical_patch(arrays: Sequence[np.ndarray], center, source_spacing, target_spacing, patch_size, orders, fills):
    source_size = [max(2, int(round(p * t / s))) for p, t, s in zip(patch_size, target_spacing, source_spacing)]
    outputs = []
    for array, order, fill in zip(arrays, orders, fills):
        crop = _crop_pad(array, center, source_size, fill)
        outputs.append(_resize_exact(crop, patch_size, order))
    return outputs


def extract_component_patch(coordinates: np.ndarray, center, source_spacing, target_spacing, patch_size):
    source_size = np.asarray([max(2, int(round(p * t / s)))
                              for p, t, s in zip(patch_size, target_spacing, source_spacing)], dtype=np.int64)
    starts = np.asarray([int(round(c - n / 2)) for c, n in zip(center, source_size)], dtype=np.int64)
    local = coordinates.astype(np.int64) - starts[None]
    valid = np.all((local >= 0) & (local < source_size[None]), axis=1)
    source = np.zeros(tuple(int(v) for v in source_size), dtype=np.uint8)
    if valid.any():
        source[tuple(local[valid].T)] = 1
    target = _resize_exact(source, patch_size, 0) > 0
    if coordinates.size and not target.any():
        # Nearest-neighbour downsampling can erase a one-voxel lesion. Preserve
        # its physical existence at the mapped component centre.
        mapped = np.rint((np.asarray(center) - starts) * np.asarray(patch_size) / source_size).astype(int)
        mapped = np.clip(mapped, 0, np.asarray(patch_size) - 1)
        target[tuple(mapped)] = True
    return target.astype(np.uint8)


def encode_points(points: Sequence[Sequence[int]], shape: Sequence[int], sigma: float) -> np.ndarray:
    seeds = np.zeros(tuple(shape), dtype=bool)
    for point in points:
        point = tuple(int(v) for v in point)
        if all(0 <= p < n for p, n in zip(point, shape)):
            seeds[point] = True
    if not seeds.any():
        return np.zeros(tuple(shape), dtype=np.float32)
    return np.exp(-ndimage.distance_transform_edt(~seeds) / max(float(sigma), 1e-6)).astype(np.float32)


def _sample_points(mask: np.ndarray, count: int, rng: np.random.Generator, deepest_first: bool = False,
                   distance: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Sample points inside ``mask``.

    ``distance`` may hold a precomputed EDT of the same mask (offline cache);
    it is masked here so a shared per-component map cannot leak depth from a
    neighbouring component.
    """
    points = np.argwhere(mask)
    if not len(points) or count <= 0:
        return []
    selected = []
    if deepest_first:
        depth = ndimage.distance_transform_edt(mask) if distance is None else np.where(mask, distance, -1.0)
        selected.append(tuple(int(v) for v in np.unravel_index(int(depth.argmax()), depth.shape)))
    while len(selected) < count:
        selected.append(tuple(int(v) for v in points[int(rng.integers(0, len(points)))]))
    return selected[:count]


def _sample_points_across_components(mask: np.ndarray, count: int, rng: np.random.Generator,
                                     component_ids: np.ndarray | None = None,
                                     depth: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Spread positive seeds over as many connected targets as the budget allows.

    ``component_ids`` / ``depth`` let the cached path reuse a connected-component
    labelling and a per-component depth map computed once offline.
    """
    if component_ids is None:
        components, component_count = ndimage.label(mask.astype(bool), structure=STRUCTURE_26)
    else:
        components, component_count = component_ids, int(component_ids.max())
    if component_count == 0 or count <= 0:
        return []
    candidates = []
    for component_id in range(1, component_count + 1):
        component = components == component_id
        if not component.any():
            continue
        if depth is not None:
            masked = np.where(component, depth, -1.0)
            deepest = tuple(int(v) for v in np.unravel_index(int(masked.argmax()), masked.shape))
        else:
            distance = ndimage.distance_transform_edt(component)
            deepest = tuple(int(v) for v in np.unravel_index(int(distance.argmax()), distance.shape))
        candidates.append((int(component.sum()), component, deepest))
    if not candidates:
        return []
    # Cover large and small lesions alike before spending additional clicks.
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [item[2] for item in candidates[:count]]
    cursor = 0
    while len(selected) < count:
        component = candidates[cursor % len(candidates)][1]
        points = np.argwhere(component)
        selected.append(tuple(int(v) for v in points[int(rng.integers(0, len(points)))]))
        cursor += 1
    return selected


def _random_walk(mask: np.ndarray, length: int, rng: np.random.Generator,
                 distance: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    if not mask.any() or length <= 0:
        return []
    depth = ndimage.distance_transform_edt(mask) if distance is None else np.where(mask, distance, 0.0)
    current = np.asarray(np.unravel_index(int(depth.argmax()), depth.shape), dtype=np.int64)
    path = [tuple(int(v) for v in current)]
    offsets = np.asarray([(z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
                          if (z, y, x) != (0, 0, 0)], dtype=np.int64)
    for _ in range(length - 1):
        candidates = current[None] + offsets
        valid = np.all((candidates >= 0) & (candidates < np.asarray(mask.shape)[None]), axis=1)
        candidates = candidates[valid]
        candidates = candidates[np.asarray([mask[tuple(point)] for point in candidates], dtype=bool)]
        if not len(candidates):
            break
        current = candidates[int(rng.integers(0, len(candidates)))]
        path.append(tuple(int(v) for v in current))
    return path


def build_prompts(target: np.ndarray, other_lesions: np.ndarray, pet_raw: np.ndarray, prompt_type: str,
                  rng: np.random.Generator, config: dict,
                  component_ids: np.ndarray | None = None,
                  depth: np.ndarray | None = None,
                  negative_exclusion: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], tuple[float, float, float]]:
    """Build the prompt channels for one sample.

    ``component_ids`` and ``depth`` are optional offline maps (connected
    components of ``target`` and a per-component depth/EDT map). Supplying them
    removes the repeated scipy EDT work; the sampled points are unchanged.
    """
    shape = target.shape
    excluded = np.zeros(shape, dtype=bool) if negative_exclusion is None else negative_exclusion.astype(bool)
    zero = lambda: np.zeros(shape, dtype=np.float32)
    click, bbox, scribble = np.stack([zero(), zero()]), np.stack([zero(), zero()]), np.stack([zero(), zero()])
    if prompt_type == "automatic":
        # The image encoder/decoder must remain a complete nnU-Net-style
        # segmenter. Prompt and feedback branches receive true zero inputs.
        pass
    elif prompt_type == "click":
        pos = _sample_points_across_components(target > 0, int(config["clicks_pos"]), rng, component_ids, depth)
        hotspot = (other_lesions > 0)
        if hotspot.sum() < int(config["clicks_neg"]):
            valid = (~target.astype(bool)) & (~excluded) & np.isfinite(pet_raw)
            threshold = np.percentile(pet_raw[valid], 98) if valid.any() else np.inf
            hotspot |= valid & (pet_raw >= threshold)
        neg = _sample_points(hotspot, int(config["clicks_neg"]), rng)
        click = np.stack([encode_points(pos, shape, config["click_sigma"]), encode_points(neg, shape, config["click_sigma"])])
    elif prompt_type == "bbox":
        if component_ids is None:
            components, component_count = ndimage.label(target > 0, structure=STRUCTURE_26)
        else:
            components, component_count = component_ids, int(component_ids.max())
        if component_count:
            jitter = np.maximum(1, np.rint(float(config["bbox_jitter_mm"]) / np.asarray(config["target_spacing"])).astype(int))
            inside = zero()
            for component_id in range(1, component_count + 1):
                points = np.argwhere(components == component_id)
                if not len(points):
                    continue
                lo, hi = points.min(0), points.max(0) + 1
                lo = np.maximum(0, lo + rng.integers(-jitter, jitter + 1))
                hi = np.minimum(shape, hi + rng.integers(-jitter, jitter + 1))
                hi = np.maximum(hi, lo + 1)
                inside[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] = 1.0
            signed = ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(1.0 - inside)
            signed = signed / max(float(np.abs(signed).max()), 1.0)
            bbox = np.stack([inside, signed.astype(np.float32)])
    elif prompt_type == "scribble":
        length = int(rng.integers(config["scribble_length"][0], config["scribble_length"][1] + 1))
        if component_ids is None:
            components, component_count = ndimage.label(target > 0, structure=STRUCTURE_26)
        else:
            components, component_count = component_ids, int(component_ids.max())
        pos = []
        if component_count:
            per_component = max(1, length // component_count)
            for component_id in range(1, component_count + 1):
                component = components == component_id
                if component.any():
                    pos.extend(_random_walk(component, per_component, rng, depth))
        negative_region = other_lesions > 0
        if not negative_region.any():
            valid = (~target.astype(bool)) & (~excluded) & np.isfinite(pet_raw)
            threshold = np.percentile(pet_raw[valid], 98) if valid.any() else np.inf
            negative_region = valid & (pet_raw >= threshold)
        neg = _random_walk(negative_region, max(1, length // 2), rng)
        scribble = np.stack([encode_points(pos, shape, config["click_sigma"]), encode_points(neg, shape, config["click_sigma"])])
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")
    return {"click": click.astype(np.float32), "bbox": bbox.astype(np.float32), "scribble": scribble.astype(np.float32)}, PROMPT_ROUTE[prompt_type]


def normalize_modalities(ct: np.ndarray, pet: np.ndarray, plan: dict) -> np.ndarray:
    ct_lo, ct_hi = plan.get("ct_clip", [-1000.0, 1000.0])
    ct = np.clip(ct, float(ct_lo), float(ct_hi))
    body = ct > -950
    mean = float(plan.get("ct_mean", ct[body].mean() if body.any() else ct.mean()))
    std = float(plan.get("ct_std", (ct[body].std() if body.any() else ct.std()) + 1e-6))
    ct = (ct - mean) / std
    lo, hi = plan["pet_clip"]
    clipped = np.clip(np.nan_to_num(pet, nan=0.0), lo, hi)
    pet_standard = (clipped - float(plan["pet_mean"])) / float(plan["pet_std"])
    suv_log = np.log1p(np.maximum(pet, 0.0)) / max(float(plan["suv_log_cap"]), 1e-6)
    return np.stack([ct, pet_standard, np.clip(suv_log, 0.0, 1.0)], axis=0).astype(np.float32)


def build_edit_mask(selected_lesion: np.ndarray, config: dict) -> np.ndarray:
    """Fast rectangular edit ROI around exactly one selected lesion."""
    mask = np.zeros_like(selected_lesion, dtype=np.float32)
    points = np.argwhere(selected_lesion > 0)
    if not len(points):
        return mask
    spacing = np.asarray(config.get("target_spacing", (1.0, 1.0, 1.0)), dtype=np.float64)
    margin = np.maximum(1, np.ceil(float(config.get("prompt_edit_margin_mm", 24.0)) / spacing).astype(int))
    lo = np.maximum(0, points.min(0) - margin)
    hi = np.minimum(np.asarray(mask.shape), points.max(0) + 1 + margin)
    mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] = 1.0
    return mask


class SingleLesionDataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[CaseRecord], stage: str, config: dict, plan: dict,
                 prompt_type: str | None = None):
        self.records, self.stage, self.config, self.plan = list(records), stage, config, plan
        self.prompt_type = prompt_type
        self.patch_size = tuple(int(v) for v in config["patch_size"])
        self.target_spacing = tuple(float(v) for v in plan["target_spacing"])
        self.config = dict(config, target_spacing=self.target_spacing)
        self.epoch = 0
        self.entries: list[tuple[int, int]] = []
        if stage != "train":
            for record_index, record in enumerate(self.records):
                _, _, _, spacing = _load_case(record.ct_path, record.pet_path, record.label_path)
                lesions, _ = _component_catalog(record.label_path, tuple(spacing))
                lesions = list(lesions)
                limit = int(config.get("max_val_lesions_per_case", 0))
                if limit and len(lesions) > limit:
                    order = np.linspace(0, len(lesions) - 1, limit).round().astype(int)
                    lesions = [sorted(lesions, key=lambda x: x.volume_mm3)[i] for i in order]
                self.entries.extend((record_index, lesion.component_id) for lesion in lesions)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.config["samples_per_epoch"]) if self.stage == "train" else len(self.entries)

    def _select(self, index: int, rng: np.random.Generator):
        if self.stage == "train":
            record_index = int(rng.integers(0, len(self.records)))
            record = self.records[record_index]
            _, _, _, spacing = _load_case(record.ct_path, record.pet_path, record.label_path)
            lesions, _ = _component_catalog(record.label_path, tuple(spacing))
            if not lesions:
                raise RuntimeError(f"No lesion in {record.case_id}")
            volumes = np.asarray([x.volume_mm3 for x in lesions], dtype=np.float64)
            # Prefer small lesions without allowing one-voxel annotations to
            # dominate an epoch. The fourth-root schedule is deliberately mild.
            reference = float(np.median(volumes))
            weights = np.clip((reference / np.maximum(volumes, 1.0)) ** 0.25, 0.25, 4.0)
            weights /= weights.sum()
            lesion = lesions[int(rng.choice(len(lesions), p=weights))]
            return record_index, lesion.component_id
        return self.entries[index]

    def __getitem__(self, index: int):
        seed = self.config["seed"] + index * 104729 + (self.epoch * 1000003 if self.stage == "train" else 0)
        rng = np.random.default_rng(seed)
        automatic = (
            self.stage == "train"
            and self.prompt_type is None
            and rng.random() < float(self.config.get("automatic_training_probability", 0.0))
        )
        random_automatic_patch = (
            automatic
            and rng.random() < float(self.config.get("automatic_random_patch_probability", 0.0))
        )
        if random_automatic_patch:
            record_index, component_id = int(rng.integers(0, len(self.records))), 0
        else:
            record_index, component_id = self._select(index, rng)
        record = self.records[record_index]
        ct, pet, label, spacing = _load_case(record.ct_path, record.pet_path, record.label_path)
        coordinates = None
        if random_automatic_patch:
            body = np.isfinite(ct) & (ct > -950.0)
            center = []
            for axis in range(3):
                reduce_axes = tuple(i for i in range(3) if i != axis)
                occupied = np.flatnonzero(body.any(axis=reduce_axes))
                lo, hi = (int(occupied[0]), int(occupied[-1])) if occupied.size else (0, ct.shape[axis] - 1)
                center.append(float(rng.integers(lo, hi + 1)))
            center = tuple(center)
        else:
            lesions, coordinates = _component_catalog(record.label_path, tuple(spacing))
            lesion = next(x for x in lesions if x.component_id == component_id)
            center = lesion.center_zyx
        ct_p, pet_p, all_lesions = extract_physical_patch(
            [ct, pet, label], center, spacing, self.target_spacing,
            self.patch_size, orders=(1, 1, 0), fills=(-1000.0, 0.0, 0),
        )
        target = (all_lesions > 0.5).astype(np.uint8)
        if automatic:
            selected_target = np.zeros_like(target)
        else:
            selected_target = extract_component_patch(
                coordinates[component_id], center, spacing, self.target_spacing, self.patch_size,
            )
        prompt_type = (
            "automatic" if automatic else
            (self.prompt_type or self.config["prompt_types"][int(rng.integers(0, len(self.config["prompt_types"])))])
        )
        # The complete target supervises automatic segmentation. Only the one
        # selected lesion may generate a positive prompt; neighbouring lesions
        # are protected targets, never negative examples.
        prompts, route = build_prompts(
            selected_target, np.zeros_like(target), pet_p, prompt_type, rng, self.config,
            negative_exclusion=target,
        )
        edit_mask = build_edit_mask(selected_target, self.config) if not automatic else np.zeros_like(target, np.float32)
        image = normalize_modalities(ct_p, pet_p, self.plan)
        output = {
            "image": torch.from_numpy(image),
            "click": torch.from_numpy(prompts["click"]),
            "bbox": torch.from_numpy(prompts["bbox"]),
            "scribble": torch.from_numpy(prompts["scribble"]),
            "prev_prob": torch.zeros((1, *self.patch_size), dtype=torch.float32),
            "route": torch.tensor(route, dtype=torch.float32),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "selected_target": torch.from_numpy(selected_target[None].astype(np.float32)),
            "edit_mask": torch.from_numpy(edit_mask[None].astype(np.float32)),
            "volume_mm3": torch.tensor(float(max(int(selected_target.sum()), int(target.sum() if automatic else 0))) *
                                        float(np.prod(self.target_spacing)), dtype=torch.float32),
            "spacing": torch.tensor(self.target_spacing, dtype=torch.float32),
            "case_id": record.case_id,
            "component_id": torch.tensor(component_id, dtype=torch.int64),
            "prompt_type": prompt_type,
        }
        return output


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
