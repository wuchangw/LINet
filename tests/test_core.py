from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_config
from data import (CaseRecord, SingleLesionDataset, build_prompts, connected_lesions,
                  extract_component_patch, normalize_modalities)
from losses import physical_volume_prompt_tversky_loss
from model import build_model
from engine import add_corrective_clicks, add_prediction_error_prompts
from inference import (_axis_starts, _prompt_influence_mask, _round_centers,
                       predict_auto_volume, predict_roi)


class DataTests(unittest.TestCase):
    def test_connected_components_use_physical_volume(self):
        label = np.zeros((12, 12, 12), dtype=np.uint8)
        label[1, 1, 1] = 1; label[7:9, 7:9, 7:9] = 1
        components, lesions = connected_lesions(label, (2.0, 3.0, 4.0))
        self.assertEqual(len(lesions), 2)
        self.assertEqual(sorted(round(x.volume_mm3) for x in lesions), [24, 192])
        self.assertEqual(int(components.max()), 2)

    def test_all_prompt_types_are_clean_and_routed(self):
        target = np.zeros((24, 24, 24), dtype=np.uint8); target[9:15, 9:15, 9:15] = 1
        other = np.zeros_like(target); other[2:5, 2:5, 2:5] = 1
        pet = other.astype(np.float32) * 8.0
        config = get_config(); config["target_spacing"] = (3.0, 3.0, 3.0)
        for index, prompt_type in enumerate(("click", "bbox", "scribble")):
            prompts, route = build_prompts(target, other, pet, prompt_type, np.random.default_rng(10 + index), config)
            self.assertEqual(route[index], 1.0)
            self.assertGreater(float(np.abs(prompts[prompt_type]).sum()), 0.0)
            for other_type in set(prompts) - {prompt_type}:
                self.assertEqual(float(np.abs(prompts[other_type]).sum()), 0.0)

    def test_automatic_mode_has_no_prompt_signal(self):
        shape = (16, 16, 16)
        prompts, route = build_prompts(
            np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8),
            np.zeros(shape, dtype=np.float32), "automatic", np.random.default_rng(1), get_config(),
        )
        self.assertEqual(route, (0.0, 0.0, 0.0))
        self.assertEqual(sum(float(np.abs(value).sum()) for value in prompts.values()), 0.0)

    def test_positive_clicks_cover_distinct_lesions_first(self):
        target = np.zeros((32, 32, 32), dtype=np.uint8)
        target[3:6, 3:6, 3:6] = 1
        target[14:18, 14:18, 14:18] = 1
        target[25:28, 25:28, 25:28] = 1
        config = get_config(); config["clicks_pos"] = 3; config["clicks_neg"] = 0
        prompts, _ = build_prompts(target, np.zeros_like(target), np.zeros_like(target, dtype=np.float32),
                                   "click", np.random.default_rng(7), config)
        seeds = prompts["click"][0] > 0.999
        _, lesions = connected_lesions(seeds, (1.0, 1.0, 1.0))
        self.assertEqual(len(lesions), 3)

    def test_distributed_prompts_create_multiple_rois(self):
        first = {"pos": [[5, 5, 5], [100, 100, 100]]}
        second = {"pos_scribble": [[50, 50, 50], [52, 50, 50]]}
        self.assertEqual(_round_centers(first), [(5.0, 5.0, 5.0), (100.0, 100.0, 100.0)])
        self.assertEqual(_round_centers(second), [(51.0, 50.0, 50.0)])

    def test_negative_prompts_can_anchor_false_positive_refinement(self):
        self.assertEqual(_round_centers({"neg": [[8, 9, 10]]}), [(8.0, 9.0, 10.0)])
        self.assertEqual(_round_centers({"neg_scribble": [[20, 20, 20], [22, 20, 20]]}), [(21.0, 20.0, 20.0)])

    def test_sliding_window_starts_cover_last_voxel(self):
        starts = _axis_starts(length=101, window=32, overlap=0.5)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 69)
        self.assertTrue(all(b > a for a, b in zip(starts, starts[1:])))

    def test_prompt_influence_keeps_remote_automatic_component_unchanged(self):
        probability = np.zeros((32, 32, 32), dtype=np.float32)
        probability[3:9, 3:9, 3:9] = 0.9
        probability[22:27, 22:27, 22:27] = 0.9
        config = get_config(); config["prompt_refinement_radius_mm"] = 2.0
        influence = _prompt_influence_mask(
            {"neg": [[5, 5, 5]]}, [0, 0, 0], [32, 32, 32], [32, 32, 32],
            (1.0, 1.0, 1.0), probability, config,
        )
        self.assertTrue(influence[3:9, 3:9, 3:9].all())
        self.assertFalse(influence[22:27, 22:27, 22:27].any())

    def test_automatic_sliding_window_covers_complete_volume(self):
        class ConstantAutoModel(torch.nn.Module):
            def forward_auto(self, image):
                return image[:, :1] * 0.0

        shape = (20, 20, 20)
        config = get_config(); config.update({"device": "cpu", "patch_size": (16, 16, 16)})
        plan = {"pet_clip": [0.0, 1.0], "pet_mean": 0.0, "pet_std": 1.0,
                "suv_log_cap": float(np.log1p(1.0))}
        probability, windows = predict_auto_volume(
            ConstantAutoModel(), np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32),
            (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (16, 16, 16), plan, config,
            torch.float16, overlap=0.5,
        )
        self.assertEqual(probability.shape, shape)
        self.assertEqual(windows, 8)
        self.assertTrue(np.allclose(probability, 0.5, atol=1e-5))

    def test_automatic_training_accepts_background_only_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("ct.nii.gz", "pet.nii.gz", "label.nii.gz")]
            for path in paths:
                nib.save(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.float32), np.eye(4)), path)
            record = CaseRecord("negative", *(str(path) for path in paths))
            config = get_config(); config.update({
                "patch_size": (8, 8, 8), "automatic_training_probability": 1.0,
                "automatic_random_patch_probability": 1.0,
            })
            plan = {"target_spacing": [1.0, 1.0, 1.0], "pet_clip": [0.0, 1.0],
                    "pet_mean": 0.0, "pet_std": 1.0, "suv_log_cap": float(np.log1p(1.0))}
            sample = SingleLesionDataset([record], "train", config, plan)[0]
            self.assertEqual(sample["prompt_type"], "automatic")
            self.assertEqual(int(sample["component_id"]), 0)
            self.assertEqual(float(sample["target"].sum()), 0.0)
            self.assertEqual(float(sample["route"].sum()), 0.0)

    def test_suv_channel_preserves_absolute_order(self):
        plan = {"pet_clip": [0.0, 20.0], "pet_mean": 1.0, "pet_std": 2.0, "suv_log_cap": float(np.log1p(20.0))}
        ct = np.zeros((4, 4, 4), dtype=np.float32)
        pet = np.zeros_like(ct); pet[1, 1, 1] = 2.0; pet[2, 2, 2] = 10.0
        image = normalize_modalities(ct, pet, plan)
        self.assertGreater(float(image[2, 2, 2, 2]), float(image[2, 1, 1, 1]))

    def test_single_voxel_survives_physical_downsampling(self):
        result = extract_component_patch(np.asarray([[10, 10, 10]], dtype=np.int32), (10, 10, 10),
                                         (2.0, 2.0, 2.0), (4.0, 4.0, 4.0), (8, 8, 8))
        self.assertEqual(int(result.sum()), 1)


class LossAndModelTests(unittest.TestCase):
    @staticmethod
    def _probability_like_shape(dst_slices):
        return tuple(int(s.stop - s.start) for s in dst_slices)

    def test_correction_uses_actual_previous_probability(self):
        shape = (8, 8, 8); probability = torch.rand(1, 1, *shape)
        batch = {
            "click": torch.zeros(1, 2, *shape), "bbox": torch.zeros(1, 2, *shape),
            "scribble": torch.zeros(1, 2, *shape), "prev_prob": torch.zeros(1, 1, *shape),
            "route": torch.zeros(1, 3), "target": torch.zeros(1, 1, *shape),
        }
        batch["target"][0, 0, 3:5, 3:5, 3:5] = 1
        corrected = add_corrective_clicks(batch, probability, get_config(), 0, 0)
        self.assertTrue(torch.equal(corrected["prev_prob"], probability))
        self.assertEqual(float(corrected["route"][0, 0]), 1.0)
        self.assertGreater(float(corrected["click"][0, 0].sum()), 0.0)

    def test_correction_spreads_over_missed_components(self):
        shape = (20, 20, 20)
        batch = {
            "click": torch.zeros(1, 2, *shape), "bbox": torch.zeros(1, 2, *shape),
            "scribble": torch.zeros(1, 2, *shape), "prev_prob": torch.zeros(1, 1, *shape),
            "route": torch.zeros(1, 3), "target": torch.zeros(1, 1, *shape),
        }
        batch["target"][0, 0, 2:5, 2:5, 2:5] = 1
        batch["target"][0, 0, 14:17, 14:17, 14:17] = 1
        corrected = add_corrective_clicks(batch, torch.zeros(1, 1, *shape), get_config(), 0, 0)
        seeds = corrected["click"][0, 0].numpy() > 0.999
        _, lesions = connected_lesions(seeds, (1.0, 1.0, 1.0))
        self.assertEqual(len(lesions), 2)

    def test_false_positive_creates_negative_prompt_for_selected_roi(self):
        shape = (16, 16, 16)
        batch = {
            "click": torch.zeros(1, 2, *shape), "bbox": torch.zeros(1, 2, *shape),
            "scribble": torch.zeros(1, 2, *shape), "route": torch.tensor([[1.0, 0.0, 0.0]]),
            "target": torch.zeros(1, 1, *shape), "selected_target": torch.zeros(1, 1, *shape),
            "edit_mask": torch.ones(1, 1, *shape),
        }
        batch["target"][0, 0, 3:6, 3:6, 3:6] = 1
        batch["selected_target"].copy_(batch["target"])
        probability = torch.zeros(1, 1, *shape)
        probability[0, 0, 10:13, 10:13, 10:13] = 1
        corrected = add_prediction_error_prompts(batch, probability, get_config(), 0, 0)
        self.assertGreater(float(corrected["click"][0, 1].sum()), 0.0)

    def test_joint_forward_encodes_once_and_preserves_outside_edit_roi(self):
        model = build_model(get_config()); model.eval()
        shape = (32, 32, 32)
        image = torch.randn(1, 3, *shape)
        click = torch.zeros(1, 2, *shape); click[0, 0, 16, 16, 16] = 1
        edit = torch.zeros(1, 1, *shape); edit[..., 12:21, 12:21, 12:21] = 1
        calls = {"encoder": 0}
        original = model.image_encoder.forward

        def counted(*args, **kwargs):
            calls["encoder"] += 1
            return original(*args, **kwargs)

        model.image_encoder.forward = counted
        try:
            with torch.no_grad():
                automatic, final = model.forward_joint(
                    image, click, torch.zeros_like(click), torch.zeros_like(click),
                    torch.tensor([[1.0, 0.0, 0.0]]), edit,
                )
        finally:
            model.image_encoder.forward = original
        self.assertEqual(calls["encoder"], 1)
        outside = edit == 0
        self.assertTrue(torch.equal(automatic[0][outside], final[0][outside]))

    def test_small_lesion_has_larger_custom_loss(self):
        logits = torch.zeros(1, 1, 8, 8, 8)
        target = torch.zeros_like(logits); target[..., 3:5, 3:5, 3:5] = 1
        small = physical_volume_prompt_tversky_loss(logits, target, torch.tensor([50.0]), None, None)
        large = physical_volume_prompt_tversky_loss(logits, target, torch.tensor([5000.0]), None, None)
        self.assertTrue(torch.isfinite(small)); self.assertGreater(float(small), float(large))

    def test_model_shapes_parameter_limit_and_checkpoint(self):
        config = get_config(); model = build_model(config)
        self.assertLess(model.parameter_count(), 30_000_000)
        shape = (32, 32, 32)
        image = torch.randn(1, 3, *shape); prompt = torch.zeros(1, 2, *shape)
        prompt[:, 0, 16, 16, 16] = 1
        route = torch.tensor([[1.0, 0.0, 0.0]])
        with torch.no_grad():
            outputs = model(image, prompt, torch.zeros_like(prompt), torch.zeros_like(prompt),
                            torch.zeros(1, 1, *shape), route)
        self.assertEqual([tuple(x.shape[2:]) for x in outputs], [(32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"; torch.save(model.state_dict(), path)
            clone = build_model(config); clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)

    def test_three_experts_do_not_share_parameters(self):
        model = build_model(get_config())
        click = {id(x) for x in model.prompt_experts.click.parameters()}
        bbox = {id(x) for x in model.prompt_experts.bbox.parameters()}
        scribble = {id(x) for x in model.prompt_experts.scribble.parameters()}
        self.assertFalse(click & bbox); self.assertFalse(click & scribble); self.assertFalse(bbox & scribble)

    def test_moving_click_moves_prompt_response(self):
        shape = (32, 32, 32); blank = torch.zeros(1, 2, *shape)
        first, second = blank.clone(), blank.clone()
        first[:, 0, 8, 8, 8] = 1.0; second[:, 0, 24, 24, 24] = 1.0
        route = torch.tensor([[1.0, 0.0, 0.0]])
        logits = [torch.zeros(1, 1, *shape)]
        a = build_model(get_config())._add_spatial_prompt_prior(logits, first, blank, blank, route)[0]
        b = build_model(get_config())._add_spatial_prompt_prior(logits, second, blank, blank, route)[0]
        delta_a = float((a - b)[0, 0, 8, 8, 8]); delta_b = float((b - a)[0, 0, 24, 24, 24])
        self.assertGreater(delta_a, 1.0); self.assertGreater(delta_b, 1.0)

    def test_forward_auto_bypasses_prompt_and_feedback_branches(self):
        config = get_config(); model = build_model(config); model.eval()
        shape = (32, 32, 32)
        image = torch.randn(1, 3, *shape)
        with torch.no_grad():
            auto_a = model.forward_auto(image)
            auto_b = model.forward_auto(image)
            blank = torch.zeros(1, 2, *shape)
            refine = model.forward_refine(image, blank, blank, blank,
                                          torch.zeros(1, 1, *shape), torch.zeros(1, 3))
        self.assertEqual([tuple(x.shape[2:]) for x in auto_a],
                         [(32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)])
        # The automatic pass depends on the image only.
        self.assertTrue(torch.allclose(auto_a[0], auto_b[0]))
        # With zero prompts and zero feedback the refinement pass is
        # mathematically identical to the pure automatic pass: the prompt
        # branch can never leak signal into the automatic result.
        self.assertTrue(torch.allclose(auto_a[0], refine[0]))
        # A real click activates the separate refinement path.
        with torch.no_grad():
            clicked = blank.clone(); clicked[:, 0, 16, 16, 16] = 1.0
            refine_click = model.forward_refine(image, clicked, blank, blank,
                                                torch.zeros(1, 1, *shape),
                                                torch.tensor([[1.0, 0.0, 0.0]]))
        self.assertFalse(torch.allclose(auto_a[0], refine_click[0]))

    def test_predict_roi_runs_exactly_one_refinement_pass(self):
        config = get_config(); config["device"] = "cpu"
        model = build_model(config); model.eval()
        calls = {"refine": 0, "auto": 0}
        original_refine, original_auto = model.forward_refine, model.forward_auto

        def counting_refine(*args, **kwargs):
            calls["refine"] += 1
            return original_refine(*args, **kwargs)

        def counting_auto(*args, **kwargs):
            calls["auto"] += 1
            return original_auto(*args, **kwargs)

        model.forward_refine = counting_refine; model.forward_auto = counting_auto
        shape = (40, 40, 40)
        ct = np.zeros(shape, dtype=np.float32); pet = np.zeros_like(ct)
        initial = np.zeros(shape, dtype=np.float32); initial[10:14, 10:14, 10:14] = 0.9
        plan = {"pet_clip": [0.0, 1.0], "pet_mean": 0.0, "pet_std": 1.0,
                "suv_log_cap": float(np.log1p(1.0))}
        prompt = {"pos": [[20, 20, 20]]}
        try:
            refined, influence, dst = predict_roi(
                model, ct, pet, (20.0, 20.0, 20.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
                (32, 32, 32), prompt, plan, config, torch.float32,
                initial_probability=initial,
            )
        finally:
            model.forward_refine = original_refine; model.forward_auto = original_auto
        # Exactly one prompt-update forward pass, and the automatic path stays idle.
        self.assertEqual(calls["refine"], 1)
        self.assertEqual(calls["auto"], 0)
        # The influence mask covers the ROI overlap region (same shape as refined).
        self.assertEqual(influence.shape, refined.shape)
        self.assertEqual(influence.shape, self._probability_like_shape(dst))
        # The prompted voxel is inside the influence region.
        self.assertTrue(influence.any())


if __name__ == "__main__":
    unittest.main()
