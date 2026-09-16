from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import (CACHE_VERSION, BankPatchDataset, CachedPatchDataset, VolumeCache,
                   build_cache, build_patch_bank, cache_is_usable, cache_dir_for,
                   target_shape, verify)
from config import get_config
from data import CaseRecord, SingleLesionDataset, build_registry, load_or_create_plan


def _synthetic_case(root: Path, case_id: str = "case_a", shape=(40, 40, 40), spacing=(2.0, 2.0, 2.0)):
    """A tiny but valid case: body-like CT, noisy PET with two hot lesions."""
    rng = np.random.default_rng(0)
    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[6:34, 6:34, 6:34] = 40.0
    # Background uptake must not be a constant zero, otherwise the PET
    # normalisation percentiles degenerate and the channel collapses.
    pet = np.zeros(shape, dtype=np.float32)
    pet[6:34, 6:34, 6:34] = 0.8 + 0.2 * rng.random((28, 28, 28)).astype(np.float32)
    label = np.zeros(shape, dtype=np.uint8)
    label[14:20, 14:20, 14:20] = 1
    label[26:30, 26:30, 26:30] = 1
    pet[label > 0] = 9.0
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    # nnU-Net naming: images use the modality suffix, labels are bare case ids.
    paths = {}
    for key, suffix, array, dtype in (("ct", "ct", ct, np.float32), ("pet", "pet", pet, np.float32),
                                      ("label", "", label, np.uint8)):
        name = f"{case_id}_{suffix}.nii.gz" if suffix else f"{case_id}.nii.gz"
        path = root / name
        nib.save(nib.Nifti1Image(array.astype(dtype), affine), str(path))
        paths[key] = str(path)
    return CaseRecord(case_id, paths["ct"], paths["pet"], paths["label"])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.record = _synthetic_case(self.root)
        self.cache_dir = self.root / "cache"
        self.config = get_config()
        self.config.update({
            "images_dir": str(self.root), "labels_dir": str(self.root),
            "split_json": str(self.root / "split.json"),
            "plan_path": str(self.root / "plan.json"),
            "output_dir": str(self.root / "runs"),
            "patch_size": (16, 16, 16), "num_workers": 0,
            "ct_suffix": "ct", "pet_suffix": "pet", "allow_missing": True,
        })
        (self.root / "split.json").write_text(json.dumps([
            {"train": [self.record.case_id], "val": [self.record.case_id]}
        ]), encoding="utf-8")
        self.plan = load_or_create_plan([self.record], self.config["plan_path"])

    def tearDown(self):
        self.temp.cleanup()

    def test_target_shape_keeps_physical_extent(self):
        # 40 voxels at 2mm resampled to 4mm spacing covers the same 80mm.
        self.assertEqual(target_shape((40, 40, 40), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0)), (20, 20, 20))

    def test_build_cache_writes_volumes_and_catalogue(self):
        meta = build_cache(self.config, self.cache_dir, records=[self.record])
        self.assertTrue(cache_is_usable(self.cache_dir))
        self.assertEqual(meta["version"], CACHE_VERSION)
        store = VolumeCache(self.cache_dir)
        catalog = store.catalog(self.record.case_id)
        self.assertEqual(len(catalog["lesions"]), 2)
        volumes = store.volumes(self.record.case_id)
        self.assertEqual(volumes["ct"].shape, tuple(catalog["shape"]))
        self.assertEqual(volumes["label"].dtype, np.uint8)

    def test_prompt_cache_limit_and_single_selected_lesion(self):
        self.config["cached_lesions_per_case"] = 1
        self.config["automatic_training_probability"] = 0.0
        build_cache(self.config, self.cache_dir, records=[self.record])
        catalog = VolumeCache(self.cache_dir).catalog(self.record.case_id)
        self.assertEqual(len(catalog["lesions"]), 2)
        self.assertEqual(len(catalog["prompt_component_ids"]), 1)
        registry = build_registry(self.config)
        dataset = CachedPatchDataset(
            registry["train"], "train", self.config, self.plan,
            prompt_type="click", cache_dir=self.cache_dir,
        )
        sample = dataset[0]
        selected = sample["selected_target"][0].numpy() > 0
        all_lesions = sample["target"][0].numpy() > 0
        positive_seed = sample["click"][0].numpy() > 0.999
        negative_seed = sample["click"][1].numpy() > 0.999
        self.assertTrue(selected.any())
        self.assertTrue(np.all(selected <= all_lesions))
        self.assertTrue(np.all(positive_seed <= selected))
        self.assertFalse(bool((negative_seed & all_lesions).any()))

    def test_cached_dataset_matches_sample_contract(self):
        build_cache(self.config, self.cache_dir, records=[self.record])
        registry = build_registry(self.config)
        original = SingleLesionDataset(registry["train"], "train", self.config, self.plan)
        cached = CachedPatchDataset(registry["train"], "train", self.config, self.plan,
                                    cache_dir=self.cache_dir)
        for dataset in (original, cached):
            dataset.set_epoch(0)
        a, b = original[0], cached[0]
        self.assertEqual(set(a.keys()), set(b.keys()))
        for key in a:
            if hasattr(a[key], "shape"):
                self.assertEqual(tuple(a[key].shape), tuple(b[key].shape), key)
        patch = tuple(self.config["patch_size"])
        self.assertEqual(tuple(b["image"].shape), (3, *patch))
        self.assertEqual(tuple(b["target"].shape), (1, *patch))
        self.assertTrue(np.isfinite(b["image"].numpy()).all())

    def test_cached_patch_is_self_consistent_with_its_label(self):
        build_cache(self.config, self.cache_dir, records=[self.record])
        store = VolumeCache(self.cache_dir)
        catalog = store.catalog(self.record.case_id)
        centre = tuple(catalog["lesions"][0]["center_zyx"])
        image, lesions = store.patch(self.record.case_id, centre)
        self.assertEqual(image.shape, (3, *self.config["patch_size"]))
        self.assertGreater(int(lesions.sum()), 0)
        # The PET hotspot channel must light up exactly where the lesion is.
        pet_channel = image[1]
        self.assertGreater(float(pet_channel[lesions > 0].mean()), float(pet_channel[lesions == 0].mean()))

    def test_verify_reports_geometry_and_catalogue(self):
        build_cache(self.config, self.cache_dir, records=[self.record])
        report = verify(self.config, self.cache_dir, samples=3)
        self.assertIn("geometry", report)
        self.assertIn("catalogue", report)
        self.assertEqual(report["catalogue"][0]["source_lesions"], 2)
        self.assertLessEqual(report["catalogue"][0]["cached_lesions"], 2)
        self.assertTrue(all(row["lesion_overlap_dice"] > 0.3 for row in report["geometry"]))

    def test_patch_bank_round_trip(self):
        build_cache(self.config, self.cache_dir, records=[self.record])
        result = build_patch_bank(self.config, self.cache_dir, patches_per_lesion=2, bank_limit=4)
        self.assertEqual(result["patches"], 4)
        registry = build_registry(self.config)
        dataset = BankPatchDataset(registry["train"], "train", self.config, self.plan,
                                   cache_dir=self.cache_dir)
        dataset.set_epoch(0)
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (3, *self.config["patch_size"]))
        self.assertIsInstance(sample["prompt_type"], str)

    def test_missing_cache_is_detected(self):
        self.assertFalse(cache_is_usable(self.cache_dir))
        self.assertTrue(str(cache_dir_for(self.config)).endswith("cache"))


if __name__ == "__main__":
    unittest.main()
