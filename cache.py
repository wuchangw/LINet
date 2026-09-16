"""Offline patch cache for LINet.

The training loop used to resample every patch on the fly: each sample cropped a
window out of the original-resolution NIfTI volume and ran ``scipy.ndimage.zoom``
on it (order 1 for CT/PET, order 0 for labels). That work is identical every
epoch, so it is done once here and stored on disk.

Layout produced by ``build``::

    <cache_dir>/
        cache_meta.json          # version, spacing, patch size, case list, plan
        volumes/<case_id>_*.npy  # mmap CT/PET/label/instance arrays
        catalog/<case_id>.json   # all lesions plus at most N prompt-lesion ids
        bank/
            index.json           # one entry per stored patch
            patch_<n>.npz        # image (3, D, H, W) fp16, lesions (1, D, H, W) uint8

``CachedPatchDataset`` slices the cached arrays with plain numpy (no zoom, no
NIfTI decompression). Automatic targets contain every lesion in the crop, while
exactly one cached instance is allowed to generate a random prompt.

CLI::

    python cache.py build  --profile smoke [--patches-per-lesion 8] [--bank-limit 512]
    python cache.py verify --profile smoke
    python cache.py info   --profile smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage

from config import get_config
from data import (CaseRecord, _load_case, build_edit_mask, build_prompts, build_registry,
                  connected_lesions, load_or_create_plan, normalize_modalities, seed_everything)

CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# geometry helpers (same physical conventions as data.extract_physical_patch)
# ---------------------------------------------------------------------------

def target_shape(source_shape: Sequence[int], source_spacing, target_spacing) -> tuple[int, int, int]:
    """Voxel shape after resampling a whole volume to the target spacing."""
    return tuple(max(2, int(round(n * s / t))) for n, s, t in zip(source_shape, source_spacing, target_spacing))


def _zoom_volume(array: np.ndarray, shape: Sequence[int], order: int) -> np.ndarray:
    """Resample a whole volume to a new voxel grid without changing its extent."""
    result = ndimage.zoom(array, [float(d) / float(s) for s, d in zip(array.shape, shape)],
                          order=order, mode="nearest", prefilter=order > 1)
    fixed = np.zeros(tuple(shape), dtype=result.dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(result.shape, shape))
    fixed[slices] = result[slices]
    return fixed


def _crop_pad(array: np.ndarray, center: Sequence[float], size: Sequence[int], value: float) -> np.ndarray:
    """Zero-copy-friendly crop in target-spacing voxel space (fills out of bounds)."""
    starts = [int(round(c - s / 2.0)) for c, s in zip(center, size)]
    ends = [start + int(s) for start, s in zip(starts, size)]
    src = tuple(slice(max(0, a), min(n, b)) for a, b, n in zip(starts, ends, array.shape))
    crop = array[src]
    pads = [(max(0, -a), max(0, b - n)) for a, b, n in zip(starts, ends, array.shape)]
    if any(pad for pad in pads):
        return np.pad(crop, pads, mode="constant", constant_values=value)
    return crop


# ---------------------------------------------------------------------------
# build: volumes + lesion catalog
# ---------------------------------------------------------------------------

def _prompt_component_ids(lesions, limit: int) -> list[int]:
    """Choose a stable size-stratified subset instead of the first N lesions."""
    ordered = sorted(lesions, key=lambda lesion: lesion.volume_mm3)
    if limit <= 0 or len(ordered) <= limit:
        return [int(lesion.component_id) for lesion in ordered]
    indices = np.linspace(0, len(ordered) - 1, int(limit)).round().astype(int)
    return [int(ordered[index].component_id) for index in np.unique(indices)]


def build_case(record: CaseRecord, plan: dict, out_dir: Path, dtype: str = "float16",
               max_prompt_lesions: int = 0) -> dict:
    ct, pet, label, spacing = _load_case(record.ct_path, record.pet_path, record.label_path)
    shape = target_shape(ct.shape, spacing, plan["target_spacing"])
    ct_r = _zoom_volume(ct, shape, order=1)
    pet_r = _zoom_volume(pet, shape, order=1)
    label_r = (_zoom_volume(label.astype(np.uint8), shape, order=0) > 0).astype(np.uint8)

    volumes_dir = out_dir / "volumes"; catalog_dir = out_dir / "catalog"
    volumes_dir.mkdir(parents=True, exist_ok=True); catalog_dir.mkdir(parents=True, exist_ok=True)
    # Lesions are catalogued on the cached label, so centres are already in
    # target-spacing voxel coordinates and need no further mapping.
    instances, lesions = connected_lesions(label_r, tuple(plan["target_spacing"]))
    # Uncompressed arrays can be memory-mapped by every DataLoader worker. This
    # removes repeated gzip/NPZ decompression without duplicating image patches.
    np.save(volumes_dir / f"{record.case_id}_ct.npy", ct_r.astype(dtype))
    np.save(volumes_dir / f"{record.case_id}_pet.npy", pet_r.astype(dtype))
    np.save(volumes_dir / f"{record.case_id}_label.npy", label_r)
    np.save(volumes_dir / f"{record.case_id}_instances.npy", instances.astype(np.uint16))
    prompt_ids = _prompt_component_ids(lesions, int(max_prompt_lesions))
    catalog = {
        "case_id": record.case_id,
        "shape": list(int(v) for v in shape),
        "source_shape": list(int(v) for v in ct.shape),
        "source_spacing": [float(v) for v in spacing],
        "prompt_component_ids": prompt_ids,
        "lesions": [
            {"component_id": lesion.component_id,
             "center_zyx": [float(v) for v in lesion.center_zyx],
             "voxels": int(lesion.voxels),
             "volume_mm3": float(lesion.volume_mm3)}
            for lesion in lesions
        ],
    }
    (catalog_dir / f"{record.case_id}.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def build_cache(config: dict, out_dir: Path | None = None, records: Sequence[CaseRecord] | None = None) -> dict:
    out_dir = Path(out_dir or cache_dir_for(config))
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_registry(config)
    plan = load_or_create_plan(registry["train"], config["plan_path"])
    if records is None:
        records = list(registry["train"]) + list(registry["val"])
    started = time.time()
    catalogs = {}
    for index, record in enumerate(records, start=1):
        catalogs[record.case_id] = build_case(
            record, plan, out_dir,
            max_prompt_lesions=int(config.get("cached_lesions_per_case", 0)),
        )
        print(f"  [{index}/{len(records)}] {record.case_id}: "
              f"{len(catalogs[record.case_id]['lesions'])} lesion(s), shape={catalogs[record.case_id]['shape']}")
    meta = {
        "version": CACHE_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - started, 2),
        "target_spacing": [float(v) for v in plan["target_spacing"]],
        "patch_size": [int(v) for v in config["patch_size"]],
        "cached_lesions_per_case": int(config.get("cached_lesions_per_case", 0)),
        "plan_path": str(config["plan_path"]),
        "cases": [record.case_id for record in records],
        "case_shapes": {key: value["shape"] for key, value in catalogs.items()},
    }
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Volume cache written to {out_dir} in {meta['seconds']}s")
    return meta


# ---------------------------------------------------------------------------
# build: pre-sliced patch bank
# ---------------------------------------------------------------------------

def build_patch_bank(config: dict, out_dir: Path | None = None, patches_per_lesion: int = 8,
                     bank_limit: int = 512, seed: int = 0) -> dict:
    """Materialise ready-to-use training patches once, so training only reads
    them back. Every patch stores the 3-channel image plus the every-lesion mask
    of the window; prompts are still generated on the fly from the stored mask
    so click/bbox/scribble randomness is preserved."""
    out_dir = Path(out_dir or cache_dir_for(config))
    meta_path = out_dir / "cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No volume cache at {out_dir}; run: python cache.py build --profile <profile>")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    preprocess = resolve_plan(meta, out_dir)
    patch_size = tuple(int(v) for v in config["patch_size"])
    rng = np.random.default_rng(seed)
    store = VolumeCache(out_dir)
    bank_dir = out_dir / "bank"; bank_dir.mkdir(parents=True, exist_ok=True)
    index, count = [], 0
    for case_id in meta["cases"]:
        catalog = json.loads((out_dir / "catalog" / f"{case_id}.json").read_text(encoding="utf-8"))
        volumes = store.volumes(case_id)
        ct, pet, label, instances = volumes["ct"], volumes["pet"], volumes["label"], volumes["instances"]
        allowed = set(catalog.get("prompt_component_ids", []))
        for lesion in catalog["lesions"]:
            if allowed and int(lesion["component_id"]) not in allowed:
                continue
            base = lesion["center_zyx"]
            for _ in range(int(patches_per_lesion)):
                if count >= int(bank_limit):
                    break
                jitter = rng.integers(-4, 5, size=3) * np.asarray(patch_size) / 16.0
                center = tuple(float(v) for v in np.asarray(base) + jitter)
                ct_p = _crop_pad(ct, center, patch_size, -1000.0).astype(np.float32)
                pet_p = _crop_pad(pet, center, patch_size, 0.0).astype(np.float32)
                lesion_p = _crop_pad(label, center, patch_size, 0.0)
                selected_p = (_crop_pad(instances, center, patch_size, 0.0) == int(lesion["component_id"]))
                image = normalize_modalities(ct_p, pet_p, preprocess)
                name = f"patch_{count:06d}.npz"
                component_ids, depth = component_depth_maps(selected_p)
                # float16 halves the bank size; clip first so a degenerate
                # normalisation cannot produce inf in the cast.
                np.savez_compressed(bank_dir / name,
                                    image=np.clip(image, -60000.0, 60000.0).astype(np.float16),
                                    lesions=lesion_p.astype(np.uint8),
                                    selected=selected_p.astype(np.uint8),
                                    component_ids=component_ids.astype(np.uint16),
                                    depth=depth.astype(np.float16))
                index.append({"case_id": case_id, "file": name,
                              "center_zyx": [float(v) for v in center],
                              "component_id": int(lesion["component_id"]),
                              "volume_mm3": float(lesion["volume_mm3"])})
                count += 1
            if count >= int(bank_limit):
                break
        if count >= int(bank_limit):
            break
    (bank_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Patch bank: {count} patch(es) in {bank_dir}")
    return {"patches": count, "index": str(bank_dir / "index.json")}


def component_depth_maps(lesions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of a patch mask plus a per-component depth map.

    The depth map is the Euclidean distance transform *within each component*,
    which is exactly what prompt sampling needs to pick the deepest voxel of a
    lesion. Computing it here removes the repeated scipy EDT calls from the
    training loop.
    """
    component_ids, count = ndimage.label(lesions.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    depth = np.zeros(lesions.shape, dtype=np.float32)
    for component_id in range(1, int(count) + 1):
        component = component_ids == component_id
        if component.any():
            depth[component] = ndimage.distance_transform_edt(component)[component]
    return component_ids.astype(np.uint16), depth


def resolve_plan(meta: dict, out_dir: Path) -> dict:
    """The preprocessing plan referenced by the cache (pet_clip / mean / std)."""
    for candidate in (Path(meta.get("plan_path", "")), Path(out_dir) / "preprocess_plan.json"):
        if str(candidate) and candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError("Preprocessing plan not found for the cache; rebuild with cache.py build")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def cache_dir_for(config: dict) -> Path:
    return Path(config["output_dir"]) / "cache"


def cache_is_usable(cache_dir: Path) -> bool:
    meta_path = Path(cache_dir) / "cache_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(meta.get("version", -1)) == CACHE_VERSION


class VolumeCache:
    """Read-only access to the cached volumes and lesion catalogues."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "cache_meta.json").read_text(encoding="utf-8"))
        self.plan = resolve_plan(self.meta, self.dir)
        self.target_spacing = tuple(float(v) for v in self.meta["target_spacing"])
        self.patch_size = tuple(int(v) for v in self.meta["patch_size"])
        self._volumes: dict[str, dict] = {}
        self._catalogs: dict[str, dict] = {}

    def catalog(self, case_id: str) -> dict:
        if case_id not in self._catalogs:
            self._catalogs[case_id] = json.loads(
                (self.dir / "catalog" / f"{case_id}.json").read_text(encoding="utf-8")
            )
        return self._catalogs[case_id]

    def volumes(self, case_id: str):
        if case_id not in self._volumes:
            base = self.dir / "volumes"
            paths = {key: base / f"{case_id}_{key}.npy" for key in ("ct", "pet", "label", "instances")}
            if all(path.exists() for path in paths.values()):
                self._volumes[case_id] = {
                    key: np.load(path, mmap_mode="r") for key, path in paths.items()
                }
            else:
                # Read compatibility for version-1 caches during diagnostics.
                loaded = np.load(base / f"{case_id}.npz")
                payload = {key: loaded[key] for key in loaded.files}
                payload["instances"] = ndimage.label(
                    payload["label"] > 0, structure=np.ones((3, 3, 3), dtype=np.uint8),
                )[0].astype(np.uint16)
                self._volumes[case_id] = payload
        return self._volumes[case_id]

    def patch(self, case_id: str, center) -> tuple[np.ndarray, np.ndarray]:
        """Slice a training patch straight out of the cached arrays."""
        volumes = self.volumes(case_id)
        ct = _crop_pad(volumes["ct"].astype(np.float32), center, self.patch_size, -1000.0)
        pet = _crop_pad(volumes["pet"].astype(np.float32), center, self.patch_size, 0.0)
        label = _crop_pad(volumes["label"], center, self.patch_size, 0.0)
        return normalize_modalities(ct, pet, self.plan), (label > 0).astype(np.uint8)

    def patch_with_instance(self, case_id: str, center, component_id: int):
        image, all_lesions = self.patch(case_id, center)
        instances = _crop_pad(self.volumes(case_id)["instances"], center, self.patch_size, 0)
        selected = (instances == int(component_id)).astype(np.uint8) if component_id else np.zeros_like(all_lesions)
        return image, all_lesions, selected


# ---------------------------------------------------------------------------
# dataset (drop-in for SingleLesionDataset)
# ---------------------------------------------------------------------------

class CachedPatchDataset(torch.utils.data.Dataset):
    """Patch dataset backed by the offline cache.

    Same contract as ``SingleLesionDataset``: ``set_epoch``, identical sample
    keys, identical prompt construction and normalisation. The only difference
    is that patches are sliced instead of resampled.
    """

    def __init__(self, records: Sequence[CaseRecord], stage: str, config: dict, plan: dict,
                 prompt_type: str | None = None, cache_dir: Path | None = None):
        self.records = list(records)
        self.stage, self.prompt_type = stage, prompt_type
        self.config = dict(config, target_spacing=tuple(float(v) for v in plan["target_spacing"]))
        self.patch_size = tuple(int(v) for v in config["patch_size"])
        self.cache = VolumeCache(cache_dir or cache_dir_for(config))
        # Case-level arrays are independent of patch size; allow smoke tests or
        # deployments to change the crop without rebuilding the volume cache.
        self.cache.patch_size = self.patch_size
        self.epoch = 0
        self.entries: list[tuple[str, int]] = []
        if stage != "train":
            for record in self.records:
                lesions = list(self.cache.catalog(record.case_id)["lesions"])
                limit = int(config.get("max_val_lesions_per_case", 0))
                if limit and len(lesions) > limit:
                    order = np.linspace(0, len(lesions) - 1, limit).round().astype(int)
                    lesions = [sorted(lesions, key=lambda x: x["volume_mm3"])[i] for i in order]
                self.entries.extend((record.case_id, int(lesion["component_id"])) for lesion in lesions)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.config["samples_per_epoch"]) if self.stage == "train" else len(self.entries)

    def _select(self, index: int, rng: np.random.Generator, prompt_only: bool = False):
        if self.stage == "train":
            record = self.records[int(rng.integers(0, len(self.records)))]
            catalog = self.cache.catalog(record.case_id)
            lesions = list(catalog["lesions"])
            if prompt_only:
                allowed = set(catalog.get("prompt_component_ids", []))
                lesions = [lesion for lesion in lesions if int(lesion["component_id"]) in allowed]
            if not lesions:
                raise RuntimeError(f"No lesion in {record.case_id}")
            volumes = np.asarray([x["volume_mm3"] for x in lesions], dtype=np.float64)
            reference = float(np.median(volumes))
            weights = np.clip((reference / np.maximum(volumes, 1.0)) ** 0.25, 0.25, 4.0)
            weights /= weights.sum()
            return record.case_id, int(lesions[int(rng.choice(len(lesions), p=weights))]["component_id"])
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
            automatic and rng.random() < float(self.config.get("automatic_random_patch_probability", 0.0))
        )
        if random_automatic_patch:
            case_id = self.records[int(rng.integers(0, len(self.records)))].case_id
            component_id = 0
            center = self._random_body_center(case_id, rng)
        else:
            case_id, component_id = self._select(index, rng, prompt_only=not automatic)
            catalog = self.cache.catalog(case_id)
            lesion = next(x for x in catalog["lesions"] if int(x["component_id"]) == component_id)
            center = tuple(float(v) for v in lesion["center_zyx"])
        image, all_lesions, selected_target = self.cache.patch_with_instance(case_id, center, component_id)
        target = all_lesions
        if automatic:
            selected_target.fill(0)
        prompt_type = (
            "automatic" if automatic else
            (self.prompt_type or self.config["prompt_types"][int(rng.integers(0, len(self.config["prompt_types"])))])
        )
        prompts, route = build_prompts(
            selected_target, np.zeros_like(target), image[1], prompt_type, rng, self.config,
            negative_exclusion=target,
        )
        edit_mask = build_edit_mask(selected_target, self.config) if not automatic else np.zeros_like(target, np.float32)
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
                                        float(np.prod(self.cache.target_spacing)),
                                       dtype=torch.float32),
            "spacing": torch.tensor(self.cache.target_spacing, dtype=torch.float32),
            "case_id": case_id,
            "component_id": torch.tensor(component_id, dtype=torch.int64),
            "prompt_type": prompt_type,
        }
        return output

    def _random_body_center(self, case_id: str, rng: np.random.Generator):
        volumes = self.cache.volumes(case_id)
        ct = volumes["ct"]
        body = np.isfinite(ct) & (ct > -950.0)
        center = []
        for axis in range(3):
            reduce_axes = tuple(i for i in range(3) if i != axis)
            occupied = np.flatnonzero(body.any(axis=reduce_axes))
            lo, hi = (int(occupied[0]), int(occupied[-1])) if occupied.size else (0, ct.shape[axis] - 1)
            center.append(float(rng.integers(lo, hi + 1)))
        return tuple(center)


class BankPatchDataset(CachedPatchDataset):
    """Dataset that reads the pre-sliced patch bank when one is present.

    Prompts are generated from the stored mask, so augmentation randomness is
    unchanged; the sliding-window crop and resampling are skipped entirely.
    """

    def __init__(self, records, stage, config, plan, prompt_type=None, cache_dir=None):
        super().__init__(records, stage, config, plan, prompt_type, cache_dir)
        index_path = Path(cache_dir or cache_dir_for(config)) / "bank" / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No patch bank at {index_path}; run: python cache.py build --profile <profile> --bank-limit N"
            )
        self.bank_dir = index_path.parent
        self.bank = json.loads(index_path.read_text(encoding="utf-8"))
        self.bank_by_case: dict[str, list[dict]] = {}
        for entry in self.bank:
            self.bank_by_case.setdefault(entry["case_id"], []).append(entry)

    def __len__(self):
        if self.stage == "train":
            return int(self.config["samples_per_epoch"])
        return len(self.bank)

    def __getitem__(self, index: int):
        seed = self.config["seed"] + index * 104729 + self.epoch * 1000003
        rng = np.random.default_rng(seed)
        entry = self.bank[int(rng.integers(0, len(self.bank)))] if self.stage == "train" \
            else self.bank[index % len(self.bank)]
        payload = np.load(self.bank_dir / entry["file"])
        image = payload["image"].astype(np.float32)
        target = (payload["lesions"] > 0).astype(np.uint8)
        selected_target = (payload["selected"] > 0).astype(np.uint8) if "selected" in payload.files else target.copy()
        # Offline maps: prompt sampling reuses them instead of recomputing EDTs.
        component_ids = payload["component_ids"].astype(np.int32) if "component_ids" in payload.files else None
        depth = payload["depth"].astype(np.float32) if "depth" in payload.files else None
        automatic = rng.random() < float(self.config.get("automatic_training_probability", 0.0))
        if automatic:
            selected_target.fill(0)
        prompt_type = (
            "automatic" if automatic else
            (self.prompt_type or self.config["prompt_types"][int(rng.integers(0, len(self.config["prompt_types"])))])
        )
        prompts, route = build_prompts(
            selected_target, np.zeros_like(target), image[1], prompt_type, rng, self.config,
            component_ids=component_ids, depth=depth, negative_exclusion=target,
        )
        edit_mask = build_edit_mask(selected_target, self.config) if not automatic else np.zeros_like(target, np.float32)
        return {
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
                                        float(np.prod(self.cache.target_spacing)),
                                       dtype=torch.float32),
            "spacing": torch.tensor(self.cache.target_spacing, dtype=torch.float32),
            "case_id": entry["case_id"],
            "component_id": torch.tensor(int(entry["component_id"]), dtype=torch.int64),
            "prompt_type": prompt_type,
        }


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(config: dict, cache_dir: Path | None = None, samples: int = 6) -> dict:
    """Compare the cached pipeline against the original one at the SAME physical
    location, and report how resampling changed the lesion catalogue.

    Two separate questions are answered:

    1. Geometry: for a lesion centre taken from the original source-resolution
       catalogue, is the cached patch the same physical window? Measured as the
       residual after the best integer alignment (a sub-voxel grid phase shift
       is expected; a large residual would mean a real bug).
    2. Catalogue: resampling the label to target spacing can merge or drop
       tiny components, which changes which lesions the sampler can pick.
    """
    from data import _component_catalog, extract_physical_patch

    cache_dir = Path(cache_dir or cache_dir_for(config))
    store = VolumeCache(cache_dir)
    registry = build_registry(config)
    target_spacing = store.target_spacing
    report = {"cases": {}, "geometry": [], "catalogue": []}

    for record in list(registry["train"]) + list(registry["val"])[:1]:
        ct, pet, label, spacing = _load_case(record.ct_path, record.pet_path, record.label_path)
        source_lesions, coordinates = _component_catalog(record.label_path, tuple(spacing))
        cached_catalog = store.catalog(record.case_id)
        source_volumes = sorted(round(x.volume_mm3, 1) for x in source_lesions)
        cached_volumes = sorted(round(x["volume_mm3"], 1) for x in cached_catalog["lesions"])
        report["catalogue"].append({
            "case_id": record.case_id,
            "source_lesions": len(source_lesions),
            "cached_lesions": len(cached_catalog["lesions"]),
            "source_total_volume_mm3": round(float(sum(source_volumes)), 1),
            "cached_total_volume_mm3": round(float(sum(cached_volumes)), 1),
        })
        for lesion in list(source_lesions)[:max(1, samples // 3)]:
            center_source = tuple(float(v) for v in lesion.center_zyx)
            center_cached = tuple(c * s / t for c, s, t in zip(center_source, spacing, target_spacing))
            ct_o, pet_o, label_o = extract_physical_patch(
                [ct, pet, label], center_source, spacing, target_spacing,
                store.patch_size, orders=(1, 1, 0), fills=(-1000.0, 0.0, 0),
            )
            image_o = normalize_modalities(ct_o, pet_o, store.plan)
            image_c, label_c = store.patch(record.case_id, center_cached)
            mask_o = (label_o > 0.5)
            diff = np.abs(image_o - image_c)
            best = None
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        shifted = np.roll(np.roll(np.roll(image_c, dz, 1), dy, 2), dx, 3)
                        value = float(np.abs(image_o - shifted).mean())
                        if best is None or value < best[0]:
                            best = (value, (dz, dy, dx))
            mask_c = (label_c > 0)
            overlap = float(2 * (mask_o & mask_c).sum() / max(int(mask_o.sum()) + int(mask_c.sum()), 1))
            report["geometry"].append({
                "case_id": record.case_id,
                "component_id": int(lesion.component_id),
                "image_mean_abs_diff": round(float(diff.mean()), 4),
                "best_shift": list(best[1]),
                "image_mean_abs_diff_after_shift": round(best[0], 4),
                "lesion_overlap_dice": round(overlap, 4),
                "source_mask_voxels": int(mask_o.sum()),
                "cached_mask_voxels": int(mask_c.sum()),
            })

    report["geometry_mean_abs_diff"] = round(
        float(np.mean([row["image_mean_abs_diff"] for row in report["geometry"]])), 4)
    report["geometry_mean_abs_diff_after_shift"] = round(
        float(np.mean([row["image_mean_abs_diff_after_shift"] for row in report["geometry"]])), 4)
    report["geometry_mean_dice"] = round(
        float(np.mean([row["lesion_overlap_dice"] for row in report["geometry"]])), 4)
    return report


def main():
    parser = argparse.ArgumentParser(description="Offline patch cache for LINet")
    parser.add_argument("command", choices=("build", "verify", "info"))
    parser.add_argument("--profile", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--images-dir", help="Directory containing <case>_<modality>.nii.gz files")
    parser.add_argument("--labels-dir", help="Directory containing <case>.nii.gz label files")
    parser.add_argument("--split-json", help="nnU-Net-style train/validation split JSON")
    parser.add_argument("--ct-suffix", help="CT modality suffix (default: 0000)")
    parser.add_argument("--pet-suffix", help="PET modality suffix (default: 0001)")
    parser.add_argument("--output-dir", help="Run directory; sets the default plan/cache location")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cached-lesions-per-case", type=int,
                        help="Maximum size-stratified prompt lesions cached per case")
    parser.add_argument("--patches-per-lesion", type=int, default=8)
    parser.add_argument("--bank-limit", type=int, default=512)
    parser.add_argument("--no-bank", action="store_true", help="Only cache volumes, skip the patch bank")
    parser.add_argument("--samples", type=int, default=6, help="Samples compared by the verify command")
    args = parser.parse_args()
    config = get_config(args.profile)
    for key in ("images_dir", "labels_dir", "split_json", "ct_suffix", "pet_suffix", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.output_dir:
        config["plan_path"] = str(Path(args.output_dir) / "preprocess_plan.json")
    if args.cached_lesions_per_case is not None:
        config["cached_lesions_per_case"] = int(args.cached_lesions_per_case)
    cache_dir = Path(args.cache_dir) if args.cache_dir else cache_dir_for(config)

    if args.command == "build":
        build_cache(config, cache_dir)
        if not args.no_bank:
            build_patch_bank(config, cache_dir, args.patches_per_lesion, args.bank_limit)
    elif args.command == "verify":
        report = verify(config, cache_dir, args.samples)
        print(json.dumps(report, indent=2))
    else:
        meta = json.loads((cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
        print(json.dumps(meta, indent=2))
        bank_index = cache_dir / "bank" / "index.json"
        if bank_index.exists():
            print(f"bank patches: {len(json.loads(bank_index.read_text(encoding='utf-8')))}")


if __name__ == "__main__":
    main()
