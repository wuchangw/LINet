from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_config
from data import SingleLesionDataset, build_registry, load_or_create_plan, seed_everything
from engine import Trainer
from model import build_model


def build_dataset(records, stage, config, plan, prompt_type=None):
    """Pick the patch source: cached (fast) or on-the-fly (bit-exact legacy).

    ``--cache off``      always use the original on-the-fly resampling;
    ``--cache auto``     use the offline cache when it exists, else fall back;
    ``--cache required`` fail loudly when the cache is missing.
    """
    mode = config.get("cache_mode", "off")
    if mode == "off":
        return SingleLesionDataset(records, stage, config, plan, prompt_type)
    from cache import BankPatchDataset, CachedPatchDataset, cache_dir_for, cache_is_usable
    cache_dir = Path(config.get("cache_dir") or cache_dir_for(config))
    if not cache_is_usable(cache_dir):
        if mode == "required":
            raise FileNotFoundError(
                f"No usable cache at {cache_dir}. Build it first: "
                f"python cache.py build --profile {config.get('profile', 'smoke')}"
            )
        print(f"[cache] no usable cache at {cache_dir}; falling back to on-the-fly resampling")
        return SingleLesionDataset(records, stage, config, plan, prompt_type)
    if config.get("cache_use_bank") and (cache_dir / "bank" / "index.json").exists():
        print(f"[cache] using pre-sliced patch bank at {cache_dir / 'bank'}")
        return BankPatchDataset(records, stage, config, plan, prompt_type, cache_dir)
    print(f"[cache] using cached volumes at {cache_dir}")
    return CachedPatchDataset(records, stage, config, plan, prompt_type, cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight nnU-Net multi-lesion interactive segmentation")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--images-dir"); parser.add_argument("--labels-dir"); parser.add_argument("--split-json")
    parser.add_argument("--ct-suffix", help="nnU-Net CT modality suffix (default: 0000)")
    parser.add_argument("--pet-suffix", help="nnU-Net PET modality suffix (default: 0001)")
    parser.add_argument("--output-dir"); parser.add_argument("--epochs", type=int); parser.add_argument("--device")
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--max-val-lesions-per-case", type=int)
    parser.add_argument("--patch-size", type=int, nargs=3, metavar=("D", "H", "W"),
                        help="Override the profile patch size, for example: --patch-size 96 96 96")
    parser.add_argument("--cache", choices=("off", "auto", "required"), default="auto",
                        help="Use the offline case cache built by cache.py (default: auto)")
    parser.add_argument("--cache-dir", help="Cache directory (default: <output-dir>/cache)")
    parser.add_argument("--use-bank", action="store_true",
                        help="Read the pre-sliced patch bank instead of the cached volumes")
    return parser.parse_args()


def main():
    args = parse_args(); config = get_config(args.profile)
    for key in ("images_dir", "labels_dir", "split_json", "ct_suffix", "pet_suffix", "output_dir", "epochs",
                "device", "samples_per_epoch", "max_val_lesions_per_case"):
        value = getattr(args, key)
        if value is not None: config[key] = value
    if args.patch_size is not None:
        config["patch_size"] = tuple(int(value) for value in args.patch_size)
    if args.output_dir:
        config["plan_path"] = str(Path(args.output_dir) / "preprocess_plan.json")
    config["cache_mode"] = args.cache
    if args.cache_dir: config["cache_dir"] = args.cache_dir
    config["cache_use_bank"] = bool(args.use_bank)
    seed_everything(int(config["seed"])); registry = build_registry(config)
    plan = load_or_create_plan(registry["train"], config["plan_path"])
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(config["output_dir"]) / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Profile={config['profile']} device={config['device']} patch={config['patch_size']}")
    print(f"Cases: train={len(registry['train'])} val={len(registry['val'])}; spacing={plan['target_spacing']}")
    train_set = build_dataset(registry["train"], "train", config, plan)
    val_sets = {prompt: build_dataset(registry["val"], "val", config, plan, prompt)
                for prompt in config["prompt_types"]}
    routine_config = dict(config)
    routine_config["max_val_lesions_per_case"] = int(config.get("routine_val_lesions_per_case", 2))
    routine_val_set = build_dataset(registry["val"], "val", routine_config, plan, "click")
    loader_args = dict(batch_size=config["batch_size"], num_workers=config["num_workers"],
                       pin_memory=str(config["device"]).startswith("cuda"))
    if int(config["num_workers"]) > 0:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loaders = {key: DataLoader(value, shuffle=False, **loader_args) for key, value in val_sets.items()}
    routine_val_loader = DataLoader(routine_val_set, shuffle=False, **loader_args)
    model = build_model(config)
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Routine validation: click only, {len(routine_val_set)} lesions; full validation every {config.get('full_validation_interval', 50)} epochs")
    Trainer(model, train_loader, val_loaders, config, routine_val_loader=routine_val_loader).train()


if __name__ == "__main__":
    main()
