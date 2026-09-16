from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_config
from data import SingleLesionDataset, build_registry, load_or_create_plan, seed_everything
from engine import evaluate, evaluate_click_curve, volume_strata
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--profile", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--images-dir"); parser.add_argument("--labels-dir"); parser.add_argument("--split-json")
    parser.add_argument("--output", default="test_report.json"); parser.add_argument("--device")
    parser.add_argument("--clicks", type=int, default=5, help="Fixed number of positive and negative clicks")
    parser.add_argument("--prompts", nargs="+", choices=("click", "bbox", "scribble"),
                        help="Prompt modalities to evaluate (default: checkpoint/config modalities)")
    parser.add_argument("--click-curve", action="store_true", help="Additionally evaluate rounds 1-7")
    args = parser.parse_args(); config = get_config(args.profile)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config.update(checkpoint.get("config", {}))
    for key in ("images_dir", "labels_dir", "split_json", "device"):
        if getattr(args, key) is not None: config[key] = getattr(args, key)
    config["clicks_pos"] = int(args.clicks); config["clicks_neg"] = int(args.clicks)
    if args.prompts:
        config["prompt_types"] = tuple(args.prompts)
    seed_everything(config["seed"]); registry = build_registry(config)
    plan = load_or_create_plan(registry["train"], config["plan_path"])
    loaders = {prompt: DataLoader(SingleLesionDataset(registry["val"], "val", config, plan, prompt),
                                  batch_size=1, shuffle=False, num_workers=0)
               for prompt in config["prompt_types"]}
    model = build_model(config).to(config["device"]); model.load_state_dict(checkpoint["model"], strict=True)
    dtype = torch.float16 if config["amp_dtype"] == "float16" else torch.bfloat16
    report = evaluate(model, loaders, torch.device(config["device"]), config, dtype)
    target_mode = config.get("target_mode", "all_lesions_in_patch")
    report["evaluation_scope"] = target_mode
    report["metric_note"] = (
        "Metrics are computed per anchor patch against every visible lesion in that patch; "
        "they are not merged whole-case metrics. HD95/ASD are in mm."
        if target_mode == "all_lesions_in_patch" else
        "Metrics are computed per prompted single-lesion patch. HD95/ASD are in mm."
    )
    report["volume_strata"] = {
        prompt: volume_strata(report[prompt]["per_lesion"]) for prompt in config["prompt_types"]
    }
    if args.click_curve:
        curve_config = dict(config, clicks_pos=1, clicks_neg=1)
        curve_loader = DataLoader(SingleLesionDataset(registry["val"], "val", curve_config, plan, "click"),
                                  batch_size=1, shuffle=False, num_workers=0)
        report["iterative_click_curve"] = evaluate_click_curve(
            model, curve_loader, torch.device(config["device"]), curve_config, dtype, rounds=7,
        )
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    compact = {key: report[key]["summary"] for key in config["prompt_types"]}
    compact["fixed_positive_clicks"] = int(args.clicks)
    compact["fixed_negative_clicks"] = int(args.clicks)
    if "iterative_click_curve" in report:
        compact["iterative_dice_by_round"] = [x["dice_mean"] for x in report["iterative_click_curve"]["rounds"]]
        compact["noc_at_0.80"] = report["iterative_click_curve"]["noc_at_0.80"]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
