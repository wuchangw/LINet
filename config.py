from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

BASE_CONFIG = {
    "images_dir": str(WORKSPACE / "data3d" / "imagesTr"),
    "labels_dir": str(WORKSPACE / "data3d" / "labelsTr"),
    "split_json": str(WORKSPACE / "data3d" / "psma_demo.json"),
    # AutoPET convention: 0000 is CT and 0001 is PET. Override these for
    # datasets with the reverse nnU-Net modality order (for example DeepPSMA).
    "ct_suffix": "0000",
    "pet_suffix": "0001",
    "split_key": 0,
    "output_dir": str(ROOT / "runs" / "smoke"),
    "plan_path": str(ROOT / "runs" / "smoke" / "preprocess_plan.json"),
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patch_size": (80, 80, 80),
    "batch_size": 1,
    "num_workers": 0,
    "samples_per_epoch": 32,
    "max_val_lesions_per_case": 12,
    "epochs": 2,
    "allow_missing": True,
    "amp": True,
    "amp_dtype": "float16",
    "grad_accum": 1,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "features": (24, 48, 96, 192, 256),
    "blocks": (1, 2, 3, 4, 4),
    "decoder_convs": (1, 1, 1, 1),
    "strides": ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    "kernels": ((3, 3, 3),) * 5,
    "prompt_base_channels": 8,
    "prompt_max_channels": 32,
    "deep_supervision": True,
    "prompt_types": ("click", "bbox", "scribble"),
    # Supervise every connected lesion visible in an anchor patch. Set to
    # "single_lesion" only for reproducing the original lesion-specific setup.
    "target_mode": "all_lesions_in_patch",
    "clicks_pos": 5,
    "clicks_neg": 5,
    "click_sigma": 2.0,
    "bbox_jitter_mm": 8.0,
    "scribble_length": (8, 40),
    # Legacy compatibility only. Joint training shares one image-encoder pass.
    "two_pass_probability": 0.0,
    # Train the image-only nnU-Net path explicitly. The second pass then learns
    # to refine that automatic prediction with corrective prompts.
    "automatic_training_probability": 0.50,
    "automatic_random_patch_probability": 0.50,
    # Cache prompt geometry for at most N size-stratified lesions per case;
    # automatic segmentation still sees the complete all-lesion target.
    "cached_lesions_per_case": 8,
    "prompt_edit_margin_mm": 24.0,
    "auto_loss_weight": 1.0,
    "refine_loss_weight": 1.0,
    "corrective_clicks_per_round": 5,
    # Whole-volume automatic inference and local prompt-update controls.
    "auto_sliding_window_overlap": 0.5,
    "auto_threshold": 0.5,
    "prompt_refinement_radius_mm": 48.0,
    "prompt_bbox_margin_mm": 8.0,
    "custom_loss_weight": 0.70,
    "small_lesion_reference_mm3": 1000.0,
    "volume_weight_cap": 4.0,
    "validation_interval": 1,
}

PROFILES = {
    "smoke": {},
    "full": {
        "output_dir": str(ROOT / "runs" / "full"),
        "plan_path": str(ROOT / "runs" / "full" / "preprocess_plan.json"),
        "patch_size": (128, 128, 128),
        "epochs": 1000,
        "samples_per_epoch": 250,
        "max_val_lesions_per_case": 0,
        "allow_missing": False,
        "num_workers": 4,
        "grad_accum": 2,
        "validation_interval": 5,
        "routine_val_lesions_per_case": 2,
        "full_validation_interval": 50,
    },
}


def get_config(profile: str = "smoke") -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {tuple(PROFILES)}")
    config = deepcopy(BASE_CONFIG)
    config.update(PROFILES[profile])
    config["profile"] = profile
    return config
