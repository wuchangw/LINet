from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def case_metrics(prediction: np.ndarray, target: np.ndarray, spacing) -> dict:
    pred, truth = prediction.astype(bool), target.astype(bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    both_empty = not pred.any() and not truth.any()
    dice = 1.0 if both_empty else (2.0 * tp) / max(2 * tp + fp + fn, 1)
    iou = 1.0 if both_empty else tp / max(tp + fp + fn, 1)
    # Empty/empty is a correct prediction. For one-sided-empty masks, the
    # corresponding precision/recall is zero instead of being silently dropped.
    ppv = 1.0 if both_empty else tp / max(tp + fp, 1)
    sen = 1.0 if both_empty else tp / max(tp + fn, 1)
    if not pred.any() or not truth.any():
        hd95 = math.inf if pred.any() != truth.any() else 0.0
        asd = math.inf if pred.any() != truth.any() else 0.0
    else:
        pred_surface = pred ^ ndimage.binary_erosion(pred, structure=np.ones((3, 3, 3)), border_value=0)
        truth_surface = truth ^ ndimage.binary_erosion(truth, structure=np.ones((3, 3, 3)), border_value=0)
        distance_to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing)
        distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
        distances = np.concatenate([distance_to_truth[pred_surface], distance_to_pred[truth_surface]])
        hd95 = float(np.percentile(distances, 95)) if distances.size else math.inf
        asd = float(distances.mean()) if distances.size else math.inf
    return {
        "dice": float(dice), "iou": float(iou), "ppv": float(ppv), "sen": float(sen),
        "hd95_mm": float(hd95), "asd_mm": float(asd),
        "tp": tp, "fp": fp, "fn": fn, "empty_prediction": int(not pred.any()),
        "target_voxels": int(truth.sum()), "prediction_voxels": int(pred.sum()),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0, "dice_mean": math.nan, "dice_std": math.nan,
                "iou_mean": math.nan, "iou_std": math.nan,
                "ppv_mean": math.nan, "ppv_std": math.nan,
                "sen_mean": math.nan, "sen_std": math.nan,
                "hd95_strict_mean_mm": math.nan, "hd95_finite_mean_mm": math.nan,
                "asd_strict_mean_mm": math.nan, "asd_finite_mean_mm": math.nan,
                "empty_prediction_rate": math.nan}
    dice = np.asarray([row["dice"] for row in rows], dtype=np.float64)
    iou = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    ppv = np.asarray([row["ppv"] for row in rows], dtype=np.float64)
    sen = np.asarray([row["sen"] for row in rows], dtype=np.float64)
    hd = np.asarray([row["hd95_mm"] for row in rows], dtype=np.float64)
    asd = np.asarray([row["asd_mm"] for row in rows], dtype=np.float64)
    finite_hd = hd[np.isfinite(hd)]; finite_asd = asd[np.isfinite(asd)]
    return {
        "cases": len(rows),
        "dice_mean": float(dice.mean()), "dice_std": float(dice.std()),
        "iou_mean": float(iou.mean()), "iou_std": float(iou.std()),
        "ppv_mean": float(ppv.mean()), "ppv_std": float(ppv.std()),
        "sen_mean": float(sen.mean()), "sen_std": float(sen.std()),
        "hd95_strict_mean_mm": float(hd.mean()) if np.isfinite(hd).all() else math.inf,
        "hd95_finite_mean_mm": float(finite_hd.mean()) if finite_hd.size else math.inf,
        "hd95_finite_std_mm": float(finite_hd.std()) if finite_hd.size else math.inf,
        "asd_strict_mean_mm": float(asd.mean()) if np.isfinite(asd).all() else math.inf,
        "asd_finite_mean_mm": float(finite_asd.mean()) if finite_asd.size else math.inf,
        "asd_finite_std_mm": float(finite_asd.std()) if finite_asd.size else math.inf,
        "empty_predictions": int(sum(row["empty_prediction"] for row in rows)),
        "empty_prediction_rate": float(np.mean([row["empty_prediction"] for row in rows])),
    }
