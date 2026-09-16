from __future__ import annotations

import torch
import torch.nn.functional as F


def _outputs(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def nnunet_dice_ce(logits, target, eps=1e-6):
    probability = torch.sigmoid(logits)
    dims = tuple(range(2, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()
    ce = F.binary_cross_entropy_with_logits(logits, target)
    return dice + ce


def physical_volume_prompt_tversky_loss(logits, target, volume_mm3, positive_prompt, negative_prompt,
                                         reference_mm3=1000.0, weight_cap=4.0, eps=1e-6):
    """The single added objective: size-adaptive Tversky plus seed fidelity."""
    probability = torch.sigmoid(logits)
    dims = tuple(range(2, probability.ndim))
    volume = volume_mm3.reshape(-1, 1).to(probability.dtype).clamp_min(1.0)
    smallness = (float(reference_mm3) / volume).sqrt().clamp(1.0, float(weight_cap))
    fraction = (volume / float(reference_mm3)).clamp(0.0, 1.0)
    beta = 0.80 - 0.20 * fraction
    alpha = 1.0 - beta
    tp = (probability * target).sum(dims)
    fp = (probability * (1.0 - target)).sum(dims)
    fn = ((1.0 - probability) * target).sum(dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    region = (smallness * (1.0 - tversky).pow(0.75)).mean()

    seed_terms = []
    if positive_prompt is not None:
        seed = positive_prompt > 0.99
        if seed.any():
            seed_terms.append(-torch.log(probability[seed].clamp_min(eps)).mean())
    if negative_prompt is not None:
        seed = negative_prompt > 0.99
        if seed.any():
            seed_terms.append(-torch.log((1.0 - probability[seed]).clamp_min(eps)).mean())
    seed_loss = torch.stack(seed_terms).mean() if seed_terms else logits.new_zeros(())
    return region + 0.20 * seed_loss


def deep_supervision_loss(outputs, target, volume_mm3, positive_prompt, negative_prompt, config):
    outputs = _outputs(outputs)
    weights = torch.as_tensor([1.0 / (2 ** i) for i in range(len(outputs))], device=target.device, dtype=target.dtype)
    if len(outputs) > 1:
        weights[-1] = 0.0
    weights /= weights.sum().clamp_min(1e-8)
    total = target.new_zeros(())
    standard_total = target.new_zeros(())
    custom_total = target.new_zeros(())
    for index, (weight, logits) in enumerate(zip(weights, outputs)):
        resized = target if logits.shape[2:] == target.shape[2:] else F.interpolate(target, logits.shape[2:], mode="nearest")
        standard = nnunet_dice_ce(logits, resized)
        custom = logits.new_zeros(())
        if index == 0:
            custom = physical_volume_prompt_tversky_loss(
                logits, target, volume_mm3, positive_prompt, negative_prompt,
                reference_mm3=config["small_lesion_reference_mm3"],
                weight_cap=config["volume_weight_cap"],
            )
        standard_total = standard_total + weight * standard
        custom_total = custom_total + custom
        total = total + weight * standard + float(config["custom_loss_weight"]) * custom
    return total, standard_total, custom_total


def dice_from_logits(logits, target, threshold=0.5):
    prediction = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    dims = tuple(range(2, prediction.ndim))
    intersection = (prediction & truth).sum(dims).float()
    denominator = prediction.sum(dims).float() + truth.sum(dims).float()
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()

