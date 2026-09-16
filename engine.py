from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from data import STRUCTURE_26, encode_points
from losses import deep_supervision_loss, dice_from_logits
from metrics import case_metrics, summarize


def _first(outputs):
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def _move(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _seed_maps(batch):
    positive = torch.maximum(batch["click"][:, 0:1], batch["scribble"][:, 0:1])
    negative = torch.maximum(batch["click"][:, 1:2], batch["scribble"][:, 1:2])
    return positive, negative


def forward_batch(model, batch):
    return model(batch["image"], batch["click"], batch["bbox"], batch["scribble"],
                 batch["prev_prob"], batch["route"])


def _is_automatic_batch(batch):
    prompt_types = batch.get("prompt_type")
    if not isinstance(prompt_types, (list, tuple)) or not prompt_types:
        return False
    return all(value == "automatic" for value in prompt_types)


def forward_sample(model, batch):
    """Route each batch to the right pass: pure auto for automatic samples,
    single prompt-update refinement otherwise."""
    if _is_automatic_batch(batch):
        return model.forward_auto(batch["image"])
    return forward_batch(model, batch)


def add_corrective_clicks(batch, probability, config, epoch: int, batch_index: int):
    corrected = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    corrected["prev_prob"] = probability.detach()
    pred = (probability.detach().cpu().numpy() >= 0.5)
    truth = batch["target"].detach().cpu().numpy() >= 0.5
    for item in range(pred.shape[0]):
        fn = truth[item, 0] & ~pred[item, 0]
        fp = pred[item, 0] & ~truth[item, 0]
        rng = np.random.default_rng(int(config["seed"]) + epoch * 100003 + batch_index * 101 + item)

        def deepest_per_component(mask):
            if not mask.any():
                return []
            from scipy import ndimage
            components, count = ndimage.label(mask, structure=STRUCTURE_26)
            points = []
            for component_id in range(1, count + 1):
                component = components == component_id
                distance = ndimage.distance_transform_edt(component)
                point = np.unravel_index(int(distance.argmax()), distance.shape)
                points.append(tuple(int(v) for v in point))
            return points[:int(config.get("corrective_clicks_per_round", config.get("clicks_pos", 5)))]

        pos = deepest_per_component(fn if fn.any() else truth[item, 0])
        neg = deepest_per_component(fp)
        pos_map = torch.from_numpy(encode_points(pos, pred.shape[2:], config["click_sigma"])).to(probability.device)
        neg_map = torch.from_numpy(encode_points(neg, pred.shape[2:], config["click_sigma"])).to(probability.device)
        corrected["click"][item, 0] = torch.maximum(corrected["click"][item, 0], pos_map)
        corrected["click"][item, 1] = torch.maximum(corrected["click"][item, 1], neg_map)
        corrected["route"][item, 0] = 1.0
    return corrected


def add_prediction_error_prompts(batch, probability, config, epoch: int, batch_index: int):
    """Add one-lesion FN and local FP prompts without changing prompt type.

    The cached selected-lesion truth is used only to simulate interaction. The
    complete target protects every other lesion from being treated as a false
    positive.
    """
    corrected = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    pred = probability.detach().cpu().numpy() >= 0.5
    truth = batch["target"].detach().cpu().numpy() >= 0.5
    selected = batch["selected_target"].detach().cpu().numpy() >= 0.5
    edit = batch["edit_mask"].detach().cpu().numpy() >= 0.5
    for item in range(pred.shape[0]):
        rng = np.random.default_rng(int(config["seed"]) + epoch * 100003 + batch_index * 101 + item)

        def deepest(mask):
            if not mask.any():
                return []
            distance = ndimage.distance_transform_edt(mask)
            return [tuple(int(v) for v in np.unravel_index(int(distance.argmax()), distance.shape))]

        false_negative = selected[item, 0] & ~pred[item, 0]
        false_positive = pred[item, 0] & ~truth[item, 0] & edit[item, 0]
        positive = deepest(false_negative)
        negative = deepest(false_positive)
        if corrected["route"][item, 0] > 0:  # click
            if positive:
                value = torch.from_numpy(encode_points(positive, pred.shape[2:], config["click_sigma"])).to(probability.device)
                corrected["click"][item, 0] = torch.maximum(corrected["click"][item, 0], value)
            if negative:
                value = torch.from_numpy(encode_points(negative, pred.shape[2:], config["click_sigma"])).to(probability.device)
                corrected["click"][item, 1] = torch.maximum(corrected["click"][item, 1], value)
        elif corrected["route"][item, 2] > 0:  # scribble expert; seed is expanded by its encoder
            if positive:
                value = torch.from_numpy(encode_points(positive, pred.shape[2:], config["click_sigma"])).to(probability.device)
                corrected["scribble"][item, 0] = torch.maximum(corrected["scribble"][item, 0], value)
            if negative:
                value = torch.from_numpy(encode_points(negative, pred.shape[2:], config["click_sigma"])).to(probability.device)
                corrected["scribble"][item, 1] = torch.maximum(corrected["scribble"][item, 1], value)
    return corrected


@torch.no_grad()
def evaluate(model, loaders: dict, device, config, amp_dtype):
    model.eval(); reports = {}; all_dice = []
    amp_enabled = bool(config["amp"] and str(device).startswith("cuda"))
    for prompt_type, loader in loaders.items():
        rows, losses = [], []
        for batch in loader:
            batch = _move(batch, device)
            positive, negative = _seed_maps(batch)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                _, outputs = model.forward_joint(
                    batch["image"], batch["click"], batch["bbox"], batch["scribble"],
                    batch["route"], batch["edit_mask"],
                )
                loss, _, _ = deep_supervision_loss(outputs, batch["target"], batch["volume_mm3"],
                                                    positive, negative, config)
            probability = torch.sigmoid(_first(outputs).float())
            prediction = probability >= 0.5
            for item in range(prediction.shape[0]):
                row = case_metrics(prediction[item, 0].cpu().numpy(), batch["target"][item, 0].cpu().numpy(),
                                   batch["spacing"][item].cpu().numpy())
                row.update({"case_id": batch["case_id"][item],
                            "component_id": int(batch["component_id"][item].item()),
                            "volume_mm3": float(batch["volume_mm3"][item].item())})
                rows.append(row)
            losses.append(float(loss.item()))
        summary = summarize(rows); summary["loss"] = float(np.mean(losses)) if losses else math.nan
        reports[prompt_type] = {"summary": summary, "per_lesion": rows}
        all_dice.append(summary["dice_mean"])
    reports["macro_dice"] = float(np.mean(all_dice)) if all_dice else math.nan
    return reports


@torch.no_grad()
def evaluate_click_curve(model, loader, device, config, amp_dtype, rounds=7):
    """Evaluate real iterative correction: one initial pair, then model-error clicks."""
    model.eval(); amp_enabled = bool(config["amp"] and str(device).startswith("cuda"))
    per_round = [[] for _ in range(rounds)]
    trajectories = []
    for batch_index, raw_batch in enumerate(loader):
        batch = _move(raw_batch, device); lesion_trajectory = []
        for round_index in range(rounds):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                _, outputs = model.forward_joint(
                    batch["image"], batch["click"], batch["bbox"], batch["scribble"],
                    batch["route"], batch["edit_mask"],
                )
            probability = torch.sigmoid(_first(outputs).float())
            prediction = probability >= 0.5
            metrics = case_metrics(prediction[0, 0].cpu().numpy(), batch["target"][0, 0].cpu().numpy(),
                                   batch["spacing"][0].cpu().numpy())
            metrics.update({"case_id": batch["case_id"][0],
                            "component_id": int(batch["component_id"][0].item()),
                            "volume_mm3": float(batch["volume_mm3"][0].item())})
            per_round[round_index].append(metrics); lesion_trajectory.append(metrics["dice"])
            if round_index + 1 < rounds:
                batch = add_corrective_clicks(batch, probability, config, 0, batch_index * rounds + round_index)
        trajectories.append(lesion_trajectory)
    summaries = [summarize(rows) for rows in per_round]
    no_clicks = []
    for trajectory in trajectories:
        reached = next((index + 1 for index, dice in enumerate(trajectory) if dice >= 0.80), rounds + 1)
        no_clicks.append(reached)
    return {"rounds": summaries, "per_lesion_dice": trajectories,
            "noc_at_0.80": float(np.mean(no_clicks)) if no_clicks else math.nan,
            "failures_at_0.80": int(sum(value > rounds for value in no_clicks))}


def volume_strata(rows: list[dict]) -> dict:
    bins = {"micro_lt_100mm3": [], "small_100_1000mm3": [], "large_ge_1000mm3": []}
    for row in rows:
        volume = row["volume_mm3"]
        key = "micro_lt_100mm3" if volume < 100 else ("small_100_1000mm3" if volume < 1000 else "large_ge_1000mm3")
        bins[key].append(row)
    return {key: summarize(value) for key, value in bins.items()}


class Trainer:
    def __init__(self, model, train_loader, val_loaders, config, routine_val_loader=None):
        self.config = config
        self.device = torch.device(config["device"])
        self.model = model.to(self.device)
        self.train_loader, self.val_loaders = train_loader, val_loaders
        self.routine_val_loader = routine_val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.amp_dtype = torch.float16 if config["amp_dtype"] == "float16" else torch.bfloat16
        self.amp_enabled = bool(config["amp"] and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.best_click = -1.0
        self.output_dir = Path(config["output_dir"]); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train.log"; self.log_path.write_text("", encoding="utf-8")

    def log(self, message):
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    def _lr(self, epoch):
        return float(self.config["lr"]) * (1.0 - epoch / max(int(self.config["epochs"]), 1)) ** 0.9

    def train_epoch(self, epoch):
        self.model.train(); self.train_loader.dataset.set_epoch(epoch)
        for group in self.optimizer.param_groups: group["lr"] = self._lr(epoch)
        total_loss = total_dice = custom_sum = 0.0; start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        for index, raw_batch in enumerate(self.train_loader):
            batch = _move(raw_batch, self.device)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                image_skips = self.model.encode_image(batch["image"])
                automatic_outputs = self.model.decode_auto(image_skips, batch["image"][:, 2:3])
            automatic_probability = torch.sigmoid(_first(automatic_outputs).detach().float())
            prompted = add_prediction_error_prompts(batch, automatic_probability, self.config, epoch, index)
            prompt_active = bool((prompted["route"] != 0).any().item())
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                auto_loss, _, auto_custom = deep_supervision_loss(
                    automatic_outputs, batch["target"], batch["volume_mm3"], None, None, self.config,
                )
                if prompt_active:
                    previous = automatic_probability.to(batch["image"].dtype)
                    refined_outputs = self.model.refine_from_features(
                        image_skips, batch["image"][:, 2:3], prompted["click"], prompted["bbox"],
                        prompted["scribble"], previous, prompted["route"], edit_mask=None,
                    )
                    outputs = self.model._compose_local(
                        automatic_outputs, refined_outputs, prompted["edit_mask"],
                    )
                    positive, negative = _seed_maps(prompted)
                    refine_loss, _, refine_custom = deep_supervision_loss(
                        outputs, batch["target"], batch["volume_mm3"], positive, negative, self.config,
                    )
                else:
                    outputs = automatic_outputs
                    refine_loss = auto_loss.new_zeros(())
                    refine_custom = auto_custom.new_zeros(())
                loss = (float(self.config.get("auto_loss_weight", 1.0)) * auto_loss +
                        float(self.config.get("refine_loss_weight", 1.0)) * refine_loss)
                custom = auto_custom + refine_custom
            scaled = loss / int(self.config["grad_accum"])
            self.scaler.scale(scaled).backward() if self.scaler.is_enabled() else scaled.backward()
            measured_loss, measured_custom = float(loss.item()), float(custom.item())
            logits = _first(outputs)

            if (index + 1) % int(self.config["grad_accum"]) == 0 or index + 1 == len(self.train_loader):
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12.0)
                if self.scaler.is_enabled(): self.scaler.step(self.optimizer); self.scaler.update()
                else: self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += measured_loss; custom_sum += measured_custom
            total_dice += float(dice_from_logits(logits.detach(), batch["target"]).item())
        count = max(len(self.train_loader), 1)
        return {"loss": total_loss / count, "dice": total_dice / count,
                "custom_loss": custom_sum / count, "seconds": time.time() - start,
                "lr": self.optimizer.param_groups[0]["lr"]}

    def save(self, name, epoch, reports=None):
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch, "best_click": self.best_click, "config": self.config,
                    "reports": reports}, self.output_dir / name)

    def train(self):
        history = []
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)
        for epoch in range(int(self.config["epochs"])):
            train = self.train_epoch(epoch)
            self.log(f"Epoch {epoch + 1}/{self.config['epochs']} loss={train['loss']:.4f} "
                     f"dice={train['dice']:.4f} custom={train['custom_loss']:.4f} "
                     f"lr={train['lr']:.6g} time={train['seconds']:.1f}s")
            routine_interval = max(1, int(self.config.get("validation_interval", 5)))
            full_interval = max(routine_interval, int(self.config.get("full_validation_interval", 50)))
            final_epoch = epoch + 1 == int(self.config["epochs"])
            should_routine = epoch == 0 or (epoch + 1) % routine_interval == 0 or final_epoch
            should_full = (epoch + 1) % full_interval == 0 or final_epoch
            validation = {"routine": None, "full": None}

            if should_routine:
                loaders = {"click": self.routine_val_loader} if self.routine_val_loader is not None else {"click": self.val_loaders["click"]}
                routine_reports = evaluate(self.model, loaders, self.device, self.config, self.amp_dtype)
                validation["routine"] = routine_reports
                summary = routine_reports["click"]["summary"]
                self.log(f"  Val/routine-click: Dice={summary['dice_mean']:.4f}+/-{summary['dice_std']:.4f} "
                         f"HD95(strict)={summary['hd95_strict_mean_mm']} "
                         f"empty={summary['empty_prediction_rate']:.3f}")
                click = summary["dice_mean"]
                if click > self.best_click:
                    self.best_click = click; self.save("best.pth", epoch + 1, validation)
                    self.log(f"  -> New best routine Click Dice={click:.4f}")

            if should_full:
                full_reports = evaluate(self.model, self.val_loaders, self.device, self.config, self.amp_dtype)
                validation["full"] = full_reports
                for prompt in self.config["prompt_types"]:
                    summary = full_reports[prompt]["summary"]
                    self.log(f"  Val/full-{prompt}: Dice={summary['dice_mean']:.4f}+/-{summary['dice_std']:.4f} "
                             f"HD95(strict)={summary['hd95_strict_mean_mm']} "
                             f"empty={summary['empty_prediction_rate']:.3f}")
                self.log(f"  Val/full-macro Dice={full_reports['macro_dice']:.4f}")

            self.save("last.pth", epoch + 1, validation)
            history.append({"epoch": epoch + 1, "train": train, "validation": validation})
            (self.output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")
        peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3 if self.device.type == "cuda" else 0.0
        self.log(f"Peak CUDA allocated: {peak:.3f} GiB")
        return history
