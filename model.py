from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ExpertEncoder(nn.Module):
    def __init__(self, in_channels: int, stages: int, base: int, maximum: int):
        super().__init__()
        channels = [min(base * 2 ** i, maximum) for i in range(stages)]
        self.output_channels = tuple(channels)
        self.blocks = nn.ModuleList([
            SeparableConv3d(in_channels if i == 0 else channels[i - 1], channels[i], 1 if i == 0 else 2)
            for i in range(stages)
        ])

    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x); outputs.append(x)
        return outputs


class PromptIsolatedExpertBank(nn.Module):
    """Three prompt encoders with no shared weights or early feature mixing."""

    def __init__(self, stages: int, base: int = 8, maximum: int = 32):
        super().__init__()
        self.click = ExpertEncoder(2, stages, base, maximum)
        self.bbox = ExpertEncoder(2, stages, base, maximum)
        self.scribble = ExpertEncoder(2, stages, base, maximum)
        self.output_channels = self.click.output_channels

    @staticmethod
    def _route(features, active):
        return [x * active[:, None, None, None, None] for x in features]

    def forward(self, click, bbox, scribble, route):
        branches = [
            self._route(self.click(click), route[:, 0]),
            self._route(self.bbox(bbox), route[:, 1]),
            self._route(self.scribble(scribble), route[:, 2]),
        ]
        count = route.sum(1).clamp_min(1.0)[:, None, None, None, None]
        return [(a + b + c) / count for a, b, c in zip(*branches)]


class ErrorUncertaintyFeedback(nn.Module):
    """Encode previous probability, predictive uncertainty and soft boundary."""

    def __init__(self, stages: int, base: int = 8, maximum: int = 32):
        super().__init__()
        self.encoder = ExpertEncoder(3, stages, base, maximum)
        self.output_channels = self.encoder.output_channels

    def forward(self, probability):
        probability = probability.clamp(0.0, 1.0)
        uncertainty = 4.0 * probability * (1.0 - probability)
        high = F.max_pool3d(probability, 3, 1, 1)
        low = -F.max_pool3d(-probability, 3, 1, 1)
        boundary = (high - low).clamp(0.0, 1.0)
        return self.encoder(torch.cat([probability, uncertainty, boundary], dim=1))


class GatedPromptFeedbackFusion(nn.Module):
    def __init__(self, image_channels, prompt_channels, feedback_channels):
        super().__init__()
        self.prompt_projection = nn.Conv3d(prompt_channels, image_channels, 1, bias=False)
        self.feedback_projection = nn.Conv3d(feedback_channels, image_channels, 1, bias=False)
        hidden = max(8, image_channels // 4)
        self.gate = nn.Sequential(
            nn.Conv3d(image_channels * 3, hidden, 1, bias=False),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(hidden, image_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, prompt, feedback):
        prompt = self.prompt_projection(prompt)
        feedback = self.feedback_projection(feedback)
        gate = self.gate(torch.cat([image, prompt, feedback], dim=1))
        return image + gate * (prompt + feedback)


class SUVSmallLesionRefinement(nn.Module):
    """High-resolution depthwise residual refinement gated by absolute log-SUV."""

    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.context = nn.Conv3d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(1e-2, inplace=True),
        )
        hidden = max(8, channels // 4)
        self.suv_gate = nn.Sequential(
            nn.Conv3d(1, hidden, 3, padding=1, bias=False),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature, suv_log):
        if suv_log.shape[2:] != feature.shape[2:]:
            suv_log = F.interpolate(suv_log, feature.shape[2:], mode="trilinear", align_corners=False)
        detail = self.mix(torch.cat([self.local(feature), self.context(feature)], dim=1))
        return feature + detail * self.suv_gate(suv_log)


class LightweightDecoder(nn.Module):
    def __init__(self, encoder: ResidualEncoder, decoder_convs: Sequence[int], deep_supervision: bool):
        super().__init__()
        self.deep_supervision = deep_supervision
        transpose = get_matching_convtransp(encoder.conv_op)
        self.up, self.stages, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        n_stages = len(encoder.output_channels)
        for index in range(1, n_stages):
            below = encoder.output_channels[-index]
            skip = encoder.output_channels[-index - 1]
            stride = encoder.strides[-index]
            self.up.append(transpose(below, skip, stride, stride, bias=encoder.conv_bias))
            self.stages.append(StackedConvBlocks(
                decoder_convs[index - 1], encoder.conv_op, skip * 2, skip,
                encoder.kernel_sizes[-index - 1], 1, encoder.conv_bias,
                encoder.norm_op, encoder.norm_op_kwargs, encoder.dropout_op,
                encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, False,
            ))
            self.heads.append(nn.Conv3d(skip, 1, 1))
        self.refiner = SUVSmallLesionRefinement(encoder.output_channels[0])

    def forward(self, skips, suv_log):
        x, outputs = skips[-1], []
        for index, stage in enumerate(self.stages):
            x = self.up[index](x)
            x = stage(torch.cat([x, skips[-index - 2]], dim=1))
            if index == len(self.stages) - 1:
                x = self.refiner(x, suv_log)
            if self.deep_supervision or index == len(self.stages) - 1:
                outputs.append(self.heads[index](x))
        outputs.reverse()
        return outputs if self.deep_supervision else outputs[0]


class InteractiveNnUNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        features = tuple(config["features"])
        self.deep_supervision = bool(config.get("deep_supervision", True))
        self.image_encoder = ResidualEncoder(
            3, len(features), features, nn.Conv3d, tuple(config["kernels"]), tuple(config["strides"]),
            tuple(config["blocks"]), False, nn.InstanceNorm3d, {"affine": True}, None, None,
            nn.LeakyReLU, {"negative_slope": 1e-2, "inplace": True}, BasicBlockD, None,
            return_skips=True, disable_default_stem=False, stem_channels=None,
        )
        stages = len(features)
        base, maximum = int(config["prompt_base_channels"]), int(config["prompt_max_channels"])
        self.prompt_experts = PromptIsolatedExpertBank(stages, base, maximum)
        self.feedback = ErrorUncertaintyFeedback(stages, base, maximum)
        self.fusions = nn.ModuleList([
            GatedPromptFeedbackFusion(i, p, f)
            for i, p, f in zip(self.image_encoder.output_channels,
                               self.prompt_experts.output_channels, self.feedback.output_channels)
        ])
        self.decoder = LightweightDecoder(self.image_encoder, tuple(config["decoder_convs"]), self.deep_supervision)
        InitWeights_He(1e-2)(self); init_last_bn_before_add_to_0(self)
        # A rare foreground should not start with a 0.5 probability everywhere.
        for head in self.decoder.heads:
            if head.bias is not None:
                nn.init.constant_(head.bias, -2.0)

    @staticmethod
    def _add_spatial_prompt_prior(outputs, click, bbox, scribble, route):
        positive = torch.maximum(click[:, 0:1], scribble[:, 0:1])
        negative = torch.maximum(click[:, 1:2], scribble[:, 1:2])
        point_prior = 3.0 * (positive - negative)
        box_prior = 1.5 * bbox[:, 1:2] * route[:, 1:2, None, None, None]
        prior = point_prior + box_prior
        if isinstance(outputs, (list, tuple)):
            return [outputs[0] + prior, *outputs[1:]]
        return outputs + prior

    @staticmethod
    def _compose_local(auto_outputs, refined_outputs, edit_mask):
        """Keep the automatic prediction bit-exact outside the prompt ROI."""
        auto_list = list(auto_outputs) if isinstance(auto_outputs, (list, tuple)) else [auto_outputs]
        refined_list = list(refined_outputs) if isinstance(refined_outputs, (list, tuple)) else [refined_outputs]
        composed = []
        for automatic, refined in zip(auto_list, refined_list):
            mask = edit_mask
            if mask.shape[2:] != automatic.shape[2:]:
                mask = F.interpolate(mask, automatic.shape[2:], mode="nearest")
            # Detaching the baseline in the refinement path prevents the prompt
            # objective from degrading the complete, no-prompt segmentation.
            baseline = automatic.detach()
            composed.append(baseline + mask.to(automatic.dtype) * (refined - baseline))
        if isinstance(auto_outputs, (list, tuple)):
            return composed
        return composed[0]

    def encode_image(self, image):
        if image.shape[1] != 3:
            raise ValueError(f"image must have CT, normalized PET and log-SUV channels; got {image.shape[1]}")
        return self.image_encoder(image)

    def decode_auto(self, image_skips, suv_log):
        return self.decoder(image_skips, suv_log)

    def refine_from_features(self, image_skips, suv_log, click, bbox, scribble,
                             prev_prob, route, edit_mask=None):
        """Refine from already-computed image features (no second encoder pass)."""
        prompt_skips = self.prompt_experts(click, bbox, scribble, route.to(suv_log.dtype))
        feedback_skips = self.feedback(prev_prob)
        fused = [fusion(a, b, c) for fusion, a, b, c in
                 zip(self.fusions, image_skips, prompt_skips, feedback_skips)]
        refined = self.decoder(fused, suv_log)
        refined = self._add_spatial_prompt_prior(refined, click, bbox, scribble, route)
        return refined if edit_mask is None else self._compose_local(prev_prob, refined, edit_mask)

    def forward_joint(self, image, click, bbox, scribble, route, edit_mask):
        """One encoder pass producing both no-prompt and locally refined output."""
        image_skips = self.encode_image(image)
        automatic = self.decode_auto(image_skips, image[:, 2:3])
        if not bool((route != 0).any().item()):
            return automatic, automatic
        auto_primary = automatic[0] if isinstance(automatic, (list, tuple)) else automatic
        previous = torch.sigmoid(auto_primary.detach().float()).to(image.dtype)
        refined = self.refine_from_features(
            image_skips, image[:, 2:3], click, bbox, scribble, previous, route,
            edit_mask=None,
        )
        final = self._compose_local(automatic, refined, edit_mask)
        return automatic, final

    def forward_auto(self, image):
        """Pure nnU-Net automatic path.

        One forward pass over the image alone; the prompt experts, the error
        feedback branch and the gated fusion are bypassed entirely. The decoder
        must find every suspected lesion visible in the input as the complete
        initial segmentation, without relying on any user interaction.
        """
        image_skips = self.encode_image(image)
        return self.decode_auto(image_skips, image[:, 2:3])

    def forward_refine(self, image, click, bbox, scribble, prev_prob, route):
        """Single prompt-update refinement pass on top of the automatic result.

        ``prev_prob`` carries the frozen global automatic segmentation so the
        prompt branch only learns a local increment (add / delete / boundary
        fix / fill-in) for the prompted region.
        """
        image_skips = self.encode_image(image)
        return self.refine_from_features(
            image_skips, image[:, 2:3], click, bbox, scribble, prev_prob, route,
        )

    def forward(self, image, click, bbox, scribble, prev_prob, route):
        """Backward-compatible entry point routing to the refinement pass."""
        return self.forward_refine(image, click, bbox, scribble, prev_prob, route)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: dict) -> InteractiveNnUNet:
    model = InteractiveNnUNet(config)
    if model.parameter_count() >= 30_000_000:
        raise RuntimeError(f"Lightweight constraint violated: {model.parameter_count():,} parameters")
    return model
