"""SegFormer with lightweight color pyramids and optional masked NIR fusion."""

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from color_spaces import rgb_to_hsv, rgb_to_lab
from cross_scale_attention import CrossScaleCrissCrossAttention


class PooledColorEncoder(nn.Module):
    """Independent pool -> pointwise projection -> GELU stages at 1/4..1/32."""

    def __init__(self, channels=(8, 16, 32, 64), pooling="avg", in_channels=3):
        super().__init__()
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("aux_channels must contain four positive channel counts")
        if pooling not in ("avg", "max"):
            raise ValueError("pooling must be 'avg' or 'max'")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        pool_class = nn.AvgPool2d if pooling == "avg" else nn.MaxPool2d
        stages = []
        input_channels = in_channels
        for index, output_channels in enumerate(channels):
            stride = 4 if index == 0 else 2
            stages.append(nn.Sequential(
                pool_class(kernel_size=stride, stride=stride),
                nn.Conv2d(input_channels, output_channels, kernel_size=1),
                nn.GELU(),
            ))
            input_channels = output_channels
        self.stages = nn.ModuleList(stages)

    def forward(self, x, valid=None, return_valid=False):
        features, masks = [], []
        if valid is not None:
            x = x.masked_fill(~valid, 0)
        for stage in self.stages:
            x = stage(x)
            if valid is not None:
                # Mixed valid/invalid pooling cells cannot inject NIR evidence.
                stride = stage[0].stride
                valid = F.avg_pool2d(valid.float(), stride, stride) >= 1.0
                x = x.masked_fill(~valid, 0)
                masks.append(valid)
            features.append(x)
        if return_valid:
            return features, masks
        return features


class DualSpatialFusion(nn.Module):
    """Predict two independent spatial gates and inject an auxiliary residual."""

    def __init__(self, rgb_channels, auxiliary_channels, gate_channels=16, gate_bias=-2.0):
        super().__init__()
        self.hsv_projection = nn.Conv2d(auxiliary_channels, rgb_channels, 1, bias=False)
        self.lab_projection = nn.Conv2d(auxiliary_channels, rgb_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(3 * rgb_channels, gate_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_channels, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[2].bias, gate_bias)
        self.merge = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=1, bias=False)

    def forward(self, rgb, hsv, lab):
        hsv = self.hsv_projection(hsv)
        lab = self.lab_projection(lab)
        if hsv.shape[-2:] != rgb.shape[-2:]:
            hsv = F.interpolate(hsv, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            lab = F.interpolate(lab, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        gates = self.gate(torch.cat((rgb, hsv, lab), dim=1))
        residual = gates[:, 0:1] * hsv + gates[:, 1:2] * lab
        return rgb + self.merge(residual), gates


class MaskedNIRFusion(nn.Module):
    """Inject NIR after the unchanged HSV/Lab fusion, only on valid cells."""

    def __init__(self, rgb_channels, auxiliary_channels, gate_channels=16, gate_bias=-2.0):
        super().__init__()
        self.nir_projection = nn.Conv2d(auxiliary_channels, rgb_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * rgb_channels, gate_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[2].bias, gate_bias)
        self.merge = nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False)

    def forward(self, fused, nir, valid):
        nir = self.nir_projection(nir)
        if nir.shape[-2:] != fused.shape[-2:]:
            nir = F.interpolate(nir, size=fused.shape[-2:], mode="bilinear", align_corners=False)
            valid = F.interpolate(valid.float(), size=fused.shape[-2:],
                                  mode="bilinear", align_corners=False) >= 1.0
        nir = nir.masked_fill(~valid, 0)
        gate = self.gate(torch.cat((fused, nir), dim=1)).masked_fill(~valid, 0)
        residual = self.merge(gate * nir).masked_fill(~valid, 0)
        return fused + residual, gate


class AuxiliaryPredictionHead(nn.Module):
    """Fuse a branch's four native scales into full-size auxiliary logits."""

    def __init__(self, channels, classes, decoder_channels=16):
        super().__init__()
        if len(channels) != 4 or decoder_channels <= 0:
            raise ValueError("Auxiliary heads require four scales and positive decoder_channels")
        self.projections = nn.ModuleList([
            nn.Conv2d(channels_at_scale, decoder_channels, kernel_size=1)
            for channels_at_scale in channels
        ])
        self.classifier = nn.Sequential(
            nn.Conv2d(4 * decoder_channels, decoder_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, classes, kernel_size=1),
        )

    def forward(self, features, output_size):
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(projection(feature), size=target_size,
                          mode="bilinear", align_corners=False)
            for projection, feature in zip(self.projections, features)
        ]
        logits = self.classifier(torch.cat(projected, dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class MultiColorSegFormer(smp.Segformer):
    """Preserve the RGB MiT path; feed four fused scales only to the decoder.

    Input is normalized RGB, matching the existing train/inference pipeline.
    HSV and Lab are computed after reversing RGB normalization, then mapped
    from their fixed [0, 1] ranges to [-1, 1]. Optional independent prediction
    heads supervise the RGB, HSV, Lab and enabled NIR features before fusion.
    NIR is supplied separately in [-1, 1], with an optional validity mask.
    Optional
    cross-scale criss-cross attention enhances fused stage 3 using stage 4.
    Disagreement calibration is not part of this variant.
    """

    def __init__(
        self,
        *args,
        aux_channels=(8, 16, 32, 64),
        pooling="avg",
        gate_channels=16,
        gate_bias=-2.0,
        auxiliary_supervision=False,
        aux_decoder_channels=16,
        cross_scale_attention=False,
        cross_scale_attention_channels=64,
        nir_enabled=False,
        input_mean=(0.485, 0.456, 0.406),
        input_std=(0.229, 0.224, 0.225),
        **kwargs,
    ):
        if kwargs.get("in_channels", 3) != 3:
            raise ValueError("MultiColorSegFormer requires three RGB input channels")
        if not kwargs.get("encoder_name", "resnet34").startswith("mit_"):
            raise ValueError("MultiColorSegFormer requires a four-stage MiT encoder")
        if kwargs.get("aux_params") is not None:
            raise ValueError("Classification heads are not supported in this segmentation variant")
        if gate_channels <= 0:
            raise ValueError("gate_channels must be positive")
        super().__init__(*args, **kwargs)

        self.stage_indices = tuple(
            index for index, channels in enumerate(self.encoder.out_channels)
            if index > 0 and channels > 0
        )
        if len(self.stage_indices) != 4:
            raise ValueError("Expected four nonempty MiT feature stages")
        mean = torch.tensor(input_mean, dtype=torch.float32)
        std = torch.tensor(input_std, dtype=torch.float32)
        if mean.numel() != 3 or std.numel() != 3 or torch.any(std <= 0):
            raise ValueError("RGB mean/std must have three components and positive std")
        self.register_buffer("input_mean", mean.reshape(1, 3, 1, 1))
        self.register_buffer("input_std", std.reshape(1, 3, 1, 1))
        self.hsv_encoder = PooledColorEncoder(aux_channels, pooling)
        self.lab_encoder = PooledColorEncoder(aux_channels, pooling)
        self.fusions = nn.ModuleList([
            DualSpatialFusion(
                self.encoder.out_channels[index], width, gate_channels, gate_bias
            )
            for index, width in zip(self.stage_indices, aux_channels)
        ])
        self.cross_scale_attention = (
            CrossScaleCrissCrossAttention(
                self.encoder.out_channels[self.stage_indices[2]],
                self.encoder.out_channels[self.stage_indices[3]],
                attention_channels=cross_scale_attention_channels,
            )
            if cross_scale_attention else None
        )
        self.auxiliary_supervision = auxiliary_supervision
        self.nir_enabled = bool(nir_enabled)
        if auxiliary_supervision:
            classes = self.segmentation_head[0].out_channels
            rgb_channels = tuple(self.encoder.out_channels[index] for index in self.stage_indices)
            self.auxiliary_heads = nn.ModuleDict({
                name: AuxiliaryPredictionHead(channels, classes, aux_decoder_channels)
                for name, channels in (("rgb", rgb_channels), ("hsv", aux_channels), ("lab", aux_channels))
            })
        # Append optional modules so the legacy parameter layout is unchanged.
        if self.nir_enabled:
            self.nir_encoder = PooledColorEncoder(aux_channels, pooling, in_channels=1)
            self.nir_fusions = nn.ModuleList([
                MaskedNIRFusion(
                    self.encoder.out_channels[index], width, gate_channels, gate_bias
                )
                for index, width in zip(self.stage_indices, aux_channels)
            ])
            if auxiliary_supervision:
                self.auxiliary_heads["nir"] = AuxiliaryPredictionHead(
                    aux_channels, classes, aux_decoder_channels
                )

    def _prepare_nir(self, x, nir, nir_valid):
        if not self.nir_enabled:
            if nir is not None or nir_valid is not None:
                raise ValueError("NIR inputs require nir_enabled=True")
            return None, None
        if not isinstance(nir, torch.Tensor):
            raise ValueError("NIR-enabled model requires an explicit nir tensor with shape B1HW")
        expected_shape = (x.shape[0], 1, *x.shape[-2:])
        if nir.ndim != 4 or tuple(nir.shape) != expected_shape:
            raise ValueError(f"nir must have shape B1HW matching RGB: expected {expected_shape}")
        if not nir.is_floating_point() or nir.device != x.device:
            raise ValueError("nir must be floating-point and on the same device as RGB")
        if not torch.isfinite(nir).all():
            raise ValueError("nir must contain only finite values, including invalid locations")
        if nir_valid is None:
            valid = torch.ones_like(nir, dtype=torch.bool)
        else:
            if not isinstance(nir_valid, torch.Tensor):
                raise ValueError("nir_valid must be a bool/float tensor with shape B1HW or BHW")
            if nir_valid.ndim == 3:
                nir_valid = nir_valid.unsqueeze(1)
            if tuple(nir_valid.shape) != expected_shape or nir_valid.device != x.device:
                raise ValueError("nir_valid must have shape B1HW or BHW matching RGB on the same device")
            if nir_valid.dtype != torch.bool and not nir_valid.is_floating_point():
                raise ValueError("nir_valid must be a bool or floating-point mask")
            if not torch.isfinite(nir_valid).all():
                raise ValueError("nir_valid must contain only finite values")
            if torch.any((nir_valid < 0) | (nir_valid > 1)):
                raise ValueError("nir_valid values must be in [0, 1]")
            valid = nir_valid > 0.5
        return nir.float().masked_fill(~valid, 0), valid

    def forward_features(self, x, nir=None, nir_valid=None):
        self.check_input_shape(x)
        nir_input, valid = self._prepare_nir(x, nir, nir_valid)
        encoder_features = list(self.encoder(x))
        # Color conversion is kept in float32 even under mixed-precision use.
        rgb_input = (x.float() * self.input_std + self.input_mean).clamp(0.0, 1.0)
        hsv = self.hsv_encoder(rgb_to_hsv(rgb_input) * 2.0 - 1.0)
        lab = self.lab_encoder(rgb_to_lab(rgb_input) * 2.0 - 1.0)
        if self.nir_enabled:
            nir_features, nir_masks = self.nir_encoder(nir_input, valid, return_valid=True)
        rgb = [encoder_features[index] for index in self.stage_indices]
        fused, gates = [], []
        for fusion, rgb_stage, hsv_stage, lab_stage in zip(self.fusions, rgb, hsv, lab):
            fused_stage, stage_gates = fusion(rgb_stage, hsv_stage, lab_stage)
            fused.append(fused_stage)
            gates.append(stage_gates)
        nir_gates = []
        if self.nir_enabled:
            for index, (fusion, nir_stage, mask) in enumerate(zip(
                self.nir_fusions, nir_features, nir_masks
            )):
                fused[index], gate = fusion(fused[index], nir_stage, mask)
                nir_gates.append(gate)
        if self.cross_scale_attention is not None:
            # Enhance the fused stage only; auxiliary heads keep native features.
            fused[2] = self.cross_scale_attention(fused[2], fused[3])
        decoder_features = list(encoder_features)
        for index, fused_stage in zip(self.stage_indices, fused):
            decoder_features[index] = fused_stage
        features = {
            "rgb": rgb,
            "hsv": hsv,
            "lab": lab,
            "fused": fused,
            "gates": gates,
            "decoder_features": decoder_features,
        }
        if self.nir_enabled:
            features.update(nir=nir_features, nir_valid_features=nir_masks, nir_gates=nir_gates)
        return features

    def forward(self, x, nir=None, nir_valid=None):
        features = self.forward_features(x, nir, nir_valid)
        decoded = self.decoder(*features["decoder_features"])
        return self.segmentation_head(decoded)

    def forward_with_aux(self, x, nir=None, nir_valid=None):
        if not self.auxiliary_supervision:
            raise RuntimeError("Enable auxiliary_supervision to request branch predictions")
        features = self.forward_features(x, nir, nir_valid)
        decoded = self.decoder(*features["decoder_features"])
        return {
            "out": self.segmentation_head(decoded),
            "aux": {
                name: head(features[name], x.shape[-2:])
                for name, head in self.auxiliary_heads.items()
            },
        }
