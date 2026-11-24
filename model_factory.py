"""Explicit model variants without changing existing baseline implementations."""

import copy

import torch


def resolve_checkpoint_config(config, checkpoint):
    """Restore model/input settings and SEO splits, keeping runtime paths/device."""
    resolved = copy.deepcopy(config)
    saved = checkpoint.get("config", {})
    if "model" in saved:
        resolved["model"] = copy.deepcopy(saved["model"])
    elif "model_variant" in checkpoint:
        resolved["model"]["variant"] = checkpoint["model_variant"]
    saved_dataset = saved.get("dataset", {})
    if (
        resolved["dataset"].get("type", "").lower() == "seo"
        and saved_dataset.get("type", "").lower() == "seo"
    ):
        if "random_seed" in saved:
            resolved["random_seed"] = copy.deepcopy(saved["random_seed"])
        resolved["dataset"]["val_ratio"] = copy.deepcopy(saved_dataset.get("val_ratio", 0.2))
        # An explicit runtime protocol permits re-evaluating older weights.
        # Otherwise use the checkpoint's protocol, including Min/Max filtering.
        runtime_supervision = resolved['dataset'].get('supervision', {})
        if (isinstance(runtime_supervision, dict) and 'mode' not in runtime_supervision
                and 'supervision' in saved_dataset):
            supervision = copy.deepcopy(saved_dataset['supervision'])
            supervision.update(runtime_supervision)
            resolved['dataset']['supervision'] = supervision
    for name in ("normalization", "input_size", "in_channels", "num_classes"):
        if name in saved_dataset:
            resolved["dataset"][name] = copy.deepcopy(saved_dataset[name])
    if "config" in checkpoint and "input_size" not in saved_dataset:
        resolved["dataset"]["input_size"] = 512
    # Checkpoints predating native crops were trained and evaluated with Resize.
    resolved["dataset"]["spatial_mode"] = saved_dataset.get("spatial_mode", "resize")
    for name in ("tile_stride", "tile_batch_size"):
        if name in saved_dataset:
            resolved["dataset"][name] = copy.deepcopy(saved_dataset[name])
        else:
            resolved["dataset"].pop(name, None)
    if "config" in checkpoint:
        runtime_nir = resolved["dataset"].get("nir", {})
        saved_nir = saved_dataset.get("nir", {"enabled": False})
        if "nir" in saved_dataset or "nir" in resolved["dataset"]:
            nir = copy.deepcopy(saved_nir)
            for name in ("root_dir", "manifest"):
                if name in runtime_nir:
                    nir[name] = copy.deepcopy(runtime_nir[name])
            resolved["dataset"]["nir"] = nir
    return resolved


def build_model(config):
    variant = config["model"].get("variant", "cbam").lower()
    options = config["model"].get("multicolor", {})
    nir_options = options.get("nir", {})
    dataset_nir = config["dataset"].get("nir", {})
    if not isinstance(nir_options, dict) or not isinstance(dataset_nir, dict):
        raise ValueError("model.multicolor.nir and dataset.nir must be configuration mappings")
    nir_enabled = nir_options.get("enabled", False)
    data_nir_enabled = dataset_nir.get("enabled", False)
    if not isinstance(nir_enabled, bool) or not isinstance(data_nir_enabled, bool):
        raise ValueError("NIR enabled flags must be booleans")
    if nir_enabled != data_nir_enabled:
        raise ValueError("model.multicolor.nir.enabled and dataset.nir.enabled must agree")
    if nir_enabled and variant != "multicolor":
        raise ValueError("NIR auxiliary input requires model.variant='multicolor'")
    if variant == "baseline":
        from model import build_model as build_baseline
        return build_baseline(config)
    if variant in ("cbam", "coordatt"):
        from modela import build_model as build_attention, CBAM, CoordAtt
        model = build_attention(config)
        channels = model.segmentation_head[0].in_channels
        attention_class = CBAM if variant == "cbam" else CoordAtt
        if not isinstance(model.attention, attention_class):
            model.attention = (
                attention_class(channels=channels) if variant == "cbam"
                else attention_class(inp=channels)
            )
        return model
    if variant == "multicolor":
        from model_multicolor import MultiColorSegFormer
        normalization = config["dataset"]["normalization"]
        attention = options.get("cross_scale_attention", {})
        description = f"Building SegFormer {config['model']['encoder']} + HSV/Lab dual spatial gates"
        if attention.get("enabled", False):
            description += " + stage 3/4 cross-scale criss-cross attention"
        if nir_enabled:
            description += " + masked NIR auxiliary branch"
        print(description)
        return MultiColorSegFormer(
            encoder_name=config["model"]["encoder"],
            encoder_weights=config["model"].get("encoder_weights"),
            in_channels=config["dataset"]["in_channels"],
            classes=config["dataset"]["num_classes"],
            activation=None,
            aux_channels=tuple(options.get("aux_channels", (8, 16, 32, 64))),
            pooling=options.get("pooling", "avg"),
            gate_channels=options.get("gate_channels", 16),
            gate_bias=options.get("gate_bias", -2.0),
            auxiliary_supervision=options.get("auxiliary_supervision", False),
            aux_decoder_channels=options.get("aux_decoder_channels", 16),
            cross_scale_attention=attention.get("enabled", False),
            cross_scale_attention_channels=attention.get("attention_channels", 64),
            nir_enabled=nir_enabled,
            input_mean=normalization["mean"],
            input_std=normalization["std"],
        )
    raise ValueError(f"Unknown model variant {variant!r}; use baseline, cbam, coordatt, or multicolor")


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    return total, trainable


def initialize_from_checkpoint(model, path, include_auxiliary_heads=False):
    """Transfer matching weights; optionally retain trained auxiliary heads."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    target = model.state_dict()
    compatible = {}
    skipped = []
    prefixes = (
        "encoder.", "decoder.", "segmentation_head.", "hsv_encoder.",
        "lab_encoder.", "fusions.", "cross_scale_attention.", "nir_encoder.", "nir_fusions.",
    )
    if include_auxiliary_heads:
        prefixes += ("auxiliary_heads.",)
    for name, value in source.items():
        name = name.removeprefix("module.")
        if (
            (name.startswith(prefixes)
             or name in {"input_mean", "input_std"})
            and name in target
            and value.shape == target[name].shape
        ):
            compatible[name] = value
        else:
            skipped.append(name)
    if not any(name.startswith("encoder.") for name in compatible):
        raise ValueError("Checkpoint has no compatible RGB encoder parameters")
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "skipped": len(skipped),
        "new": len(target) - len(compatible),
    }
