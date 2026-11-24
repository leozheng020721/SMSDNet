"""Inspect local training checkpoints on CPU without changing source files."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "seo_segformer_native512_aux_cca_nir_mixup_minmax_best.pth",
    "seo_segformer_native512_aux_cca_nir_mixup_best.pth",
    "seo_segformer_native512_aux_cca_scratch_mixup_best.pth",
)


def read_history(path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        try:
            parsed.append({key: float(value) for key, value in row.items()})
        except (TypeError, ValueError):
            continue
    best = min(parsed, key=lambda row: row["ber"]) if parsed else None
    return {
        "path": str(path),
        "exists": True,
        "rows": len(rows),
        "parsed_rows": len(parsed),
        "first_epoch": parsed[0]["epoch"] if parsed else None,
        "last_epoch": parsed[-1]["epoch"] if parsed else None,
        "first_row": parsed[0] if parsed else None,
        "last_row": parsed[-1] if parsed else None,
        "best_row": best,
    }


def inspect_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    state = checkpoint.get("model_state_dict", {})
    report = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "checkpoint_keys": list(checkpoint),
        "epoch_zero_based": checkpoint.get("epoch"),
        "epoch_one_based": checkpoint.get("epoch", -1) + 1,
        "ber": checkpoint.get("ber"),
        "best_ber": checkpoint.get("best_ber"),
        "model_variant": checkpoint.get("model_variant"),
        "config": config,
        "effective_supervision_mode": config.get("dataset", {}).get("supervision", {}).get("mode", "max"),
        "effective_nir_enabled": config.get("model", {}).get("multicolor", {}).get("nir", {}).get("enabled", False),
        "state_tensor_count": len(state),
        "state_elements": sum(tensor.numel() for tensor in state.values()),
        "nir_state_keys": [key for key in state if "nir" in key.lower()],
        "scheduler_state": checkpoint.get("scheduler_state_dict"),
        "scaler_state": checkpoint.get("scaler_state_dict"),
        "optimizer_group_summary": [
            {key: value for key, value in group.items() if key != "params"}
            for group in checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
        ],
        "history": read_history(path.with_name(path.name.replace("_best.pth", "_history.csv"))),
        "initialization_provenance": {
            "encoder_weights_config": config.get("model", {}).get("encoder_weights"),
            "explicit_init_fields": {
                key: value
                for key, value in checkpoint.items()
                if any(token in key.lower() for token in ("init", "source", "pretrain"))
            },
            "limitation": "Config encoder_weights is not proof of the original --init-from or --resume command.",
        },
    }
    del checkpoint, state
    gc.collect()
    return report


def main():
    reports = [inspect_checkpoint(ROOT / "checkpoints" / name) for name in NAMES]
    output = Path(__file__).with_name("checkpoint_audit.json")
    assessment = {
        "available_no_nir_best": NAMES[2],
        "fair_nir_ablation_available": False,
        "shared_settings": [
            "MiT-B3, HSV/Lab gates, cross-scale CCA, 512 crops, 384 validation stride",
            "random_seed=42, val_ratio=0.2, grouped ground-plot split",
            "AdamW, configured 50-epoch cosine schedule, initial LR 6e-5, MixUp",
        ],
        "confounders": [
            "Current NIR Min/Max and both older checkpoints use different supervision/evaluation pixel sets.",
            "NIR eligibility filtering runs after the common group split, so NIR and no-NIR validation sample sets can differ.",
            "Current Min/Max uses batch 12 and AMP; both older runs use batch 6 and FP32.",
            "Old NIR history ends at epoch 17, while no-NIR and current Min/Max histories reach epoch 50.",
            "The Min/Max training guide transfers the old NIR best including auxiliary heads; original init commands are not embedded in checkpoints.",
        ],
        "documented_initialization": {
            "minmax": {
                "source": "REMOTE_TRAINING.md:153",
                "command_options": "--init-from checkpoints/seo_segformer_native512_aux_cca_nir_mixup_best.pth --init-auxiliary-heads",
                "evidence_strength": "Documented intended command; checkpoint metadata cannot independently prove it was used.",
            },
            "no_nir": {
                "source": "SEO_CROP_AUX_GUIDE.md:51",
                "description": "ImageNet RGB encoder; new task branches and optimizer; no shadow checkpoint transfer in documented command.",
            },
        },
        "interpretation": [
            "BER 21.0652 to 13.1386 cannot quantify a pure improvement because the Min/Max evaluation protocol changed.",
            "BER 19.6867 (no NIR) versus 13.1386 (NIR Min/Max) cannot quantify NIR benefit.",
            "The current model's own SEO validation miss rate 24.8391% indicates substantial residual misses even in-domain, relative to its automatic labels.",
            "Use exactly the same selected validation scenes and target pixels for diagnostic checkpoint comparisons; treat older checkpoint comparisons as exploratory, not controlled training ablations.",
        ],
    }
    output.write_text(json.dumps({"checkpoints": reports, "comparison_assessment": assessment}, indent=2), encoding="utf-8")
    print(str(output))
    for item in reports:
        config = item["config"]
        print(json.dumps({
            "name": Path(item["path"]).name,
            "epoch": item["epoch_one_based"],
            "ber": item["ber"],
            "best_ber": item["best_ber"],
            "model": config.get("model"),
            "dataset": config.get("dataset"),
            "training": config.get("training"),
            "seed": config.get("random_seed"),
            "history_last_epoch": item["history"]["last_epoch"],
            "history_best": item["history"]["best_row"],
            "checkpoint_keys": item["checkpoint_keys"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
