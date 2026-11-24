"""Check coarse orientation search as a bounded LoFTR follow-up."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from compare_seo_nir_registration import LoFTRMatcher, load_pair, pan_display
from nir_registration_metrics import fit_and_validate_spatial


KEYS = ("JAX_436/JAX_436_8", "OMA_930/OMA_930_26", "UCSD_291/UCSD_291_3")


def main():
    cv2.setNumThreads(1)
    model = LoFTRMatcher(HERE / "loftr_outdoor.ckpt")
    output = HERE / "rotation_probe.jsonl"
    for key in KEYS:
        source, pan, rgb, valid = load_pair(Path(r"E:\Dataset\SEO"), key)
        h, w = source.shape[:2]
        rotations = (np.eye(3), np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]]),
                     np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]]),
                     np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]]))
        trials = []
        for turns, rotation in enumerate(rotations):
            rotated = np.ascontiguousarray(np.rot90(source, turns))
            for target_name, target in (("rgb", rgb), ("pan", pan_display(pan))):
                src, dst, confidence = model.match(rotated, target)
                if len(src):
                    src = cv2.perspectiveTransform(src[:, None], np.linalg.inv(rotation.astype(np.float64)))[:, 0]
                fitted = fit_and_validate_spatial(src, dst, (w, h), (target.shape[1], target.shape[0]), confidence=confidence)
                row = {"target": target_name, "rotation_ccw_degrees": turns * 90,
                       "matches": len(src), **fitted}
                trials.append(row)
                print(json.dumps({"key": key, "target": target_name, "rotation": turns * 90,
                                  "matches": len(src), "accepted": fitted["accepted"],
                                  "reason": fitted["reason"]}), flush=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "trials": trials}) + "\n")


if __name__ == "__main__":
    main()
