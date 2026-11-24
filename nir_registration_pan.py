"""PAN-assisted registration candidates without writing or changing v1 outputs.

Every homography uses forward pixel-center coordinates. Returned point arrays
are float32 Nx2 arrays in the named image's native grid. A failed QA attempt is
retained for independent evaluation; it must not be treated as an accepted warp.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from align_seo_nir import AlignmentError, Options, _fit_and_check, resize_coordinate_matrix


@dataclass(frozen=True)
class PanOptions:
    max_features: int = 5000
    orb_features: int = 8000
    max_candidates: int = 6
    grid_max_side: int = 640
    same_grid_median_px: float = 2.0
    same_grid_p90_px: float = 5.0
    same_grid_model_max_px: float = 6.0
    qa: Options = field(default_factory=Options)


_STRATEGIES = (
    ("sift", "gray", 1.0, 0.75),
    ("sift", "green_clahe", 1.0, 0.80),
    ("sift", "gray", 2.0, 0.75),
    ("sift", "gray_clahe", 2.0, 0.80),
    ("orb", "gray", 1.0, 0.80),
    ("orb", "green_clahe", 2.0, 0.80),
)


def _points(points, h):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points.copy()
    return cv2.perspectiveTransform(points[:, None], np.asarray(h, np.float64))[:, 0]


def _stretch_pan(pan):
    pan = np.asarray(pan)
    if pan.ndim != 2 or not np.issubdtype(pan.dtype, np.number):
        raise ValueError("PAN must be a numeric two-dimensional array")
    finite = np.isfinite(pan)
    if not finite.any():
        raise ValueError("PAN has no finite measurements")
    low, high = np.percentile(pan[finite], (1, 99))
    values = np.where(finite, pan, low).astype(np.float32)
    return np.uint8(np.clip((values - low) / max(high - low, 1e-6), 0, 1) * 255)


def _check_rgb(rgb, name):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"{name} must be HxWx3 uint8 RGB")
    if min(rgb.shape[:2]) < 16:
        raise ValueError(f"{name} is too small for feature matching")
    return rgb


def _gray(image, mode):
    if image.ndim == 2:
        result = image
    elif mode.startswith("green"):
        result = image[:, :, 1]
    else:
        result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mode.endswith("clahe"):
        result = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(result)
    return np.ascontiguousarray(result)


def _match(source_gray, target_gray, detector, ratio, options):
    if detector == "sift":
        extractor = cv2.SIFT_create(nfeatures=options.max_features, contrastThreshold=0.01)
        norm = cv2.NORM_L2
    else:
        extractor = cv2.ORB_create(nfeatures=options.orb_features, edgeThreshold=15,
                                   fastThreshold=7)
        norm = cv2.NORM_HAMMING
    kp1, des1 = extractor.detectAndCompute(source_gray, None)
    kp2, des2 = extractor.detectAndCompute(target_gray, None)
    stats = {"keypoints_source": len(kp1), "keypoints_target": len(kp2)}
    empty = np.empty((0, 2), np.float32)
    if des1 is None or des2 is None or min(len(des1), len(des2)) < 2:
        return empty, empty.copy(), stats
    matcher = cv2.BFMatcher(norm)
    forward = [pair[0] for pair in matcher.knnMatch(des1, des2, k=2)
               if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    reverse = {(pair[0].trainIdx, pair[0].queryIdx)
               for pair in matcher.knnMatch(des2, des1, k=2)
               if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    matches = [match for match in forward
               if (match.queryIdx, match.trainIdx) in reverse]
    stats["forward_ratio_matches"] = len(forward)
    return (np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 2),
            np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 2), stats)


def _trial(source_base, target_base, target_size, detector, mode, scale, ratio, options):
    """Match in a scaled detection grid; fit and validate in the base source grid."""
    base_size = (source_base.shape[1], source_base.shape[0])
    if target_base.shape[:2] != source_base.shape[:2]:
        raise ValueError("Internal matching grids must have the same dimensions")
    detection_size = tuple(max(16, round(value * scale)) for value in base_size)
    source_gray = cv2.resize(_gray(source_base, mode), detection_size,
                             interpolation=cv2.INTER_CUBIC)
    target_gray = cv2.resize(_gray(target_base, mode), detection_size,
                             interpolation=cv2.INTER_CUBIC)
    strategy = f"{detector}_{mode}_s{scale:g}"
    row = {"strategy": strategy, "matcher": detector, "scale": scale, "ratio": ratio,
           "base_size_wh": base_size, "detection_size_wh": detection_size,
           "target_size_wh": target_size, "inliers": 0, "inlier_ratio": 0.0,
           "qa_passed": False, "qa_error": None, "qa": None,
           "H_base_to_target_raw": None, "H_base_to_target": None,
           "inlier_mask_raw": np.empty(0, bool)}
    cv2.setRNGSeed(options.qa.seed)
    source_points, target_points, stats = _match(
        source_gray, target_gray, detector, ratio, options)
    row.update(stats)
    # cv.resize maps centers, so dividing keypoint coordinates by scale is wrong.
    inverse_detection = np.linalg.inv(resize_coordinate_matrix(base_size, detection_size))
    source_points = _points(source_points, inverse_detection)
    target_points = _points(target_points, inverse_detection)
    row["points_src_base"] = source_points
    row["points_dst_target"] = _points(
        target_points, resize_coordinate_matrix(base_size, target_size))
    row["mutual_matches"] = len(source_points)
    if len(source_points) >= 4:
        raw, mask = cv2.findHomography(
            source_points, target_points, cv2.RANSAC, options.qa.ransac_threshold,
            maxIters=10000, confidence=0.999)
        if raw is not None:
            row["H_base_to_target_raw"] = resize_coordinate_matrix(base_size, target_size) @ raw
            row["H_base_to_target"] = row["H_base_to_target_raw"]
            row["inlier_mask_raw"] = mask.ravel().astype(bool)
            row["inliers"] = int(row["inlier_mask_raw"].sum())
            row["inlier_ratio"] = row["inliers"] / len(source_points)
    if len(source_points) < options.qa.min_matches:
        row["qa_error"] = f"Only {len(source_points)} mutual matches"
        return row
    try:
        cv2.setRNGSeed(options.qa.seed)
        quality = _fit_and_check(source_points, target_points, base_size, target_size, options.qa)
        row["qa_passed"] = True
        row["qa"] = quality
        row["inliers"] = quality["inliers"]
        row["inlier_ratio"] = quality["inlier_ratio"]
        row["H_base_to_target"] = np.asarray(quality["H_native_to_target"], np.float64)
    except (AlignmentError, cv2.error) as exc:
        row["qa_error"] = str(exc)
    return row


def assess_pan_rgb_grid(pan, target_rgb, options=None):
    """Independently estimate PAN-to-RGB; equal dimensions never prove identity."""
    options = options or PanOptions()
    target_rgb = _check_rgb(target_rgb, "target_rgb")
    pan_gray = _stretch_pan(pan)
    ph, pw = pan_gray.shape
    th, tw = target_rgb.shape[:2]
    shrink = min(1.0, options.grid_max_side / max(pw, ph))
    base_size = (max(16, round(pw * shrink)), max(16, round(ph * shrink)))
    source_base = cv2.resize(pan_gray, base_size, interpolation=cv2.INTER_AREA)
    target_base = cv2.resize(target_rgb, base_size, interpolation=cv2.INTER_AREA)
    pan_to_base = resize_coordinate_matrix((pw, ph), base_size)
    base_to_pan = np.linalg.inv(pan_to_base)
    attempts = []
    for mode, ratio in (("gray", 0.75), ("gray_clahe", 0.80)):
        result = _trial(source_base, target_base, (tw, th), "sift", mode, 1, ratio, options)
        result["points_src_pan"] = _points(result.pop("points_src_base"), base_to_pan)
        result["points_dst_rgb"] = result.pop("points_dst_target")
        result["qa_source_grid"] = "resized_pan_matching_grid"
        result["qa_target_grid"] = "rgb_native"
        result["H_pan_to_rgb_raw"] = (
            result["H_base_to_target_raw"] @ pan_to_base
            if result["H_base_to_target_raw"] is not None else None)
        result["H_pan_to_rgb"] = (result["H_base_to_target"] @ pan_to_base
                                   if result["H_base_to_target"] is not None else None)
        attempts.append(result)
    passed = [attempt for attempt in attempts if attempt["qa_passed"]]
    grid = {"verified": False, "same_grid": False, "H_pan_to_rgb": None,
            "points_src_pan": np.empty((0, 2), np.float32),
            "points_dst_rgb": np.empty((0, 2), np.float32),
            "pan_size_wh": (pw, ph), "rgb_size_wh": (tw, th),
            "attempts": attempts}
    if not passed:
        grid["reason"] = "PAN-to-RGB matching did not pass independent QA"
        return grid
    chosen = min(passed, key=lambda attempt: attempt["qa"]["holdout_median_rgb_px"])
    source_points, target_points = chosen["points_src_pan"], chosen["points_dst_rgb"]
    h = chosen["H_pan_to_rgb"]
    residual = np.linalg.norm(_points(source_points, h) - target_points, axis=1)
    # Assess identity using inliers only, while keeping all points for the caller.
    native_threshold = options.qa.ransac_threshold * max(tw / base_size[0], th / base_size[1])
    supported = residual <= native_threshold
    identity_error = np.linalg.norm(source_points[supported] - target_points[supported], axis=1)
    coordinates = np.array([(x, y) for y in np.linspace(0, ph - 1, 5)
                            for x in np.linspace(0, pw - 1, 5)], np.float32)
    model_displacement = np.linalg.norm(_points(coordinates, h) - coordinates, axis=1)
    median = float(np.median(identity_error)) if len(identity_error) else float("inf")
    p90 = float(np.percentile(identity_error, 90)) if len(identity_error) else float("inf")
    maximum = float(np.max(model_displacement))
    same_grid = ((pw, ph) == (tw, th) and median <= options.same_grid_median_px
                 and p90 <= options.same_grid_p90_px
                 and maximum <= options.same_grid_model_max_px)
    grid.update({"verified": True, "same_grid": same_grid, "H_pan_to_rgb": h,
                 "points_src_pan": source_points, "points_dst_rgb": target_points,
                 "strategy": chosen["strategy"], "qa": chosen["qa"],
                 "identity_median_px": median, "identity_p90_px": p90,
                 "model_displacement_max_px": maximum,
                 "reason": "Identity supported" if same_grid else "Verified nonidentity mapping"})
    return grid


def estimate_pan_candidates(source_rgb, pan, target_rgb, options=None):
    """Return at most six MSI-to-PAN candidates and an independent grid check.

    Candidate fields include points_src_native, points_dst_pan, strategy,
    mutual_matches, qa_passed, qa_error, qa, and H_native_to_pan[_raw].
    H_native_to_rgb and points_dst_rgb are populated only when the separate
    PAN-to-RGB mapping passed QA. That does not accept the MSI-to-PAN candidate:
    callers must still inspect qa_passed or apply their own common evaluation.
    """
    options = options or PanOptions()
    if not 1 <= options.max_candidates <= len(_STRATEGIES):
        raise ValueError("max_candidates must be between one and six")
    source_rgb = _check_rgb(source_rgb, "source_rgb")
    target_rgb = _check_rgb(target_rgb, "target_rgb")
    pan_gray = _stretch_pan(pan)
    ph, pw = pan_gray.shape
    sh, sw = source_rgb.shape[:2]
    pan_base = cv2.resize(pan_gray, (sw, sh), interpolation=cv2.INTER_AREA)
    grid = assess_pan_rgb_grid(pan, target_rgb, options)
    candidates = []
    for detector, mode, scale, ratio in _STRATEGIES[:options.max_candidates]:
        candidate = _trial(source_rgb, pan_base, (pw, ph), detector, mode, scale, ratio, options)
        candidate["points_src_native"] = candidate.pop("points_src_base")
        candidate["points_dst_pan"] = candidate.pop("points_dst_target")
        candidate["H_native_to_pan_raw"] = candidate.pop("H_base_to_target_raw")
        candidate["H_native_to_pan"] = candidate.pop("H_base_to_target")
        candidate["qa_source_grid"] = "msi_native"
        candidate["qa_target_grid"] = "pan_native"
        candidate["H_native_to_rgb"] = None
        candidate["points_dst_rgb"] = None
        if grid["verified"]:
            candidate["points_dst_rgb"] = _points(candidate["points_dst_pan"], grid["H_pan_to_rgb"])
            if candidate["H_native_to_pan"] is not None:
                candidate["H_native_to_rgb"] = grid["H_pan_to_rgb"] @ candidate["H_native_to_pan"]
        candidates.append(candidate)
    return {"grid": grid, "candidates": candidates}
