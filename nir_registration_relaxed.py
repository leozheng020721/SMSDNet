"""Fit UNVERIFIED image-registration candidates with basic geometry safeguards.

This module deliberately does not require spatial holdout, image coverage, or
occupied-cell thresholds. RANSAC support and residuals use the fitting matches;
they are training diagnostics, never independent registration validation.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from align_seo_nir import AlignmentError, resize_coordinate_matrix, validate_homography
from nir_registration_metrics import _balanced_subset


@dataclass(frozen=True)
class RelaxedOptions:
    min_matches: int = 8
    min_inliers: int = 6
    min_inlier_ratio: float = 0.2
    ransac_threshold_msi: float = 2.5
    min_valid_fraction: float = 0.10
    max_points: int = 3000
    balance_grid: tuple = (4, 4)
    seed: int = 42
    max_iterations: int = 10000
    ransac_confidence: float = 0.999
    models: tuple = ("partial_affine", "affine", "homography")
    min_axis_std_msi: float = 0.25
    min_singular_value_ratio: float = 1e-3
    duplicate_precision: int = 6
    complexity_support_gain: float = 0.10
    complexity_min_extra_inliers: int = 2
    complexity_residual_gain: float = 0.25
    complexity_min_residual_drop_msi: float = 0.15
    comparable_support_fraction: float = 0.95


_DOF = {"partial_affine": 4, "affine": 6, "homography": 8}


def _opencv_out_of_memory(exc):
    if not isinstance(exc, cv2.error):
        return False
    message = str(exc).lower()
    return (getattr(exc, "code", None) == -4
            or any(token in message for token in ("memory", "outofmemory", "allocat")))


def _size(values):
    if len(values) != 2:
        raise ValueError("Image sizes must contain width and height")
    if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
           or int(value) != value or value < 2 for value in values):
        raise ValueError("Image dimensions must be integers of at least two pixels")
    return tuple(int(value) for value in values)


def _check_options(options):
    if options.min_matches < 8 or options.min_inliers < 6:
        raise ValueError("At least eight matches and six inliers are required")
    if not options.min_inliers <= options.min_matches <= options.max_points <= 3000:
        raise ValueError("Require min_inliers <= min_matches <= max_points <= 3000")
    if not 0 < options.min_inlier_ratio <= 1 or options.ransac_threshold_msi <= 0:
        raise ValueError("Inlier ratio and RANSAC threshold must be positive")
    if not 0 <= options.min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between zero and one")
    if len(options.balance_grid) != 2 or any(value <= 0 for value in options.balance_grid):
        raise ValueError("balance_grid must contain two positive cell counts")
    if not options.models or any(model not in _DOF for model in options.models):
        raise ValueError("Unknown or empty model selection")
    if not 0 < options.ransac_confidence < 1 or options.max_iterations < 1:
        raise ValueError("Invalid RANSAC confidence or iteration count")
    if options.min_axis_std_msi <= 0 or not 0 < options.min_singular_value_ratio < 1:
        raise ValueError("Invalid two-dimensional identifiability thresholds")
    if not 0 <= options.complexity_support_gain <= 1 or options.complexity_min_extra_inliers < 1:
        raise ValueError("Invalid model-complexity support thresholds")
    if (not 0 <= options.complexity_residual_gain < 1
            or options.complexity_min_residual_drop_msi < 0
            or not 0 < options.comparable_support_fraction <= 1):
        raise ValueError("Invalid model-complexity residual thresholds")


def _transform(points, h):
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[:, None], h)[:, 0]


def _two_dimensional(points, options):
    centered = np.asarray(points, np.float64) - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) != 2 or not np.isfinite(singular).all() or singular[0] <= 0:
        return False
    return bool(singular[1] / math.sqrt(len(points)) >= options.min_axis_std_msi
                and singular[1] / singular[0] >= options.min_singular_value_ratio)


def _unique_indices(source, target, confidence, precision):
    order = np.arange(len(source))
    if confidence is not None:
        order = order[np.argsort(-confidence, kind="stable")]
    seen_source, seen_target, selected = set(), set(), []
    source_keys, target_keys = np.round(source, precision), np.round(target, precision)
    for index in order:
        source_key, target_key = tuple(source_keys[index]), tuple(target_keys[index])
        if source_key in seen_source or target_key in seen_target:
            continue
        seen_source.add(source_key)
        seen_target.add(target_key)
        selected.append(index)
    return np.asarray(selected, int)


def _hull_fraction(points, size):
    return float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32))) / np.prod(size))


def _geometry(h_native, source_size, target_size):
    h_native = validate_homography(h_native, source_size, target_size)
    sw, sh = source_size
    tw, th = target_size
    source_corners = np.float64([[-0.5, -0.5], [sw - 0.5, -0.5],
                                [sw - 0.5, sh - 0.5], [-0.5, sh - 0.5]])
    target_corners = np.float32([[-0.5, -0.5], [tw - 0.5, -0.5],
                                [tw - 0.5, th - 0.5], [-0.5, th - 0.5]])
    mapped = _transform(source_corners, h_native).astype(np.float32)
    overlap, _ = cv2.intersectConvexConvex(mapped, target_corners)
    return h_native, {
        "footprint_area_ratio": float(cv2.contourArea(mapped, oriented=True) / (tw * th)),
        "footprint_overlap_fraction": float(np.clip(overlap / (tw * th), 0, 1)),
        "footprint_corners_target_xy": mapped.tolist(),
        "actual_valid_mask_checked": False,
        "valid_fraction_caution": "Geometric footprint overlap is an upper bound; driver must check the actual warped valid mask",
    }


def _estimate(model, source, working, options):
    cv2.setRNGSeed(options.seed)
    if model == "homography":
        return cv2.findHomography(source, working, cv2.RANSAC,
                                  options.ransac_threshold_msi,
                                  maxIters=options.max_iterations,
                                  confidence=options.ransac_confidence)
    estimator = cv2.estimateAffinePartial2D if model == "partial_affine" else cv2.estimateAffine2D
    affine, mask = estimator(source, working, method=cv2.RANSAC,
                             ransacReprojThreshold=options.ransac_threshold_msi,
                             maxIters=options.max_iterations,
                             confidence=options.ransac_confidence, refineIters=10)
    return (np.vstack((affine, (0, 0, 1))) if affine is not None else None), mask


def _fit_model(model, source, working, source_size, target_size, options):
    row = {"model": model, "model_dof": _DOF[model], "accepted": False,
           "H_native_to_target": None, "reason": None, "metrics": {}}
    try:
        h, raw_mask = _estimate(model, source, working, options)
        if h is None or raw_mask is None:
            raise AlignmentError("RANSAC could not estimate this model")
        h_full, geometry = _geometry(
            resize_coordinate_matrix(source_size, target_size) @ h,
            source_size, target_size)
        row["metrics"].update(geometry)
        if geometry["footprint_overlap_fraction"] < options.min_valid_fraction:
            raise AlignmentError("Geometric overlap is below min_valid_fraction")
        predicted = _transform(source, h)
        errors = np.linalg.norm(predicted - working, axis=1)
        raw_mask = raw_mask.ravel().astype(bool)
        inliers = raw_mask & np.isfinite(errors) & (errors <= options.ransac_threshold_msi)
        count, ratio = int(inliers.sum()), float(inliers.mean())
        row["metrics"].update(ransac_inliers=int(raw_mask.sum()), inliers=count,
                               inlier_ratio=ratio)
        if count < options.min_inliers or ratio < options.min_inlier_ratio:
            raise AlignmentError("RANSAC inlier support is below the relaxed requirements")
        if (not _two_dimensional(source[inliers], options)
                or not _two_dimensional(working[inliers], options)):
            raise AlignmentError("Model inliers are collinear or lack two-dimensional identifiability")
        row["metrics"].update({
            "training_all_median_msi_px": float(np.median(errors)),
            "training_all_p90_msi_px": float(np.percentile(errors, 90)),
            "training_inlier_median_msi_px": float(np.median(errors[inliers])),
            "training_inlier_p90_msi_px": float(np.percentile(errors[inliers], 90)),
            "source_inlier_hull_fraction": _hull_fraction(source[inliers], source_size),
            "target_inlier_hull_fraction": _hull_fraction(working[inliers], source_size),
            "fit_scope": "All selected matching points; no untouched validation subset",
            "residual_kind": "TRAINING_RESIDUAL",
        })
        row.update(accepted=True, H_native_to_target=h_full.tolist(),
                   H_native_to_resized_target=h.tolist(), _errors=errors, _inliers=inliers)
    except (AlignmentError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        if _opencv_out_of_memory(exc):
            raise
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def _select(candidates, options):
    chosen = None
    selection_reason = None
    for candidate in sorted(candidates, key=lambda row: row["model_dof"]):
        if not candidate["accepted"]:
            continue
        if chosen is None:
            chosen, selection_reason = candidate, "Simplest geometrically acceptable supported model"
            continue
        best_count = chosen["metrics"]["inliers"]
        new_count = candidate["metrics"]["inliers"]
        best_median = chosen["metrics"]["training_inlier_median_msi_px"]
        new_median = candidate["metrics"]["training_inlier_median_msi_px"]
        extra = max(options.complexity_min_extra_inliers,
                    math.ceil(options.complexity_support_gain * best_count))
        support_gain = (new_count >= best_count + extra
                        and new_median <= max(best_median + 0.5,
                                              0.6 * options.ransac_threshold_msi))
        new_on_old_support = float(np.median(candidate["_errors"][chosen["_inliers"]]))
        required_drop = max(options.complexity_min_residual_drop_msi,
                            options.complexity_residual_gain * best_median)
        residual_gain = (new_count >= options.comparable_support_fraction * best_count
                         and best_median - new_on_old_support >= required_drop)
        if support_gain or residual_gain:
            chosen = candidate
            selection_reason = ("More complex model materially improves supported matches"
                                if support_gain else "More complex model materially reduces training residuals on prior support")
    return chosen, selection_reason


def fit_relaxed(source_xy, target_xy, source_size_wh, target_size_wh,
                confidence=None, options=None):
    """Return an UNVERIFIED candidate fitted in native-MSI error units.

    source_xy and target_xy must be matching Nx2 native pixel-center coordinates.
    Target points are mapped to a target working grid with the MSI dimensions,
    including the half-pixel translation. Duplicate source/target locations do
    not count as extra observations. No identity or resize fallback is created.

    accepted means only the relaxed support and basic geometry checks passed.
    The caller must check actual warped validity before saving, mark the output
    UNVERIFIED, and assess PAN-to-current-RGB correspondence separately.
    """
    options = options or RelaxedOptions()
    metrics = {"validation_status": "UNVERIFIED", "independently_validated": False,
               "spatial_holdout_required": False, "coverage_threshold_required": False,
               "residual_kind": "TRAINING_RESIDUAL",
               "caution": "Accepted is not verified: all fitting points can share matcher errors"}
    result = {"accepted": False, "H_native_to_target": None, "model": None,
              "reason": None, "metrics": metrics, "candidates": [],
              "validation_status": "UNVERIFIED"}
    try:
        _check_options(options)
        source_size, target_size = _size(source_size_wh), _size(target_size_wh)
        source, target = np.asarray(source_xy, np.float64), np.asarray(target_xy, np.float64)
        if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
            raise ValueError("Correspondences must be matching Nx2 arrays")
        scores = None if confidence is None else np.asarray(confidence, np.float64)
        if scores is not None and scores.shape != (len(source),):
            raise ValueError("confidence must contain one score per correspondence")
        metrics.update(input_matches=len(source), source_size_wh=list(source_size),
                       target_size_wh=list(target_size), max_points=options.max_points,
                       ransac_threshold_msi=options.ransac_threshold_msi)
        finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if scores is not None:
            finite &= np.isfinite(scores)
        sw, sh = source_size
        tw, th = target_size
        inside = ((source[:, 0] >= 0) & (source[:, 0] < sw)
                  & (source[:, 1] >= 0) & (source[:, 1] < sh)
                  & (target[:, 0] >= 0) & (target[:, 0] < tw)
                  & (target[:, 1] >= 0) & (target[:, 1] < th))
        metrics.update(nonfinite_matches=int((~finite).sum()),
                       out_of_bounds_matches=int((finite & ~inside).sum()))
        source, target = source[finite & inside], target[finite & inside]
        if scores is not None:
            scores = scores[finite & inside]
        unique = _unique_indices(source, target, scores, options.duplicate_precision)
        metrics["duplicate_matches_removed"] = len(source) - len(unique)
        source, target = source[unique], target[unique]
        if scores is not None:
            scores = scores[unique]
        selected = _balanced_subset(source, source_size, options.balance_grid,
                                     options.max_points, scores, options.seed)
        source, target = source[selected], target[selected]
        metrics["selected_matches"] = len(source)
        if len(source) < options.min_matches:
            raise AlignmentError(f"Only {len(source)} unique finite in-bounds correspondences")
        working = _transform(target, resize_coordinate_matrix(target_size, source_size))
        if not _two_dimensional(source, options) or not _two_dimensional(working, options):
            raise AlignmentError("Matching points are collinear or lack two-dimensional identifiability")
        candidates = [_fit_model(model, source, working, source_size, target_size, options)
                      for model in sorted(set(options.models), key=_DOF.__getitem__)]
        chosen, selection_reason = _select(candidates, options)
        if chosen is not None:
            metrics.update(chosen["metrics"], model_dof=chosen["model_dof"],
                           selection_reason=selection_reason)
            result.update(accepted=True, H_native_to_target=chosen["H_native_to_target"],
                          H_native_to_resized_target=chosen["H_native_to_resized_target"],
                          model=chosen["model"], reason=None)
        else:
            result["reason"] = "No supported model passed basic geometry and relaxed support checks"
        result["candidates"] = [{key: value for key, value in candidate.items()
                                  if not key.startswith("_")} for candidate in candidates]
    except (AlignmentError, ValueError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        if _opencv_out_of_memory(exc):
            raise
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result
