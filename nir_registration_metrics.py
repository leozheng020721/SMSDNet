"""Read-only diagnostics for comparing NIR registration candidates.

Structural scores are supporting evidence, not geometric ground truth. Dense
matches from one LoFTR run remain correlated even after a random train/test
split. Use spatially separated holdout regions and independently sourced
control points; do not refit on those points before reporting holdout error.
"""

import cv2
import numpy as np

from align_seo_nir import AlignmentError, resize_coordinate_matrix, validate_homography


def homography_geometry(h, source_size_wh, target_size_wh):
    """Return the production geometry checks without silently repairing H."""
    try:
        matrix = validate_homography(h, source_size_wh, target_size_wh)
    except (AlignmentError, ValueError, np.linalg.LinAlgError) as exc:
        return {"valid": False, "reason": str(exc)}
    width, height = source_size_wh
    corners = np.float64([[[-0.5, -0.5], [width - 0.5, -0.5],
                           [width - 0.5, height - 0.5], [-0.5, height - 0.5]]])
    mapped = cv2.perspectiveTransform(corners, matrix)[0]
    area = cv2.contourArea(mapped.astype(np.float32), oriented=True)
    return {"valid": True, "reason": None,
            "footprint_area_ratio": float(area / np.prod(target_size_wh)),
            "footprint_corners_target_xy": mapped.tolist()}


def _point_arrays(source_xy, target_xy):
    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError("Control points must be matching N x 2 arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Control points must be finite")
    return source, target


def spatial_holdout_split(source_xy, target_xy, source_size_wh, grid=(4, 4),
                          holdout_cells=None, guard_px=4.0):
    """Split whole native-source cells and remove training points near them.

    Fit only train_indices. Report test_indices BEFORE refitting on all points.
    The source-space guard is a minimum safeguard, not proof of independent
    observations; repeated texture and a shared matcher can still cause errors.
    """
    source, target = _point_arrays(source_xy, target_xy)
    width, height = source_size_wh
    cols, rows = grid
    if min(width, height, cols, rows) <= 0 or guard_px < 0:
        raise ValueError("Image/grid dimensions must be positive; guard must be nonnegative")
    if holdout_cells is None:
        holdout_cells = tuple(row * cols + row % cols for row in range(rows))
    held = np.asarray(sorted(set(holdout_cells)), dtype=int)
    if not len(held) or np.any(held < 0) or np.any(held >= cols * rows):
        raise ValueError("holdout_cells must contain valid flattened grid indices")
    inside = ((source[:, 0] >= 0) & (source[:, 0] < width)
              & (source[:, 1] >= 0) & (source[:, 1] < height))
    cell_x = np.floor(source[:, 0] * cols / width).astype(int)
    cell_y = np.floor(source[:, 1] * rows / height).astype(int)
    test = inside & np.isin(cell_y * cols + cell_x, held)
    train = inside & ~test
    for cell in held:
        row, col = divmod(int(cell), cols)
        x0, x1 = col * width / cols, (col + 1) * width / cols
        y0, y1 = row * height / rows, (row + 1) * height / rows
        dx = np.maximum(np.maximum(x0 - source[:, 0], source[:, 0] - x1), 0)
        dy = np.maximum(np.maximum(y0 - source[:, 1], source[:, 1] - y1), 0)
        train &= dx * dx + dy * dy >= guard_px * guard_px
    return {"train_indices": np.flatnonzero(train), "test_indices": np.flatnonzero(test),
            "excluded_indices": np.flatnonzero(~train & ~test), "grid": list(grid),
            "holdout_cells": held.tolist(), "guard_source_px": float(guard_px),
            "caution": "Spatially separated matches from one matcher are not external ground truth."}


def _cell_ids(points, size_wh, grid):
    return (np.floor(points[:, 1] * grid[1] / size_wh[1]).astype(int) * grid[0]
            + np.floor(points[:, 0] * grid[0] / size_wh[0]).astype(int))


def _hull_fraction(points, size_wh):
    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / np.prod(size_wh))


def _balanced_subset(points, size_wh, grid, limit, confidence, seed):
    """Cap dense matches with round-robin sampling across fixed source cells."""
    if len(points) <= limit:
        return np.arange(len(points))
    cells = _cell_ids(points, size_wh, grid)
    rng = np.random.default_rng(seed)
    buckets = []
    for cell in sorted(np.unique(cells)):
        indices = rng.permutation(np.flatnonzero(cells == cell))
        if confidence is not None:
            indices = indices[np.argsort(-confidence[indices], kind="stable")]
        buckets.append(list(indices))
    selected = []
    depth = 0
    while len(selected) < limit:
        for bucket in buckets:
            if depth < len(bucket):
                selected.append(bucket[depth])
                if len(selected) == limit:
                    break
        depth += 1
    return np.asarray(selected, int)


def fit_and_validate_spatial(source_xy, target_xy, source_size_wh, target_size_wh,
                             confidence=None, max_points=3000, seed=42, grid=(4, 4),
                             holdout_cells=None, guard_px=4.0, ransac_threshold=1.5,
                             min_matches=20, min_inliers=15, min_inlier_ratio=0.6,
                             min_coverage=0.05, min_holdout_points=8,
                             min_train_cells=4, min_holdout_cells=3,
                             max_holdout_median=0.75, max_holdout_p90=3.0,
                             min_holdout_fraction=0.65):
    """Fit training blocks and validate untouched blocks in native-MSI pixels.

    source_xy must already use native MSI coordinates; target_xy must use the
    original target RGB/PAN grid, not a matcher-resized tensor. The returned H
    maps those native source coordinates to original target coordinates. This
    function never refits on holdout matches. max_points is capped at 3000.
    Both training inliers and holdout points must cover a two-dimensional area.
    """
    metrics = {"source_size_wh": list(source_size_wh), "target_size_wh": list(target_size_wh),
               "grid": list(grid), "guard_source_px": float(guard_px),
               "evaluation_units": "native MSI pixels on resized target grid",
               "holdout_is_external_ground_truth": False,
               "caution": "Spatial LoFTR/PAN-match holdout reduces leakage but shares its matcher's errors."}
    result = {"accepted": False, "reason": None, "H_native_to_target": None, "metrics": metrics}
    try:
        source, target = _point_arrays(source_xy, target_xy)
        if not 1 <= max_points <= 3000:
            raise ValueError("max_points must be between 1 and 3000")
        if min(*source_size_wh, *target_size_wh) <= 0:
            raise ValueError("Image dimensions must be positive")
        metrics["input_matches"] = len(source)
        scores = None if confidence is None else np.asarray(confidence, np.float64)
        if scores is not None and (scores.shape != (len(source),) or not np.isfinite(scores).all()):
            raise ValueError("confidence must contain one finite score per correspondence")
        sw, sh = source_size_wh
        tw, th = target_size_wh
        inside = ((source[:, 0] >= 0) & (source[:, 0] < sw)
                  & (source[:, 1] >= 0) & (source[:, 1] < sh)
                  & (target[:, 0] >= 0) & (target[:, 0] < tw)
                  & (target[:, 1] >= 0) & (target[:, 1] < th))
        metrics["out_of_bounds_matches"] = int((~inside).sum())
        source, target = source[inside], target[inside]
        if scores is not None:
            scores = scores[inside]
        indices = _balanced_subset(source, source_size_wh, grid, max_points, scores, seed)
        source, target = source[indices], target[indices]
        metrics["selected_matches"] = len(source)
        if len(source) < min_matches:
            raise AlignmentError(f"Only {len(source)} in-bounds selected correspondences")
        split = spatial_holdout_split(source, target, source_size_wh, grid, holdout_cells, guard_px)
        train, holdout = split["train_indices"], split["test_indices"]
        metrics.update(train_count=len(train), holdout_count=len(holdout),
                       guard_excluded_count=len(split["excluded_indices"]), holdout_cells=split["holdout_cells"])
        if len(train) < max(4, min_inliers) or len(holdout) < min_holdout_points:
            raise AlignmentError("Too few spatially separated training or holdout points")
        working = cv2.perspectiveTransform(target[None], resize_coordinate_matrix(target_size_wh, source_size_wh))[0]
        holdout_coverages = [_hull_fraction(points[holdout], source_size_wh) for points in (source, working)]
        source_cells = _cell_ids(source, source_size_wh, grid)
        holdout_cell_count = len(np.unique(source_cells[holdout]))
        metrics.update(holdout_source_coverage=holdout_coverages[0], holdout_target_coverage=holdout_coverages[1],
                       holdout_occupied_cells=holdout_cell_count)
        if min(holdout_coverages) < min_coverage or holdout_cell_count < min_holdout_cells:
            raise AlignmentError("Holdout points do not cover enough two-dimensional image area")
        cv2.setRNGSeed(seed)
        h, inliers = cv2.findHomography(source[train], working[train], cv2.RANSAC,
                                       ransac_threshold, maxIters=10000, confidence=0.999)
        if h is None or inliers is None:
            raise AlignmentError("Spatial training RANSAC could not estimate a homography")
        h = validate_homography(h, source_size_wh, source_size_wh)
        selected = train[inliers.ravel().astype(bool)]
        coverage = [_hull_fraction(points[selected], source_size_wh) for points in (source, working)]
        occupied = len(np.unique(source_cells[selected]))
        metrics.update(inliers=len(selected), inlier_ratio=len(selected) / len(train),
                       source_coverage=coverage[0], target_coverage=coverage[1], train_occupied_cells=occupied)
        if len(selected) < min_inliers or metrics["inlier_ratio"] < min_inlier_ratio:
            raise AlignmentError("Spatial training inlier support is insufficient")
        if min(coverage) < min_coverage or occupied < min_train_cells:
            raise AlignmentError("Training inliers do not cover enough two-dimensional image area")
        predicted = cv2.perspectiveTransform(source[holdout, None], h)[:, 0]
        errors = np.linalg.norm(predicted - working[holdout], axis=1)
        metrics.update(holdout_median_msi_px=float(np.median(errors)),
                       holdout_p90_msi_px=float(np.percentile(errors, 90)),
                       holdout_inlier_fraction=float(np.mean(errors <= ransac_threshold)))
        if (metrics["holdout_median_msi_px"] > max_holdout_median
                or metrics["holdout_p90_msi_px"] > max_holdout_p90
                or metrics["holdout_inlier_fraction"] < min_holdout_fraction):
            raise AlignmentError("Spatial holdout error exceeds the validation thresholds")
        full_h = resize_coordinate_matrix(source_size_wh, target_size_wh) @ h
        full_h = validate_homography(full_h, source_size_wh, target_size_wh)
        result.update(accepted=True, H_native_to_target=full_h.tolist(), H_native_to_resized_target=h.tolist())
    except (AlignmentError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def evaluate_control_points(h, source_xy, target_xy, target_size_wh, grid=(4, 4),
                            provenance="unspecified", used_for_fit=None):
    """Measure all supplied points, including outliers, without filtering by H.

    provenance and used_for_fit are declarations by the caller. Keep evaluation
    points outside estimation and candidate selection. Cell medians prevent a
    single dense cluster from being the only apparent source of good accuracy.
    """
    source, target = _point_arrays(source_xy, target_xy)
    width, height = target_size_wh
    cols, rows = grid
    if min(width, height, cols, rows) <= 0:
        raise ValueError("Image/grid dimensions must be positive")
    result = {"count": len(source), "provenance": provenance, "used_for_fit": used_for_fit,
              "median_target_px": None, "p90_target_px": None, "maximum_target_px": None,
              "occupied_target_cells": 0, "cell_medians_target_px": {},
              "target_convex_hull_fraction": 0.0,
              "caution": "Shared-matcher or fit-used points are not independent geometric ground truth."}
    if not len(source):
        return result
    matrix = np.asarray(h, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("H must be a finite 3 x 3 matrix")
    projected = np.c_[source, np.ones(len(source))] @ matrix.T
    if np.any(np.abs(projected[:, 2]) < 1e-10):
        raise AlignmentError("Control point projection crosses a projective horizon")
    predicted = projected[:, :2] / projected[:, 2, None]
    errors = np.linalg.norm(predicted - target, axis=1)
    result.update(median_target_px=float(np.median(errors)), p90_target_px=float(np.percentile(errors, 90)),
                  maximum_target_px=float(errors.max()),
                  unique_source_pixel_count=len(np.unique(np.rint(source), axis=0)),
                  unique_target_pixel_count=len(np.unique(np.rint(target), axis=0)))
    inside = ((target[:, 0] >= 0) & (target[:, 0] < width)
              & (target[:, 1] >= 0) & (target[:, 1] < height))
    cell_ids = (np.floor(target[:, 1] * rows / height).astype(int) * cols
                + np.floor(target[:, 0] * cols / width).astype(int))
    cells = {str(int(cell)): float(np.median(errors[inside & (cell_ids == cell)]))
             for cell in np.unique(cell_ids[inside])}
    result["cell_medians_target_px"] = cells
    result["occupied_target_cells"] = len(cells)
    result["outside_target_count"] = int((~inside).sum())
    if cells:
        result["median_of_cell_medians_target_px"] = float(np.median(list(cells.values())))
        result["maximum_cell_median_target_px"] = max(cells.values())
    if inside.sum() >= 3:
        hull = cv2.convexHull(target[inside].astype(np.float32))
        result["target_convex_hull_fraction"] = float(cv2.contourArea(hull) / (width * height))
    return result


def _gray(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    if array.ndim == 2:
        return array
    raise ValueError("Matching images must be grayscale or three-channel RGB arrays")


def _normalize(gray, mask):
    low, high = np.percentile(gray[mask], (1, 99))
    return np.clip((gray - low) / max(float(high - low), 1e-6), 0, 1).astype(np.float32)


def _correlation(first, second):
    if len(first) < 2 or min(float(np.std(first)), float(np.std(second))) < 1e-8:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def registration_support_mask(source_valid, h_native_to_target, target_size_wh):
    """Return fully supported bilinear pixels for a shared comparison mask."""
    source_valid = np.asarray(source_valid, bool)
    if source_valid.ndim != 2:
        raise ValueError("source_valid must be a two-dimensional mask")
    h = validate_homography(h_native_to_target,
                            (source_valid.shape[1], source_valid.shape[0]), target_size_wh)
    weights = cv2.warpPerspective(source_valid.astype(np.float32), h, target_size_wh,
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return weights >= 1 - 1e-6


def evaluate_registration(source_rgb, target_rgb, h_native_to_target, source_valid=None,
                          control_points=None, black_threshold=0, max_side=512, evaluation_mask=None):
    """Report geometry, usable coverage, and coarse-grid structural diagnostics.

    source_rgb is the native MSI visible matching image, not NIR. A candidate
    must use the same source_valid mask and evaluation settings as its rival.
    For comparative structural scores, pass the intersection of candidate
    support masks as evaluation_mask, so both scores use the same image area.
    control_points may contain source_xy, target_xy, provenance, used_for_fit.
    Returned scores do not automatically accept a registration candidate.
    """
    source_gray, target_gray = _gray(source_rgb), _gray(target_rgb)
    sh, sw = source_gray.shape
    th, tw = target_gray.shape
    geometry = homography_geometry(h_native_to_target, (sw, sh), (tw, th))
    result = {"geometry": geometry, "structural_scores_are_ground_truth": False}
    if not geometry["valid"]:
        return result
    if max_side < 1:
        raise ValueError("max_side must be positive")
    h = np.asarray(h_native_to_target, np.float64)
    valid_source = np.isfinite(source_gray)
    if source_valid is not None:
        if np.shape(source_valid) != source_gray.shape:
            raise ValueError("source_valid must match the native matching image")
        valid_source &= np.asarray(source_valid, bool)
    target_array = np.asarray(target_rgb)
    content = (np.any(target_array > black_threshold, axis=2) if target_array.ndim == 3
               else target_array > black_threshold) & np.isfinite(target_gray)
    kwargs = dict(flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    supported = registration_support_mask(valid_source, h, (tw, th))
    content_pixels = int(content.sum())
    result.update(valid_fraction=float(supported.mean()), target_nonblack_fraction=float(content.mean()),
                  supported_rgb_content_fraction=(float((supported & content).sum() / content_pixels)
                                                  if content_pixels else None),
                  supported_content_pixels=int((supported & content).sum()))
    # Compare at no finer than the native MSI scale to limit sharpening bias.
    scale = min(1.0, max_side / max(tw, th), sw / tw, sh / th)
    eval_size = (max(1, round(tw * scale)), max(1, round(th * scale)))
    eval_h = resize_coordinate_matrix((tw, th), eval_size) @ h
    aligned = cv2.warpPerspective(np.where(valid_source, source_gray, 0), eval_h, eval_size, **kwargs)
    target = cv2.resize(np.nan_to_num(target_gray), eval_size, interpolation=cv2.INTER_AREA)
    structure_area = supported & content
    if evaluation_mask is not None:
        if np.shape(evaluation_mask) != (th, tw):
            raise ValueError("evaluation_mask must match the target pixel grid")
        structure_area &= np.asarray(evaluation_mask, bool)
    area_mask = cv2.resize(structure_area.astype(np.float32), eval_size,
                           interpolation=cv2.INTER_AREA) >= 1 - 1e-6
    area_mask = cv2.erode(area_mask.astype(np.uint8), np.ones((9, 9), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    result.update(structure_evaluation_size_wh=list(eval_size),
                  structure_common_mask_supplied=evaluation_mask is not None,
                  structure_valid_pixels=int(area_mask.sum()), gradient_magnitude_correlation=None,
                  gradient_orientation_agreement=None, symmetric_edge_distance_eval_px=None,
                  symmetric_edge_distance_target_px_approx=None)
    if area_mask.sum() >= 64:
        first = cv2.GaussianBlur(_normalize(aligned, area_mask), (0, 0), 0.8)
        second = cv2.GaussianBlur(_normalize(target, area_mask), (0, 0), 0.8)
        gradients = [(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
                     for gray in (first, second)]
        magnitudes = [cv2.magnitude(dx, dy) for dx, dy in gradients]
        result["gradient_magnitude_correlation"] = _correlation(
            magnitudes[0][area_mask], magnitudes[1][area_mask])
        weight = np.minimum(magnitudes[0], magnitudes[1])
        use = area_mask & (weight > 1e-5)
        if use.any():
            dot = gradients[0][0] * gradients[1][0] + gradients[0][1] * gradients[1][1]
            agreement = np.abs(dot[use]) / np.maximum(magnitudes[0][use] * magnitudes[1][use], 1e-12)
            result["gradient_orientation_agreement"] = float(np.average(np.clip(agreement, 0, 1), weights=weight[use]))
        edges = [(magnitude >= max(float(np.percentile(magnitude[area_mask], 85)), 1e-5)) & area_mask
                 for magnitude in magnitudes]
        if all(edge.any() for edge in edges):
            distances = [cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                         for edge in edges]
            distance = (float(distances[0][edges[1]].mean()) + float(distances[1][edges[0]].mean())) / 2
            result["symmetric_edge_distance_eval_px"] = distance
            result["symmetric_edge_distance_target_px_approx"] = distance * np.sqrt((tw / eval_size[0]) * (th / eval_size[1]))
    if control_points is not None:
        result["control_points"] = evaluate_control_points(
            h, control_points["source_xy"], control_points["target_xy"], (tw, th),
            provenance=control_points.get("provenance", "unspecified"),
            used_for_fit=control_points.get("used_for_fit"))
    return result
