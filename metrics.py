import numpy as np
import torch


def get_metrics(hist, positive_class=1):
    """Compute BER, IoU, F1, and Kappa for one target class.

    Rows in ``hist`` are ground truth and columns are predictions. Multiclass
    matrices are collapsed to target-vs-rest so all four metrics use the same
    samples. BER is returned as a percentage; undefined metrics are NaN.
    """
    hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim != 2 or hist.shape[0] != hist.shape[1]:
        raise ValueError("hist must be a square confusion matrix")
    if not 0 <= positive_class < hist.shape[0]:
        raise ValueError(
            f"positive_class must be in [0, {hist.shape[0] - 1}], got {positive_class}"
        )
    if np.any(hist < 0):
        raise ValueError("hist cannot contain negative values")

    total = hist.sum()
    if total == 0:
        return {"ber": np.nan, "iou": np.nan, "f1": np.nan, "kappa": np.nan}

    tp = hist[positive_class, positive_class]
    fn = hist[positive_class, :].sum() - tp
    fp = hist[:, positive_class].sum() - tp
    tn = total - tp - fn - fp

    positive_count = tp + fn
    negative_count = tn + fp
    union = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn

    if positive_count == 0 or negative_count == 0:
        ber = np.nan
    else:
        false_negative_rate = fn / positive_count
        false_positive_rate = fp / negative_count
        ber = 0.5 * (false_negative_rate + false_positive_rate) * 100.0

    iou = tp / union if union > 0 else np.nan
    f1 = 2 * tp / f1_denominator if f1_denominator > 0 else np.nan

    observed_agreement = (tp + tn) / total
    true_positive_rate = positive_count / total
    predicted_positive_rate = (tp + fp) / total
    expected_agreement = (
        true_positive_rate * predicted_positive_rate
        + (1.0 - true_positive_rate) * (1.0 - predicted_positive_rate)
    )
    kappa_denominator = 1.0 - expected_agreement
    kappa = (
        (observed_agreement - expected_agreement) / kappa_denominator
        if kappa_denominator > np.finfo(np.float64).eps
        else np.nan
    )

    return {
        "ber": float(ber),
        "iou": float(iou),
        "f1": float(f1),
        "kappa": float(kappa),
    }


def fast_hist(label_pred, label_true, num_classes):
    """Build a confusion matrix while ignoring invalid labels."""
    label_pred = torch.as_tensor(label_pred, dtype=torch.int64)
    label_true = torch.as_tensor(label_true, dtype=torch.int64, device=label_pred.device)

    if label_pred.shape != label_true.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {label_pred.shape} and {label_true.shape}"
        )

    mask = (
        (label_true >= 0)
        & (label_true < num_classes)
        & (label_pred >= 0)
        & (label_pred < num_classes)
    )
    hist = torch.bincount(
        num_classes * label_true[mask] + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist.detach().cpu().numpy()
