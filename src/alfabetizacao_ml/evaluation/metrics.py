"""Metrics preserve survey weights and distinguish risk from literacy."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def _arrays(y, p, weights=None):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if not len(y) or len(y) != len(p) or len(y) != len(weights):
        raise ValueError("Metrics require equally sized, nonempty arrays")
    if not (np.isfinite(p).all() and np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Metrics require finite predictions and finite positive weights")
    return y, p, weights


def weighted_calibration_curve(y, p, weights, bins=10) -> list[dict]:
    y, p, weights = _arrays(y, p, weights)
    if not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Calibration requires binary labels and probabilities in [0,1]")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("Calibration bins must be a positive integer")
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    result = []
    for i in range(bins):
        mask = assignment == i
        if mask.any():
            result.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                           "predicted": float(np.average(p[mask], weights=weights[mask])),
                           "observed": float(np.average(y[mask], weights=weights[mask])),
                           "weight": float(weights[mask].sum()), "count": int(mask.sum())})
    return result


def select_f2_threshold(y, p, weights) -> dict:
    """Exact weighted scan: ties prefer recall, then the smallest threshold."""
    y, p, weights = _arrays(y, p, weights)
    if not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Binary labels and probabilities in [0,1] are required")
    order = np.argsort(-p, kind="stable")
    ps, ys, ws = p[order], y[order], weights[order]
    last = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    tp = np.cumsum(ws * ys)[last]
    fp = np.cumsum(ws * (1 - ys))[last]
    positives = float(np.sum(weights * y))
    fn = positives - tp
    denom = 5 * tp + 4 * fn + fp
    f2 = np.divide(5 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    recall = tp / positives if positives else np.zeros_like(tp)
    thresholds = ps[last]
    # Include threshold zero explicitly, which is the tie-preferred all-positive rule.
    thresholds = np.r_[thresholds, 0.0]
    f2 = np.r_[f2, f2[-1]]
    recall = np.r_[recall, recall[-1]]
    best = max(range(len(thresholds)), key=lambda i: (f2[i], recall[i], -thresholds[i]))
    return {"threshold": float(thresholds[best]), "f2": float(f2[best]),
            "recall": float(recall[best]), "rule": "max weighted F2; ties recall then smaller threshold"}


def classification_metrics(y, p, weights, threshold=0.5) -> dict:
    y, p, weights = _arrays(y, p, weights)
    if not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Classification metrics require binary labels and probabilities")
    predicted = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "weight": float(weights.sum()), "threshold": float(threshold),
        "risk_prevalence": float(np.average(y, weights=weights)),
        "average_precision": float(average_precision_score(y, p, sample_weight=weights)) if (y == 1).any() else None,
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weights)) if both else None,
        "precision": float(precision_score(y, predicted, sample_weight=weights, zero_division=0)),
        "recall": float(recall_score(y, predicted, sample_weight=weights, zero_division=0)),
        "f1": float(fbeta_score(y, predicted, beta=1, sample_weight=weights, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, sample_weight=weights, zero_division=0)),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "accuracy": float(accuracy_score(y, predicted, sample_weight=weights)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted, sample_weight=weights)) if both else None,
        "confusion_count": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "confusion_weighted": confusion_matrix(y, predicted, sample_weight=weights, labels=[0, 1]).tolist(),
        "calibration_curve": weighted_calibration_curve(y, p, weights),
        "class_order": ["alfabetizado", "nao_alfabetizado"],
    }


def regression_metrics(y, predictions) -> dict:
    raw = np.asarray(predictions, dtype=float)
    y, _, _ = _arrays(y, raw)
    if not (np.isfinite(y).all() and ((y >= 0) & (y <= 100)).all()):
        raise ValueError("Municipal labels must be observed percentage points in [0,100]")
    p = np.clip(raw, 0, 100)
    return {"n": len(y), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(root_mean_squared_error(y, p)), "r2": float(r2_score(y, p)) if len(y) > 1 else None,
            "clipped_count": int(np.sum(raw != p))}


def residual_band(y_reserve, predictions_reserve) -> tuple[float, float]:
    residuals = np.asarray(y_reserve) - np.clip(np.asarray(predictions_reserve), 0, 100)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Band residuals must be finite and nonempty")
    return tuple(float(x) for x in np.quantile(residuals, [0.05, 0.95]))


def band_metrics(y, predictions, quantiles) -> dict:
    p = np.clip(np.asarray(predictions), 0, 100)
    lower, upper = np.clip(p + quantiles[0], 0, 100), np.clip(p + quantiles[1], 0, 100)
    return {"band_coverage": float(np.mean((y >= lower) & (y <= upper))),
            "band_mean_width": float(np.mean(upper - lower))}


def goal_attained(value, limit, operator):
    if value is None or limit is None or not np.isfinite(float(value)) or not np.isfinite(float(limit)):
        return None
    if operator == ">":
        return bool(value > limit)
    if operator == ">=":
        return bool(value >= limit)
    raise ValueError("Unknown goal comparison operator")
