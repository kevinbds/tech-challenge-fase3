"""Paired municipality bootstrap, conditional on the observed population."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def paired_classification_bootstrap(y, probability, baseline, weights, groups,
                                    *, iterations=500, seed=42) -> dict:
    # Combining identical score/label/group tuples is exact for weighted AP and
    # Brier, and avoids replicating millions of students on every bootstrap draw.
    table = pd.DataFrame({"y": y, "p": probability, "b": baseline, "w": weights, "g": groups})
    table = table.groupby(["g", "y", "p", "b"], sort=True, dropna=False, as_index=False)["w"].sum()
    codes, unique = pd.factorize(table["g"], sort=True)
    labels, p, b, w = (table[name].to_numpy() for name in ["y", "p", "b", "w"])
    rng = np.random.default_rng(seed)
    ap, differences, brier_differences = [], [], []
    for _ in range(iterations):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weighted = w * multiplicity[codes]
        active = weighted > 0
        if len(np.unique(labels[active])) != 2:
            continue
        score = average_precision_score(labels[active], p[active], sample_weight=weighted[active])
        reference = average_precision_score(labels[active], b[active], sample_weight=weighted[active])
        ap.append(float(score))
        differences.append(float(score - reference))
        brier_differences.append(float(np.average((labels-p)**2-(labels-b)**2, weights=weighted)))
    if not ap:
        raise ValueError("No valid two-class municipality bootstrap replicates")
    return {"method": "paired municipality percentile bootstrap", "requested_iterations": iterations,
            "valid_iterations": len(ap), "municipalities": len(unique),
            "average_precision_ci95": np.quantile(ap, [0.025, 0.975]).tolist(),
            "ap_delta_vs_contextual_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "brier_delta_vs_contextual_ci95": np.quantile(brier_differences, [0.025, 0.975]).tolist(),
            "scope": "conditional empirical variability; not the complete official survey design or future national shocks"}


def paired_regression_bootstrap(y, probability, baseline, *, iterations=500, seed=42) -> dict:
    y, p, b = np.asarray(y), np.asarray(probability), np.asarray(baseline)
    delta = np.abs(y - p) - np.abs(y - b)
    rng = np.random.default_rng(seed)
    values = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(iterations)]
    return {"method": "paired municipality percentile bootstrap", "iterations": iterations,
            "mae_delta_vs_persistence": float(delta.mean()),
            "mae_delta_vs_persistence_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "scope": "conditional empirical variability; each municipality has equal weight"}
