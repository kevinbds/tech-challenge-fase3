"""Independent reference calculations for explanations and grouped uncertainty."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from test_model_protocol import cohorts
from threadpoolctl import threadpool_limits

from alfabetizacao_ml.evaluation.interpretation import compressed_context_examples, shap_summary
from alfabetizacao_ml.evaluation.uncertainty import paired_classification_bootstrap
from alfabetizacao_ml.modeling.estimators import fit_weighted, make_classifier, risk_probability
from alfabetizacao_ml.preprocessing.features import select_features, validate_student_labels


@pytest.mark.parametrize("name,params", [
    ("logistic", {}), ("random_forest", {"n_estimators": 5}),
    ("hist_gradient_boosting", {"max_iter": 12}),
])
def test_shap_additivity_and_context_compression_for_each_family(name, params):
    students, _ = cohorts()
    students = students.loc[students.ano.eq(2024)].reset_index(drop=True)
    X = select_features(students)
    y, w = validate_student_labels(students)
    with threadpool_limits(limits=2):
        model = fit_weighted(make_classifier(name, params, threads=2), X, y, w)
        summary = shap_summary(model, X, maximum=20)
        compressed, yc, wc, _, _ = compressed_context_examples(students, X, y, w)
        np.testing.assert_allclose(
            average_precision_score(y, risk_probability(model, X), sample_weight=w),
            average_precision_score(yc, risk_probability(model, compressed), sample_weight=wc), atol=1e-12)
    assert summary["additivity_max_error"] < 1e-4
    assert "before sigmoid calibration" in summary["output_scale"]
    assert "risco=1" in summary["class_label"]


def test_bootstrap_matches_direct_municipality_resampling():
    rng = np.random.default_rng(82)
    y = rng.integers(0, 2, 160)
    p, b, w = rng.choice([.1, .3, .8], 160), rng.choice([.2, .5], 160), rng.uniform(.1, 7, 160)
    groups = np.repeat(np.arange(8), 20)
    actual = paired_classification_bootstrap(y, p, b, w, groups, iterations=50, seed=6)
    rng = np.random.default_rng(6)
    ap, delta, brier = [], [], []
    for _ in range(50):
        indices = np.concatenate([np.flatnonzero(groups == group) for group in rng.integers(0, 8, 8)])
        yy, pp, bb, ww = y[indices], p[indices], b[indices], w[indices]
        score = average_precision_score(yy, pp, sample_weight=ww)
        ap.append(score)
        delta.append(score-average_precision_score(yy, bb, sample_weight=ww))
        brier.append(np.average((yy-pp)**2-(yy-bb)**2, weights=ww))
    for key, values in [("average_precision_ci95", ap), ("ap_delta_vs_contextual_ci95", delta),
                        ("brier_delta_vs_contextual_ci95", brier)]:
        np.testing.assert_allclose(actual[key], np.quantile(values, [.025, .975]), atol=1e-12)
