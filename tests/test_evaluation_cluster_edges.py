"""Silhouette must not silently omit a small observed municipal cluster."""
import numpy as np
import pandas as pd

from alfabetizacao_ml.evaluation.interpretation import cluster_municipalities


def test_silhouette_includes_rare_cluster_in_full_municipal_population():
    n = 5500
    frame = pd.DataFrame({"ano": 2025, "id_municipio": np.arange(n),
        "taxa_alfabetizacao": 60., "media_inse_2021": 4., "log_populacao_2022": 10.,
        "densidade_2022": np.arange(n) % 3 + 10.})
    frame.loc[0, "densidade_2022"] = 1e9
    result = cluster_municipalities(frame, smoke=True)
    assert result["status"] == "completed"
    assert result["selected_k"] == 2
    assert np.isfinite(result["silhouette"])
    assert len(result["assignments"]) == n
