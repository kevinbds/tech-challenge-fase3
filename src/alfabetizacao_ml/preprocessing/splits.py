"""Partitions are learned from municipality membership, never from future labels."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold


@dataclass(frozen=True)
class MunicipalitySplit:
    development: tuple[str, ...]
    reserve: tuple[str, ...]

    def mask(self, frame: pd.DataFrame, *, reserve: bool = False) -> np.ndarray:
        return frame["id_municipio"].astype(str).isin(self.reserve if reserve else self.development).to_numpy()

    @property
    def fingerprint(self) -> str:
        return sha256(("dev:" + ",".join(self.development) + "|reserve:" + ",".join(self.reserve)).encode()).hexdigest()


def municipality_split(frame: pd.DataFrame, fraction: float = 0.2, seed: int = 42) -> MunicipalitySplit:
    if frame["id_municipio"].isna().any():
        raise ValueError("Municipality grouping key cannot be missing")
    groups = np.array(sorted(frame["id_municipio"].astype(str).unique()))
    if len(groups) < 7:
        raise ValueError("At least seven municipalities are required for reserve and five folds")
    dev, reserve = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(groups, groups=groups))
    return MunicipalitySplit(tuple(sorted(groups[dev])), tuple(sorted(groups[reserve])))


def grouped_folds(frame: pd.DataFrame, y: np.ndarray, *, classification: bool,
                  n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["id_municipio"].astype(str).to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough distinct municipalities for the declared folds")
    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if classification else GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    folds = list(splitter.split(frame, y, groups))
    for train, valid in folds:
        if set(groups[train]) & set(groups[valid]):
            raise AssertionError("Municipality leakage across a validation fold")
        if classification and (len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2):
            raise ValueError("Each classification fold must contain both classes; do not inspect future labels to repair it")
    return folds
