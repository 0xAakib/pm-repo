"""Load and split the IoT intrusion capture data.

Splits are seeded and stratified so the baseline and the quantum feature map
see identical folds — comparing the two across different splits is the failure
milestone 3 exists to rule out.
"""

from dataclasses import dataclass

import numpy as np

CAPTURE_WINDOWS = ("w1-morning", "w2-evening", "w3-overnight")
N_FEATURES = 41
SEED = 20260917


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    seed: int


def load(n: int = 4096, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic stand-in shaped like the real capture, so the pipeline runs
    without shipping the dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    signal = X[:, :6] @ rng.normal(size=6)
    y = (signal + rng.normal(scale=2.0, size=n) > 0).astype(int)
    return X, y


def stratified_split(X, y, test_frac: float = 0.25, seed: int = SEED) -> Split:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    test = np.concatenate([
        rng.permutation(idx[y == c])[: int(test_frac * (y == c).sum())] for c in (0, 1)
    ])
    mask = np.zeros(len(y), dtype=bool)
    mask[test] = True
    return Split(X[~mask], y[~mask], X[mask], y[mask], seed)
