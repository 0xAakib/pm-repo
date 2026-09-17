"""The quantum feature map under test.

Milestone 3 is the premise test: whether this moves the metric outside
run-to-run variance at all. It is allowed to answer no.
"""

import numpy as np

from .data import Split

N_QUBITS = 6


def encode(X: np.ndarray, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Angle encoding onto `n_qubits`, then the second-order interaction terms
    the ZZ feature map would produce. Simulated classically — there is no
    hardware in this loop, and the write-up must not imply otherwise."""
    A = np.tanh(X[:, :n_qubits])
    pairs = [A[:, i] * A[:, j] for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return np.column_stack([A, np.stack(pairs, axis=1)])


def fit_predict(split: Split, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(encode(split.X_train), split.y_train)
    scores = model.predict_proba(encode(split.X_test))[:, 1]
    return float(roc_auc_score(split.y_test, scores))
