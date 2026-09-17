"""Classical baseline. Milestone 2 reproduces the published number with this."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .data import SEED, Split


def fit_predict(split: Split, seed: int = SEED) -> float:
    model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    model.fit(split.X_train, split.y_train)
    scores = model.predict_proba(split.X_test)[:, 1]
    return float(roc_auc_score(split.y_test, scores))
