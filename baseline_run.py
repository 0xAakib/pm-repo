#!/usr/bin/env python3
"""Milestone 2 — reproduce the classical baseline.

Reports the exact metric, the seed and the split, across five seeds, because a
single-seed number cannot be placed against a published one. Writes
baseline_log.txt, which is the artifact filed with the result.

    python baseline_run.py | tee baseline_log.txt
"""

import statistics

from src.baseline import fit_predict
from src.data import SEED, load, stratified_split

PUBLISHED = 0.8630  # headline AUC for this synthetic testbed (there is no real paper)
TOLERANCE = 0.01

def main() -> int:
    print("=" * 74)
    print("milestone 2 — classical baseline reproduction")
    print("=" * 74)
    print(f"published headline : AUC {PUBLISHED:.4f}")
    print(f"tolerance          : +/- {TOLERANCE}")
    print(f"split              : stratified, 25% test, seeded")
    print("-" * 74)

    aucs = []
    for i in range(5):
        seed = SEED + i
        X, y = load(seed=seed)
        split = stratified_split(X, y, seed=seed)
        auc = fit_predict(split, seed=seed)
        aucs.append(auc)
        print(f"seed {seed}   train={len(split.y_train):5d}  test={len(split.y_test):5d}   AUC={auc:.4f}")

    mean = statistics.mean(aucs)
    sd = statistics.stdev(aucs)
    delta = mean - PUBLISHED
    within = abs(delta) <= TOLERANCE

    print("-" * 74)
    print(f"mean AUC over 5 seeds : {mean:.4f}  (sd {sd:.4f})")
    print(f"delta vs published    : {delta:+.4f}")
    print(f"within +/- {TOLERANCE}        : {'YES' if within else 'NO'}")
    print(f"STATUS: {'REPRODUCED' if within else 'NOT REPRODUCED'}")
    print("-" * 74)
    return 0 if within else 1

if __name__ == "__main__":
    raise SystemExit(main())
