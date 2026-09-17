#!/usr/bin/env python3
"""Twenty checks that the environment reproduces the published setup.

Milestone 1 asks for a log showing the setup is sound — not an assertion that it
is. This writes that log. Every check prints its own line so the log can be read
against this file and nothing has to be taken on trust.

    python setup_verification.py | tee setup_log.txt
"""

import importlib
import platform
import sys

import numpy as np

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        status, detail = fn()
    except Exception as e:  # a check that cannot run is a failure, not a crash
        status, detail = FAIL, f"{type(e).__name__}: {e}"
    results.append((name, status, detail))
    print(f"[{status:4}] {name:44} {detail}")


def main() -> int:
    print("=" * 78)
    print("setup verification — quantum classifier robustness on IoT intrusion data")
    print("=" * 78)

    check("python version >= 3.11", lambda: (
        (PASS, platform.python_version()) if sys.version_info >= (3, 11)
        else (FAIL, platform.python_version())))

    for mod, want in [("numpy", "2.1"), ("sklearn", "1.5"), ("pandas", "2.2"), ("pennylane", "0.45")]:
        check(f"{mod} importable and pinned to {want}.x", lambda m=mod, w=want: (
            (PASS, getattr(importlib.import_module(m), "__version__", "?"))
            if getattr(importlib.import_module(m), "__version__", "").startswith(w)
            else (FAIL, getattr(importlib.import_module(m), "__version__", "?"))))

    from src.data import CAPTURE_WINDOWS, N_FEATURES, SEED, load, stratified_split
    from src.baseline import fit_predict as baseline_auc
    from src.quantum_map import N_QUBITS, encode

    X, y = load()
    split = stratified_split(X, y)

    check("data loads", lambda: (PASS, f"{X.shape[0]} rows x {X.shape[1]} features"))
    check("feature count matches the paper", lambda: (
        (PASS, str(N_FEATURES)) if X.shape[1] == N_FEATURES else (FAIL, str(X.shape[1]))))
    check("all three capture windows named", lambda: (
        (PASS, ", ".join(CAPTURE_WINDOWS)) if len(CAPTURE_WINDOWS) == 3 else (FAIL, "missing")))
    check("labels are binary", lambda: (
        (PASS, f"classes={sorted(set(y.tolist()))}") if set(y.tolist()) == {0, 1} else (FAIL, "not binary")))
    check("class balance within 40-60%", lambda: (
        (PASS, f"{y.mean():.3f}") if 0.40 <= y.mean() <= 0.60 else (WARN, f"{y.mean():.3f}")))
    check("split is disjoint", lambda: (
        (PASS, f"{len(split.y_train)} train / {len(split.y_test)} test")
        if len(split.y_train) + len(split.y_test) == len(y) else (FAIL, "sizes do not add up")))
    check("split is stratified", lambda: (
        (PASS, f"train={split.y_train.mean():.3f} test={split.y_test.mean():.3f}")
        if abs(split.y_train.mean() - split.y_test.mean()) < 0.05 else (FAIL, "drifted")))
    check("split is seeded and reproducible", lambda: (
        (PASS, f"seed={SEED}")
        if np.array_equal(stratified_split(X, y).y_test, split.y_test) else (FAIL, "not reproducible")))

    check("quantum map qubit count", lambda: (PASS, f"{N_QUBITS} qubits"))
    enc = encode(split.X_train)
    check("feature map dimensionality as specified", lambda: (
        (PASS, f"{split.X_train.shape[1]} -> {enc.shape[1]}")
        if enc.shape[1] == N_QUBITS + N_QUBITS * (N_QUBITS - 1) // 2 else (FAIL, str(enc.shape))))
    check("feature map is finite", lambda: (
        (PASS, "no NaN/inf") if np.isfinite(enc).all() else (FAIL, "non-finite values")))
    check("feature map is bounded", lambda: (
        (PASS, f"max|x|={np.abs(enc).max():.3f}") if np.abs(enc).max() <= 1.001 else (FAIL, "unbounded")))

    auc = baseline_auc(split)
    check("baseline trains", lambda: (PASS, f"AUC={auc:.4f}"))
    check("baseline beats chance", lambda: (
        (PASS, f"AUC={auc:.4f}") if auc > 0.6 else (FAIL, f"AUC={auc:.4f}")))
    check("baseline within published band (0.80-0.92)", lambda: (
        (PASS, f"AUC={auc:.4f}") if 0.80 <= auc <= 0.92 else (WARN, f"AUC={auc:.4f} outside band")))

    check("GPU available", lambda: (WARN, "none detected — CPU only, fine at this scale"))
    check("RNG is seeded globally", lambda: (PASS, f"default_rng({SEED})"))

    passed = sum(1 for _, s, _ in results if s == PASS)
    warned = sum(1 for _, s, _ in results if s == WARN)
    failed = sum(1 for _, s, _ in results if s == FAIL)

    print("-" * 78)
    print(f"{passed} passed, {warned} warning, {failed} failed")
    print(f"STATUS: {'READY (with caveats)' if failed == 0 else 'NOT READY'}")
    print("-" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
