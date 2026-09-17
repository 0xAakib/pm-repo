# Reading notes — milestone 1

My own summary of the two seed papers, and what each one actually measured.

## QuIDS: A quantum support vector machine-based intrusion detection system for IoT networks (2025)

**What it measured.** Binary intrusion detection on IoT network capture, QSVM
with a ZZ feature map against a classical SVM baseline. Reported accuracy and
F1 on a held-out split of the same capture.

**Dataset.** IoT-23-style flow records, 41 tabular features per flow. The paper
uses a subsample rather than the full capture, which matters for us: their
headline number is on roughly 4k flows, and the variance at that size is wide
enough that a single-seed comparison would not be meaningful.

**What I take from it.** It is the baseline this project is built against, and
its reported margin over the classical SVM is small — inside the range I would
expect from seed variance alone. That is the thing milestone 3 has to test
rather than assume.

## Towards Network Intrusion Detection via Quantum Machine Learning: A Reality Check (2025)

**What it measured.** A survey-plus-replication: takes several published QML
intrusion-detection results and re-runs them on matched splits with controlled
seeds. Reports how much of each claimed improvement survives.

**Dataset.** Several, including the same IoT-23 family as QuIDS, which is what
makes the two comparable at all.

**What I take from it.** Most of the improvement did not survive matched splits.
It is the reason our milestone 3 is written as a premise test with a real
possibility of "no" — the honest answer here may be that the feature map buys
nothing, and that is a publishable result rather than a failed project.

## How the two relate

QuIDS is the claim; the Reality Check paper is the reason to doubt it. Our
contribution is neither — it is the robustness question underneath both, which
neither paper measures: how either model degrades under perturbation and
distribution shift between capture windows.
