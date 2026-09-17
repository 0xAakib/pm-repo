# Quantum classifier robustness on IoT intrusion data

Testbed for milestone 1 ("Read & Digest") of the local Project Mentor track.

Two seed papers, one classical baseline, one quantum feature map, and a setup
verification script whose log is the artifact filed for verification.

## Layout

| Path | What |
|---|---|
| `setup_verification.py` | 20 environment + data checks. Writes `setup_log.txt`. |
| `setup_log.txt` | The captured output. This is the artifact filed with the result. |
| `src/data.py` | Loads and splits the IoT capture data. |
| `src/baseline.py` | Classical baseline — gradient boosting over the tabular features. |
| `src/quantum_map.py` | The quantum feature map under test. |

## Reproducing

```bash
pip install -r requirements.txt
python setup_verification.py | tee setup_log.txt
```

Expected: 19 passed, 1 warning (no GPU), 0 failed.
