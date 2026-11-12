# Validation report

**Delivery date:** 2026-11-12. **Scope:** software verification and archived evidence.
No original SmartFallMM raw data, processed arrays, or pretrained checkpoints were
attached. Recorded real-data scores remain unverified by inference/retraining.

## Completed checks

| Check | Observed result |
| --- | --- |
| Original notebook identity | Preserved byte-for-byte; SHA-256 in `evidence/source_manifest.json` |
| Source architecture syntax | All 15 extracted class AST hashes match their original definitions |
| Cross-version evidence check | Stored AST digests are checked on Python 3.12; other Python versions compare original-notebook and refactored-module ASTs directly to avoid interpreter-specific AST hash changes |
| Metric extraction | Six final summaries match their source cells at printed precision |
| Validation history extraction | 147 logged epochs retained across six experiments |
| Subject lists | Configured train/validation/test subject sets are disjoint |
| Excluded-window accounting | 13,077 − 6,874 − 942 − 2,572 = 2,689 windows across 17 unassigned IDs |
| Automated test suite | 35 tests passed locally |
| Forward parity | All four refactored models produce exactly equal checked outputs with original definitions, identical random weights, and synthetic inputs |
| Preprocessing parity | Original and refactored outputs agree exactly on the tested synthetic multi-trial input; no real-data arrays were available |
| KD mathematics | Both feature/logit objectives give finite gradients; identical-input loss checks pass |
| CSV preprocessing smoke | Synthetic paired CSV discovery, filtering, DTW, window extraction, and NPZ save passed |
| Training smoke | All six configurations completed one epoch on tiny synthetic CPU arrays |
| Checkpoint evaluation | Re-evaluating every synthetic best checkpoint gives exactly equal saved metrics |
| Student inference | Three already preprocessed synthetic windows predicted successfully, without skeleton input |
| Evidence walkthrough | All eight code cells executed successfully through an IPython in-process fallback, with saved tables and three image outputs |
| Notebook schemas | Original and companion validated with nbformat |
| Figures | Three PNGs rendered from archived values and visually inspected; the small validation-bar label was resized after inspection |
| Code checks | Ruff lint/format checks pass for source, tests, and scripts |
| Packaging checks | Local Markdown targets, source hash, JSON/TOML parsing, dependency imports, file inventory, and secret-pattern checks reviewed |

Synthetic results are structural checks only. Their scores are not represented
as SmartFallMM performance. Status is recorded in
`evidence/synthetic_validation.json`; the smoke script is included for rerunning.

## Environment actually used

Linux CPU, Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5, pandas 2.2.3,
SciPy 1.17.0, scikit-learn 1.8.0, FastDTW 0.3.4, and Matplotlib 3.10.8.
The exact direct package versions are in `requirements-tested.txt`.
They are not the original Kaggle environment, and the entire dependency graph
is not frozen. A host-provided pytest plugin emitted a deprecation warning;
none of the 35 project checks failed in the final scoped test run.

During the subsequent archive refresh, the integrity/evidence checker also passed
under Python 3.13.5: all 15 original-notebook/refactored-model AST parity checks,
six metric records, notebook schemas, subject partitions, and 70 tracked-file
SHA-256 entries validated. The three test modules not requiring FastDTW passed
(27 tests). The preprocessing test module was not rerun in that host because
`fastdtw` was not installed there; the earlier 35-test result above is from the
Python 3.12.14 validation environment with the declared dependencies installed.

## Notebook execution and rendering limits

The normal `jupyter nbconvert --execute` attempt failed during socket-kernel
startup: the host reported `Operation not permitted`, then the kernel died before
replying to `kernel_info`. It did not reach a notebook computation error.
`scripts/execute_walkthrough_inprocess.py` then executed the eight code cells in
order and captured real IPython stream/rich outputs. The mode and reason are
embedded in notebook metadata. It is not a claim that normal kernel execution
passed. The fallback can be rerun from the repository root:

```bash
python scripts/execute_walkthrough_inprocess.py
```

HTML export succeeded and contains three tables and three embedded chart images.
A browser-based whole-notebook screenshot inspection was attempted, but Chromium
startup was also blocked by the host's socket restriction. Figures were inspected
as PNGs and output tables/HTML were structurally checked; full notebook CSS layout
and GitHub-hosted rendering remain unverified. Nbconvert also emitted its generic
image-alt warning, although the PNG outputs carry descriptive alt metadata.

## Unverified work

- Raw-level dataset identity, headers, sampling rates, trial integrity, and all
  23 unreadable CSVs recorded by the source preprocessing run.
- Regenerating original real-data arrays or loading original trained checkpoints.
- Any of the six historical test metric rows by new real-data inference.
- Full retraining, exact random trajectories, repeated seeds, or uncertainty.
- Label-blind, causal, skeleton-free student preprocessing and continuous alerts.
- Generalization to elderly falls, other cohorts, or real-world uncontrolled falls.
- Successful ONNX/TensorFlow/TFLite conversion, model size, latency, or edge deployment.
- GPU execution, Windows/macOS installation, and Python versions other than 3.12.
- Remote GitHub publication, hosted CI execution, and GitHub Markdown rendering.

The primary commands and next verification steps are in `docs/REPRODUCING.md`.
