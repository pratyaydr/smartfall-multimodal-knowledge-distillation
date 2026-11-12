# Delivery change report

## Changed

- Added concise project attribution and software-citation metadata; the original Kaggle notebook URL is retained for source provenance.
- Made the evidence AST-parity check portable across Python minor versions while
  retaining the recorded Python 3.12 digest verification.

- Split notebook-bound implementation into four preserved model architectures,
  shared data/loss/evaluation utilities, and an explicit CLI.
- Added six JSON experiment configurations and a source-matched subject file.
- Replaced Kaggle filesystem assumptions at the package entry point with explicit
  paths. Added schema validation, informative empty-data errors, and overlap checks.
- Added seeds per new run, weights-only loading, strict state checks, non-finite
  loss detection, and no-overwrite output behavior.
- New checkpoints retain validation-selected thresholds, config, epoch, and
  metrics. New runs also save source hashes, histories, test probabilities, and
  confusion matrices.
- Fixed the short-filter padding guard for 13–15 rows. This guard is below the
  main source trial loader's minimum length, and parity tests cover normal input.
- Removed package dependencies on notebook installation commands and the failed
  TensorFlow/ONNX conversion stack.
- Added evidence extraction, plots, an executed walkthrough, tests, CPU CI, and
  documentation for methodology, data access, validation, and publication.

## Preserved

- Original notebook bytes, outputs, failure, and cell order.
- All six saved metric summaries, positive/negative KD comparisons, and the
  distinction between the first and later experiment families.
- Architecture/state keys, model noise behavior, loss mathematics, optimization
  settings, initialization policy, threshold search, and checkpoint criteria.
- Historical CSV/header logic, whole-group processing, filtering assumptions,
  skeleton-based DTW, label-conditioned windows, and static subject IDs.

## Verification and limits

See `docs/VALIDATION.md` and `evidence/synthetic_validation.json` for executed
checks. These validate software and recorded evidence; they do not validate
real-data score reproduction. No remote GitHub publication or license grant was
performed. No original data/checkpoints were attached, and no full retraining or
conversion experiment was run.
