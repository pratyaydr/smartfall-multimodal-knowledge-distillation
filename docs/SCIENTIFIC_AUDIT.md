# Scientific audit of the supplied notebook

Source: `notebooks/original_kaggle.ipynb`; cells are zero-indexed. This audit is
based on all 39 cells, their saved stream/error outputs, code paths, and the
current upstream dataset README checked on 2026-11-12. There are no saved plot
outputs, prediction arrays, raw data, or model checkpoint attachments.

## Actual objective and scope

The project classifies windows as ADL or fall. It explores whether multimodal
skeleton/watch training can improve an inertial model through knowledge
transfer. It implements two model families and six saved test summaries. This
repository presents an exploratory portfolio project, not a novelty claim,
published-paper reproduction, controlled ablation, or validated alerting system.

## Findings and actions

| Finding | Evidence and implication | Treatment in this repository |
| --- | --- | --- |
| Known labels influence window construction | Cell 3 chooses peak height 1.4/distance 50 for falls and 1.2/distance 100 for ADL. Even test window selection uses the target. | Preserved, prominently disclosed; a label-blind rerun is future work. |
| Whole-group normalization | Cell 3 merges each subject/activity's trials and fits mean/std to that entire group, including held-out recordings' own data and inserted zeros. Group identity also uses known activity metadata. | Preserved. This is an offline/transductive preprocessing protocol, not train-fitted streaming normalization. |
| Student inputs depend on skeleton preprocessing | All windows are aligned with skeleton data by DTW before any model is trained. Students ignore skeleton tensors in `forward`, but their temporal selection still uses skeleton. | Accelerometer-only model inference is supported for preprocessed windows; a raw unimodal pipeline is not established. |
| Future context is used | `filtfilt`, mean/std, and DTW operate on whole merged recordings. | No causal, online, early-warning, or measured latency claim. |
| CSV format discrepancy | The notebook uses `header=0` for all CSVs, then skips two additional accelerometer rows. Current upstream documentation describes headerless CSVs. | Preserved for historical traceability. On headerless files this drops the first skeleton row and first three accelerometer rows; synthetic regression test demonstrates it. Exact Kaggle bytes were not available. |
| Sampling-rate discrepancy | Filter defaults to 25 Hz. Current upstream README describes 32 Hz watch data and 30 FPS skeleton data. No explicit resampling is performed. | Preserve 25 Hz assumption; do not assign a verified number of seconds to aligned 64-frame windows. |
| DTW signal comparability | Alignment compares acceleration norm with hip-centered joint-9 position norm, rather than validated time stamps or matching physical quantities. A greedy unique-index path then discards repeated indices. | Preserve original operation; physical/time alignment quality is unverified. |
| Trial-boundary distortion | Filtering and statistics include 20 zero rows between trials. Only an accelerometer-indexed gap mask survives DTW. Original trial IDs and source indices are not retained in saved windows. | Gap rejection is tested, but complete preservation of all original trial boundaries cannot be proven from saved outputs. |
| Subject split is disjoint | Cell 5 assigns 26/2/8 IDs; saved output reports no ID overlap. | Enforced by explicit validation, including errors for overlapping configured subject sets. Raw-level identity consistency still needs dataset inspection. |
| Excluded windows | 13,077 generated minus 6,874 + 942 + 2,572 selected = 2,689 excluded across 17 IDs. | Exclusions are explicit in tables and new split summaries. |
| Dataset-version uncertainty | Saved output lists 53 observed IDs; upstream describes 51 multimodal-ready subjects, so the source mirror is not assumed identical to today's upstream release. | Require exact version/hash for real reproduction. Do not silently substitute updated files. |
| Young-only held-out fall cohort | Test IDs 56–63 are in the notebook's young group. Older ADL subjects appear in training, not as an elderly-fall test cohort. | No claim about elderly fall sensitivity. |
| Imbalanced classification | Test counts are 1,891 ADL and 681 falls. Model selection is mostly validation accuracy. | Show fall precision/recall/F1 with accuracy; a naive always-ADL baseline is about 73.52% on the recorded test counts. |
| Nonindependent windows | Sliding and peak-centered windows can overlap within recordings. | No window-independent confidence intervals or significance claims. |
| Limited model-selection evidence | Only two validation subjects; thresholds are tuned each epoch and models are tried sequentially with repeated exposure to test summaries. | Treat test comparisons as exploratory. No proof of absence of human test-set adaptation. |
| Unequal comparisons | IT KD warm-starts; LightHART KD starts from random weights. Different losses, schedules and selection criteria apply. | Preserve all settings and show both positive and negative KD comparisons. |
| Inconsistent selection criteria | Threshold grid always maximizes validation accuracy. All checkpoints maximize accuracy except cell 24, which maximizes F1 at that accuracy-tuned threshold. | Preserve both layers of selection; config records each. |
| Focal alpha is not class-dependent | Source focal implementation multiplies every sample by the same alpha=0.80. | Preserve formula; do not describe it as class alpha balancing. |
| IT logit KD differs from Bernoulli KL | Cell 12 builds two logits `[-z,z]`, so the implied positive probability is sigmoid(2z/T), rather than sigmoid(z/T). | Preserve symmetric-logit formulation; label it precisely. |
| LightHART KD receives two perturbations | Cell 24 augments shared input externally; LightHART augments again internally in training. The teacher sees only the external perturbation. | Preserve both augmentations; do not silently remove one. |
| Sparse seed provenance | Cells 8/10/12 have no explicit seed setup; cells 21/22/24 set seed 42. Earlier loader/model execution consumes RNG state. | Refactor sets seed per run and records it. This improves repeatability but cannot recreate historical random trajectories. |
| Checkpoints omit thresholds | Source saves bare `state_dict`; the best threshold lives only in memory/printed output. | New checkpoints include config, threshold, selected epoch, and validation metrics. Archived weights require an explicit threshold. |
| Short-series filtering bug | Original guard permits lengths 13–15 although SciPy's default pad length for this filter is 15. | Guard now returns short inputs unchanged below 16 rows. Main source trial loader requires at least 64 rows; parity is checked on valid synthetic trials. |
| Incomplete export attempt | Cell 28 fails importing `tensorflow_addons`; cells 29 onward have no execution count/output. Cell 30 redefines a different 64-dimensional, three-layer LightHART, while cell 21 uses 128 dimensions and five layers. It also references a Windows-only checkpoint path. | Original notebook preserved; no successful TFLite/ONNX deployment or size/latency result is claimed. |

## Preserved mathematical and experimental details

- Filter: order 4 Butterworth low pass, cutoff 7.5 with assumed sampling rate 25.
- Magnitude: norm after subtracting the group's channel mean.
- Hip-centering: subtract joint 0 independently at each frame.
- DTW: FastDTW, Euclidean distance, greedily unique sensor/skeleton index pairs.
- Windows: 64 frames; sliding step 32; duplicate exact ranges suppressed;
  gap-masked ranges rejected.
- CSV cleaning: drop rows containing any missing value, then historical no-op
  backfill; read last three acceleration and last 96 skeleton columns.
- Batch size 32, shuffled training, `drop_last=True`; validation/test keep remainders.
- Binary sigmoid outputs, `>=` threshold; thresholds from
  `np.arange(0.30,0.81,0.02)`, with first strict improvement winning ties.
- AdamW, gradient clipping norm 1.0, OneCycle or warm-restart cosine schedules.
- BCE families use positive weight 2.0 and training-only label smoothing to
  0.05/0.95. Focal families use unsmoothed labels, alpha 0.8, gamma 1.5.
- Source mean evaluation loss is a mean of batch means, so the last smaller
  batch has equal weight in that loss average. This aggregation is preserved.

## Claims the evidence supports

The notebook records six complete test metric summaries and an export failure.
Local checks establish source-matched forward behavior, synthetic preprocessing
parity, valid subject partitions, and executable training/evaluation/inference
interfaces. The evidence does not establish reproduced real-data predictions,
statistical KD benefit, robust deployment, or new state-of-the-art performance.

## Next experiment design

Freeze an exact dataset version and a final subject test set. Inspect actual
headers/time stamps and retain trial provenance. Use label-blind, trial-safe
windowing with training-fitted or causal normalization, and a skeleton-free
student preprocessing route. Then match initialization and training settings
across no-KD/KD comparisons, evaluate multiple seeds/subject folds, and report
subject-level uncertainty and continuous-stream false alarms. All corrected
results should be stored separately from the six archived rows.

## External source

[SmartFallMM upstream README](https://github.com/txst-cs-smartfall/SmartFallMM-Dataset)
was checked on 2026-11-12 for format/access context. Its current description does
not identify the exact version used in the attached Kaggle execution.
