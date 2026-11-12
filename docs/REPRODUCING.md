# Reproduction and execution guide

## Scope of reproducibility

There are three distinct checks:

1. **Evidence reproduction:** re-extract the six four-decimal metric summaries
   from the preserved notebook and verify its SHA-256.
2. **Software verification:** synthetic preprocessing, source-model parity,
   loss gradients, and one-epoch runs for all six training configurations.
3. **Experiment reproduction:** regenerate the original real-data arrays, load
   original checkpoints, and compare predictions, or fully retrain.

Only the first two were completed. Neither original raw data nor pretrained
checkpoints were attached. No raw data is downloaded automatically.

## Environment

The exact direct packages tested locally are in `requirements-tested.txt`.
The local environment was Python 3.12.14 on Linux CPU with PyTorch 2.15.0+cpu.
This is not a captured lockfile from the original Kaggle execution. The original
metadata records Python 3.12.12 but not a complete dependency lock. Broad supported
ranges live in `pyproject.toml`; only the tested local versions are verified here.

Use a new virtual environment. For CPU installation use the commands in the
README. For CUDA, choose a compatible PyTorch build from the
[official installer](https://pytorch.org/get-started/locally/), then install this
package. CUDA performance, Windows installation, and other Python versions were
not checked. FastDTW may need a C++ compiler. Core code has no TensorFlow,
ONNX, `onnx_tf`, or `tensorflow_addons` dependency.

## Data root and historical transformations

The chosen directory must contain these relative paths:

| Relative path | Input |
| --- | --- |
| `young/skeleton/` | Younger cohort skeleton CSVs |
| `young/accelerometer/watch/` | Matching watch CSVs |
| `old/skeleton/` | Older cohort skeleton CSVs |
| `old/accelerometer/watch/` | Matching watch CSVs |

Extract the Kaggle mirror locally if you have access; its source slug is recorded
in `docs/DATASET.md`. The upstream repository is an alternative source for
inspection, not an assumed bit-for-bit replacement for the mirror. Record the
actual dataset version and SHA-256 inventory before comparing counts. Neither
current data availability nor all original corrupt/empty files was verified here.

```bash
smartfall preprocess --data-root data/raw/SmartFallMM-Dataset-main --output data/processed/smartfall.npz
```

The NPZ contains `accelerometer`, `skeleton`, `labels`, `subjects`, `activities`.
Unreadable CSVs are reported and skipped as in the source. Other format failures
raise errors. An empty dataset fails clearly. Dataset output must end in `.npz`
and existing output files are not overwritten. CSV headers/row dropping and
whole-group processing intentionally follow the notebook; review the audit first.

## Train all six experiments

Run from the repository root. Every `--output` directory must be new; rename an
old run if you want to retain it and train again. Configs use the original
maximum epochs, patience, and model sizes.

```bash
smartfall train --config configs/it_student.json --data data/processed/smartfall.npz --output runs/it_student
smartfall train --config configs/teacher.json --data data/processed/smartfall.npz --output runs/teacher
smartfall train --config configs/it_kd.json --data data/processed/smartfall.npz --teacher-checkpoint runs/teacher/best.pt --student-checkpoint runs/it_student/best.pt --output runs/it_kd
smartfall train --config configs/lighthart.json --data data/processed/smartfall.npz --output runs/lighthart
smartfall train --config configs/shared_teacher.json --data data/processed/smartfall.npz --output runs/shared_teacher
smartfall train --config configs/lighthart_kd.json --data data/processed/smartfall.npz --teacher-checkpoint runs/shared_teacher/best.pt --output runs/lighthart_kd
```

`--subjects configs/subjects.json` is the default. Do not swap subjects between
experiments when interpreting comparisons. `--device auto` chooses CUDA if
available; `--device cpu` forces CPU. `--threads 1` is the conservative default
for the small CPU checks; raise it for a suitable training host. `--epochs 1`
is an optional smoke budget, not a benchmark-reproduction setting.

| Config | Maximum epochs | Patience | Loss | Scheduler | Checkpoint criterion |
| --- | ---: | ---: | --- | --- | --- |
| IT student | 60 | 12 | weighted BCE + label smoothing | OneCycle | validation accuracy |
| Multimodal teacher | 50 | 12 | weighted BCE + label smoothing | OneCycle | validation accuracy |
| IT KD | 25 | 10 | 0.85 hard + 0.05 feature + 0.10 symmetric KL | cosine restarts | validation accuracy |
| LightHART | 60 | 15 | focal | cosine restarts | validation accuracy |
| Shared-CNN teacher | 50 | 12 | focal | cosine restarts | validation accuracy |
| LightHART KD | 35 | 12 | 0.60 hard + 0.30 feature + 0.10 sigmoid MSE | cosine restarts | validation fall F1 |

For every config, threshold search maximizes validation accuracy. IT KD enables
feature loss at zero-based epoch 5 (the sixth epoch), temperature 2.0. LightHART
KD uses feature loss from its first epoch and temperature 3.0. The frozen teacher
remains in evaluation mode. LightHART KD intentionally does not load the
LightHART baseline weights. Its external plus internal augmentation is retained.

New seeds, changed execution order, dependency versions and device arithmetic
can alter training outcomes even when model code and config settings match.
The saved four-decimal rows are historical references, not promised outputs.

## Outputs and evaluation

Each new run writes:

| File | Content |
| --- | --- |
| `config.json` | Effective configuration, including an epoch override |
| `provenance.json` | Python/Torch/NumPy versions, device, source NPZ/split/init checkpoint hashes |
| `split_summary.json` | Selected/excluded windows, subjects, missing configured subjects |
| `history.json` | Epoch train loss and validation metrics |
| `best.pt` | Weights, config, selected threshold/epoch/validation metrics |
| `test_metrics.json` | One evaluation after validation selection |
| `test_predictions.npz` | Labels and probabilities for the selected test partition |

```bash
smartfall evaluate --checkpoint runs/lighthart_kd/best.pt --data data/processed/smartfall.npz --output runs/lighthart_kd/reevaluated.json
```

Evaluation uses checkpoint metadata and never searches a test threshold. An
explicit `--threshold` override is possible for a previously selected value; it
is recorded as an override/checkpoint source and should be disclosed in reports.
New evaluation/prediction JSON files refuse to overwrite existing paths.

Historical filenames map as follows; none of these weights is included:

| Source filename | Config | Printed validation-selected threshold |
| --- | --- | ---: |
| `best_student_model.pth` | `configs/it_student.json` | 0.38 |
| `best_teacher_model.pth` | `configs/teacher.json` | 0.30 |
| `best_kd_student.pth` | `configs/it_kd.json` | 0.44 |
| `best_lighthart_v41.pth` | `configs/lighthart.json` | 0.44 |
| `best_teacher_model_v3.pth` | `configs/shared_teacher.json` | 0.36 |
| `best_lighthart_kd_v4.pth` | `configs/lighthart_kd.json` | 0.50 |

```bash
smartfall evaluate --checkpoint checkpoints/best_student_model.pth --config configs/it_student.json --threshold 0.38 --data data/processed/smartfall.npz --output runs/archived_student_evaluation.json
```

Weights load with `weights_only=True`, strict key/shape checking, and CPU mapping
before the requested device move. Use checkpoint files from trusted sources.

## Prediction contract

`smartfall predict` accepts a finite nonempty float-compatible NPY array shaped
`(N,64,4)` containing historically standardized/aligned windows. It returns a
probability and binary prediction for every input, including a final partial
batch. For teacher inference, supply an aligned `(N,64,32,3)` `--skeleton` array.
The CLI does not infer activity IDs or recreate historical normalization from a
single raw window. This is a material constraint, not an implemented streaming
interface.

## Notebook walkthrough

```bash
python -m pip install -e ".[notebook]"
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/evidence_walkthrough.ipynb
```

The walkthrough reads archived evidence, shows the two figures, and checks
current model interfaces on synthetic tensors. It does not train or read the
original dataset. It is intentionally separate from the unchanged source notebook,
which cannot run here without data and stops at its original export dependency.

## Validation commands

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python scripts/check_evidence.py
python scripts/smoke_pipeline.py
python scripts/generate_figures.py
```

Review `docs/VALIDATION.md` for checks actually completed in this delivery.
