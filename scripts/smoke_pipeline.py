"""Synthetic end-to-end checks, including CSV preprocessing and all six runs.

No output here is a dataset benchmark. All files are temporary by default.
Use --keep-dir PATH to retain validation evidence outside the repository.
"""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from smartfall.cli import main as cli
from smartfall.preprocessing import preprocess
from smartfall.train import train

ROOT = Path(__file__).resolve().parents[1]


def run(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    torch.set_num_threads(1)
    # Exercise real paired-CSV discovery, filtering, DTW and window extraction.
    raw = work / "csv"
    raw.mkdir()
    for subject in [29, 38, 56]:
        for activity in [1, 10]:
            name = f"S{subject:02d}A{activity:02d}T01.csv"
            for modality, cols in [("skeleton", 96), ("accelerometer/watch", 4)]:
                directory = raw / "young" / modality
                directory.mkdir(parents=True, exist_ok=True)
                values = rng.normal(size=(220, cols)).astype("float32")
                # Header exists here to exercise the historical header=0 path.
                np.savetxt(
                    directory / name,
                    values,
                    delimiter=",",
                    header=",".join(f"c{i}" for i in range(cols)),
                    comments="",
                )
    with contextlib.redirect_stdout(io.StringIO()):
        shapes = preprocess(raw, work / "preprocessed.npz")
    assert shapes["accelerometer"][0] > 0
    # Fixed tiny arrays guarantee both classes per split for train/evaluate checks.
    subjects = np.array([29] * 8 + [38] * 4 + [56] * 4, dtype="int32")
    labels = np.arange(16) % 2
    arrays = {
        "accelerometer": rng.normal(size=(16, 64, 4)).astype("float32"),
        "skeleton": rng.normal(size=(16, 64, 32, 3)).astype("float32"),
        "subjects": subjects,
        "labels": labels.astype("int32"),
        "activities": np.where(labels, 10, 1).astype("int32"),
    }
    dataset = work / "tiny.npz"
    np.savez(dataset, **arrays)
    split = work / "subjects.json"
    split.write_text(json.dumps({"train": [29], "val": [38], "test": [56]}))
    summary = {}
    paths = {}
    for name in ["it_student", "teacher", "it_kd", "lighthart", "shared_teacher", "lighthart_kd"]:
        config = json.loads((ROOT / f"configs/{name}.json").read_text())
        config["epochs"] = 1
        config["batch_size"] = 2
        teacher_checkpoint = (
            paths.get("teacher" if name == "it_kd" else "shared_teacher")
            if name.endswith("_kd")
            else None
        )
        student_checkpoint = paths.get("it_student") if name == "it_kd" else None
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(
                config, dataset, split, work / name, "cpu", teacher_checkpoint, student_checkpoint
            )
        assert result["samples"] == 4 and np.isfinite(result["loss"])
        paths[name] = work / name / "best.pt"
        summary[name] = {"status": "passed", "test_samples": result["samples"], "finite_loss": True}
        with contextlib.redirect_stdout(io.StringIO()):
            cli(
                [
                    "evaluate",
                    "--data",
                    str(dataset),
                    "--subjects",
                    str(split),
                    "--checkpoint",
                    str(paths[name]),
                    "--output",
                    str(work / f"{name}_reevaluated.json"),
                ]
            )
        again = json.loads((work / f"{name}_reevaluated.json").read_text())
        for field in ["accuracy", "precision", "recall", "f1", "auc", "threshold", "loss"]:
            assert result[field] == again[field]
    acc = work / "input.npy"
    np.save(acc, arrays["accelerometer"][:3])
    with contextlib.redirect_stdout(io.StringIO()):
        cli(
            [
                "predict",
                "--accelerometer",
                str(acc),
                "--checkpoint",
                str(paths["lighthart_kd"]),
                "--output",
                str(work / "predictions.json"),
            ]
        )
    predictions = json.loads((work / "predictions.json").read_text())
    assert len(predictions["probabilities"]) == 3
    # Every record is structural validation, not a claim about real-data scores.
    report = {
        "scope": "synthetic CPU validation only",
        "csv_preprocessing": "passed",
        "six_training_runs": summary,
        "checkpoint_reevaluation": "exact metric equality",
        "accelerometer_only_prediction": "3 windows passed",
    }
    (work / "smoke_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-dir")
    args = parser.parse_args()
    if args.keep_dir:
        run(args.keep_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="smartfall-smoke-") as work:
            run(work)


if __name__ == "__main__":
    main()
