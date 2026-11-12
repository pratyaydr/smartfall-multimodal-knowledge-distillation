"""Command line entry points; no Kaggle runtime or credentials required."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import SmartFallDataset, load_arrays, split_arrays
from .evaluate import binary_metrics, collect_predictions
from .models import build_model
from .train import make_criterion, train
from .utils import load_checkpoint, write_json


def checkpoint_model(path, config_path=None, threshold=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = json.loads(Path(config_path).read_text()) if config_path else payload.get("config")
    if config is None:
        raise ValueError("Archived bare weights require --config and --threshold")
    cutoff = threshold if threshold is not None else payload.get("threshold")
    if cutoff is None:
        raise ValueError("Supply the validation-selected --threshold for archived weights")
    if not 0 <= cutoff <= 1:
        raise ValueError("threshold must lie in [0,1]")
    model = build_model(config["model"], config["model_kwargs"])
    load_checkpoint(model, path)
    return model, config, float(cutoff)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SmartFallMM offline research experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    preprocess = sub.add_parser("preprocess", help="Historical offline CSV -> NPZ pipeline")
    preprocess.add_argument("--data-root", required=True)
    preprocess.add_argument("--output", required=True)
    training = sub.add_parser("train", help="Train one configured experiment")
    training.add_argument("--config", required=True)
    training.add_argument("--data", required=True)
    training.add_argument("--subjects", default="configs/subjects.json")
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int)
    training.add_argument("--teacher-checkpoint")
    training.add_argument("--student-checkpoint")
    evaluation = sub.add_parser("evaluate", help="Evaluate a checkpoint without threshold retuning")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--subjects", default="configs/subjects.json")
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    prediction = sub.add_parser("predict", help="Predict from already preprocessed windows")
    prediction.add_argument("--accelerometer", required=True, help="NPY with shape (N,64,4)")
    prediction.add_argument("--skeleton", help="NPY with shape (N,64,32,3); required by teachers")
    for cmd in (evaluation, prediction):
        cmd.add_argument("--checkpoint", required=True)
        cmd.add_argument("--config", help="Required for archived bare weights")
        cmd.add_argument(
            "--threshold", type=float, help="Validation-selected value; never tuned on test"
        )
        cmd.add_argument("--output", required=True)
    for cmd in (training, evaluation, prediction):
        cmd.add_argument("--device", default="auto")
        cmd.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        if Path(args.output).suffix != ".npz":
            parser.error("preprocess --output must end in .npz")
        from .preprocessing import preprocess as process

        print(json.dumps(process(args.data_root, args.output), indent=2))
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if args.command == "train":
        config = json.loads(Path(args.config).read_text())
        if args.epochs is not None:
            config["epochs"] = args.epochs
        result = train(
            config,
            args.data,
            args.subjects,
            args.output,
            device,
            args.teacher_checkpoint,
            args.student_checkpoint,
        )
    else:
        if Path(args.output).exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        model, config, cutoff = checkpoint_model(args.checkpoint, args.config, args.threshold)
        model.to(device).eval()
        if args.command == "evaluate":
            parts, _ = split_arrays(
                load_arrays(args.data), json.loads(Path(args.subjects).read_text())
            )
            loader = torch.utils.data.DataLoader(
                SmartFallDataset(parts[args.split]), batch_size=config["batch_size"]
            )
            y, probabilities, loss = collect_predictions(
                model, loader, device, make_criterion(config, device)
            )
            result = binary_metrics(y, probabilities, cutoff)
            result.update(
                {
                    "loss": loss,
                    "split": args.split,
                    "threshold_source": "checkpoint or explicit override",
                }
            )
        else:
            acc = np.load(args.accelerometer, allow_pickle=False)
            if (
                acc.ndim != 3
                or acc.shape[1:] != (64, 4)
                or len(acc) == 0
                or not np.isfinite(acc).all()
            ):
                raise ValueError("Accelerometer must be finite, nonempty (N,64,4)")
            arrays = {"accelerometer": acc}
            if config["model"] in ("teacher", "shared_teacher") and not args.skeleton:
                raise ValueError("Multimodal teachers require --skeleton")
            if args.skeleton:
                skl = np.load(args.skeleton, allow_pickle=False)
                if skl.shape != (len(acc), 64, 32, 3) or not np.isfinite(skl).all():
                    raise ValueError("Skeleton must be finite (N,64,32,3) and aligned")
                arrays["skeleton"] = skl
            probabilities = []
            with torch.no_grad():
                for start in range(0, len(acc), config["batch_size"]):
                    batch = {
                        k: torch.as_tensor(
                            v[start : start + config["batch_size"]],
                            dtype=torch.float32,
                            device=device,
                        )
                        for k, v in arrays.items()
                    }
                    probabilities.extend(torch.sigmoid(model(batch)[0].view(-1)).cpu().tolist())
            result = {
                "probabilities": probabilities,
                "predictions": [int(p >= cutoff) for p in probabilities],
                "threshold": cutoff,
                "positive_class": "fall",
                "input_status": "already preprocessed; no streaming claim",
            }
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
