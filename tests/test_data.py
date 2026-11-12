import numpy as np
import pytest

from smartfall.data import load_arrays, make_loaders, split_arrays, validate_arrays

SPLIT = {"train": [29], "val": [38], "test": [56]}


def test_split_excludes_unassigned_subjects(sample_arrays):
    parts, summary = split_arrays(sample_arrays, SPLIT)
    assert [len(parts[k]["labels"]) for k in ("train", "val", "test")] == [4, 4, 4]
    assert summary["excluded"] == {"samples": 1, "subjects": [99]}


def test_subject_overlap_is_rejected(sample_arrays):
    with pytest.raises(ValueError, match="overlap"):
        split_arrays(sample_arrays, {"train": [29], "val": [29, 38], "test": [56]})


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("accelerometer", lambda x: x[:, :63, :]),
        ("skeleton", lambda x: x[:, :, :, 0]),
        ("labels", lambda x: np.full_like(x, 2)),
        ("activities", lambda x: np.ones_like(x)),
        ("subjects", lambda x: x.astype(float) + 0.5),
    ],
)
def test_invalid_schema(sample_arrays, field, mutation):
    sample_arrays[field] = mutation(sample_arrays[field])
    with pytest.raises(ValueError):
        validate_arrays(sample_arrays)


def test_nonfinite_inputs(sample_arrays):
    sample_arrays["accelerometer"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_arrays(sample_arrays)


def test_npz_roundtrip_and_no_silent_empty_batches(tmp_path, sample_arrays):
    import json

    data = tmp_path / "data.npz"
    np.savez(data, **sample_arrays)
    loaded = load_arrays(data)
    np.testing.assert_array_equal(loaded["labels"], sample_arrays["labels"])
    split = tmp_path / "split.json"
    split.write_text(json.dumps(SPLIT))
    with pytest.raises(ValueError, match="no batches"):
        make_loaders(data, split, {"batch_size": 32, "drop_last": True})


def test_missing_npz_keys(tmp_path):
    data = tmp_path / "bad.npz"
    np.savez(data, labels=[0])
    with pytest.raises(ValueError, match="Missing NPZ"):
        load_arrays(data)
