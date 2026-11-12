import ast
import json
from pathlib import Path

import numpy as np
import pytest

from smartfall import preprocessing as p

ROOT = Path(__file__).resolve().parents[1]


def original_namespace():
    notebook = json.loads((ROOT / "notebooks/original_kaggle.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {"WINDOW": 64, "OVERLAP": 32, "GAP": 20, "NUM_JOINTS": 32, "NUM_COORDS": 3}
    exec(
        compile(ast.Module(body=keep, type_ignores=[]), "<original preprocessing>", "exec"),
        namespace,
    )
    return namespace


def test_smv_is_mean_centered():
    values = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    np.testing.assert_allclose(p.compute_smv(values), [[np.sqrt(2)], [np.sqrt(2)]])
    np.testing.assert_array_equal(p.compute_smv(np.ones((5, 3))), np.zeros((5, 1)))


def test_short_filter_guard_handles_original_padlen_bug():
    values = np.ones((15, 3))
    np.testing.assert_array_equal(p.butterworth_filter(values), values)


def test_dtw_unique_indices_are_monotone_and_one_to_one():
    a, s = p.filter_repeated_ids([(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 3)])
    assert a == [0, 1, 2, 3] and s == [0, 1, 2, 3]


def test_trial_gaps_prevent_windows_crossing():
    acc = np.column_stack([np.arange(192), np.zeros((192, 3))])
    skl = np.zeros((192, 32, 3))
    mask = np.zeros(192, dtype=bool)
    mask[80:100] = True
    windows = p.extract_windows_with_gap_mask(acc, skl, 0, mask)
    starts = [int(a[0, 0]) for a, _ in windows]
    assert starts == [0, 128]
    assert all(not mask[s : s + 64].any() for s in starts)


def test_preserved_label_conditioned_peak_selection():
    acc = np.zeros((200, 4))
    acc[:, 0] = np.arange(200)
    acc[77, 3] = 1.3
    skl = np.zeros((200, 32, 3))
    mask = np.zeros(200, dtype=bool)
    adl = p.extract_windows_with_gap_mask(acc, skl, 0, mask)
    fall = p.extract_windows_with_gap_mask(acc, skl, 1, mask)
    assert any(int(a[0, 0]) == 45 for a, _ in adl)
    assert all(int(a[0, 0]) != 45 for a, _ in fall)


def test_original_header_and_row_trimming_are_preserved(tmp_path):
    acc = np.arange(80 * 4).reshape(80, 4)
    skl = np.arange(80 * 96).reshape(80, 96)
    ap = tmp_path / "acc.csv"
    sp = tmp_path / "skl.csv"
    np.savetxt(ap, acc, delimiter=",")
    np.savetxt(sp, skl, delimiter=",")
    a, s = p.load_trial(ap, sp)
    assert a.shape == (77, 3) and s.shape == (79, 32, 3)
    np.testing.assert_array_equal(a, acc[3:, -3:])


def test_unreadable_csv_is_reported(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.warns(RuntimeWarning, match="Skipping unreadable"):
        assert p.csv_loader(path) is None


def test_preprocessing_parity_against_source_on_synthetic_trials():
    original = original_namespace()
    rng = np.random.default_rng(8)
    trials = [
        (
            i,
            rng.normal(size=(400, 3)).astype("float32"),
            rng.normal(size=(400, 32, 3)).astype("float32"),
        )
        for i in [2, 1]
    ]
    expected = original["process_group"](trials, 1)
    actual = p.process_group(trials, 1)
    assert len(expected) == len(actual) > 0
    for (a, s), (a0, s0) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, a0)
        np.testing.assert_array_equal(s, s0)
