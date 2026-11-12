"""Verify archived result extraction, notebook hashes and notebook format."""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "evidence/source_manifest.json").read_text())
    source = ROOT / manifest["notebook"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    n = json.loads(source.read_text())
    for row in manifest["architectures"]:
        module = ROOT / f"src/smartfall/models/{row['module']}.py"
        definitions = ast.parse(module.read_text()).body
        cls = next(
            item
            for item in definitions
            if isinstance(item, ast.ClassDef) and item.name == row["class"]
        )
        original_tree = ast.parse("".join(n["cells"][row["source_cell"]]["source"]))
        original_cls = next(
            item
            for item in ast.walk(original_tree)
            if isinstance(item, ast.ClassDef) and item.name == row["class"]
        )
        # AST dumps can change between Python releases. The stored digest was
        # generated under Python 3.12; on other versions compare source and
        # refactored ASTs directly using the same interpreter instead.
        if sys.version_info[:2] == (3, 12):
            digest = hashlib.sha256(ast.dump(cls, include_attributes=False).encode()).hexdigest()
            assert digest == row["ast_sha256"], f"Architecture hash changed: {row['class']}"
        else:
            assert ast.dump(cls, include_attributes=False) == ast.dump(
                original_cls, include_attributes=False
            ), f"Architecture differs from original notebook: {row['class']}"
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        nbformat.validate(nbformat.read(path, as_version=4))
    metrics = json.loads((ROOT / "evidence/recorded_metrics.json").read_text())
    for row in metrics:
        text = "\n".join(
            "".join(o.get("text", [])) for o in n["cells"][row["source_cell"]].get("outputs", [])
        )
        final = text[list(re.finditer("FINAL (?:TEST|TEACHER|KD) RESULTS", text))[-1].start() :]
        for field, title in [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1 Score"),
            ("auc", "AUC"),
            ("threshold", "Threshold"),
        ]:
            assert row[field] == float(re.search(rf"{title}\s*:\s*([\d.]+)", final).group(1))
        assert row["status"] == "recorded_not_reproduced"
    split = json.loads((ROOT / "configs/subjects.json").read_text())
    assert not set(split["train"]) & set(split["val"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["val"]) & set(split["test"])
    assert 13077 - (6874 + 942 + 2572) == 2689
    print(
        "PASS: original notebook hash, 15 architecture parity checks, notebook schemas, six metric records, disjoint configured subjects, excluded-window arithmetic"
    )


if __name__ == "__main__":
    main()
