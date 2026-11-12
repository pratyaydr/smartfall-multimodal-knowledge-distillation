"""Execute the data-free walkthrough when socket-based kernels are unavailable.

This is an explicit fallback, not a substitute claim of a normal Jupyter kernel
run. It captures real IPython stream/rich outputs and stops at any cell error.
"""

from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/evidence_walkthrough.ipynb"
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"Walkthrough code cell {count} failed") from (
                result.error_before_exec or result.error_in_exec
            )
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        outputs.extend(
            nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata)
            for output in captured.outputs
        )
        cell.execution_count = count
        cell.outputs = outputs
    notebook.metadata["smartfall"]["execution"] = {
        "mode": "IPython in-process fallback",
        "reason": "Normal nbconvert socket kernel startup blocked by host: Operation not permitted",
        "code_cells_executed": count,
        "scope": "Archived evidence and synthetic interfaces only",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed {count} code cells in process; outputs saved and schema validated")


if __name__ == "__main__":
    main()
