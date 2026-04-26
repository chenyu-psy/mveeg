"""Public workflow facade for encoding analyses.

These helpers keep encoding scripts focused on analysis decisions while smaller
workflow modules handle paths, design checks, pattern exports, and model runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .workflow_design import run_encoding_design_check, validate_glm_formula
from .workflow_model import (
    MODEL_OUTPUT_FILES,
    TEST_METHOD_NAME,
    TEST_METHOD_VERSION,
    run_encoding,
)
from .workflow_paths import prepare_encoding_paths
from .workflow_pattern_expression import (
    LEGACY_OUTPUT_FILES,
    export_encoding_outputs,
    run_pattern_expression_workflow,
)


__all__ = [
    "LEGACY_OUTPUT_FILES",
    "MODEL_OUTPUT_FILES",
    "TEST_METHOD_NAME",
    "TEST_METHOD_VERSION",
    "export_encoding_outputs",
    "prepare_encoding_paths",
    "run_encoding",
    "run_encoding_design_check",
    "run_pattern_expression",
    "run_pattern_expression_workflow",
    "validate_glm_formula",
]


def run_pattern_expression(
    *,
    base_dir: str | Path,
    subject_ids: list[str],
    subject_inputs: dict[str, dict[str, object]],
    overwrite: bool,
    name: str,
    results_subdir: str = "main",
) -> dict[str, object]:
    """Run a pattern-expression export analysis from script settings."""

    run_paths = prepare_encoding_paths(base_dir, name, results_subdir=results_subdir)

    print(f"Running {name}")
    print(f"Detailed log file: {run_paths['log_path']}")

    run_output = run_pattern_expression_workflow(
        subject_ids=subject_ids,
        subject_inputs=subject_inputs,
        subject_results_dir=run_paths["subject_results_dir"],
        overwrite=overwrite,
        log_path=run_paths["log_path"],
    )

    export_encoding_outputs(
        run_output=run_output,
        results_dir=run_paths["results_dir"],
    )

    summary_df = pd.DataFrame(
        {
            "name": [name],
            "n_subjects_requested": [len(subject_ids)],
            "n_subjects_completed": [len(run_output["trial_summary_df"])],
            "n_subjects_skipped": [len(run_output["skipped_subjects_df"])],
        }
    )

    return {
        "paths": run_paths,
        "run_output": run_output,
        "summary_df": summary_df,
    }
