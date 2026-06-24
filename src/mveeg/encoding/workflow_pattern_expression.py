"""Pattern-expression export helpers for encoding workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .._shared.workflow_subjects import process_subjects
from .io import (
    CONDITION_OUTPUT_FILENAME,
    README_FILENAME,
    TRIAL_OUTPUT_FILENAME,
    load_saved_subject_results,
    save_subject_results,
    subject_result_exists,
    write_pattern_expression_readme,
)
from .summaries import (
    build_condition_average_pattern_expression_table,
    build_trial_pattern_expression_table,
)


LEGACY_OUTPUT_FILES = {
    "trial_expression": TRIAL_OUTPUT_FILENAME,
    "condition_expression": CONDITION_OUTPUT_FILENAME,
    "trial_summary": "trials.csv",
    "skipped_subjects": "skipped.csv",
    "readme": README_FILENAME,
}


def run_pattern_expression_workflow(
    subject_ids: list[str],
    subject_inputs: dict[str, dict[str, object]],
    subject_results_dir: str | Path,
    overwrite: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, object]:
    """Run pattern-expression export across requested subjects.

    Parameters
    ----------
    subject_ids : list[str]
        Subject IDs requested by the caller.
    subject_inputs : dict[str, dict[str, object]]
        Per-subject arrays used to build trial- and condition-level expression
        tables.
    subject_results_dir : str | Path
        Folder used for per-subject cache files.
    overwrite : bool
        Whether to rebuild subjects that already have saved cache files.
    log_path : str | Path | None
        Optional detailed log path.

    Returns
    -------
    dict[str, object]
        Trial summary, skipped subjects, trial expression, and condition
        expression tables.
    """

    trial_tables = {}
    condition_tables = {}
    trial_summary_rows = []

    def process_one_subject(subject_id: str, progress_bar):
        """Export or reuse one subject and update the shared progress bar."""

        used_saved_result = False
        if not overwrite and subject_result_exists(subject_results_dir, subject_id):
            subject_saved = load_saved_subject_results(subject_results_dir, subject_id)
            trial_table = subject_saved["trial_table"]
            condition_table = subject_saved["condition_table"]
            used_saved_result = True
        else:
            if subject_id not in subject_inputs:
                raise KeyError(
                    "Missing subject input bundle. "
                    f"Expected key '{subject_id}' in subject_inputs."
                )

            trial_table, condition_table = _run_single_subject_export(
                subject_id=subject_id,
                subject_input=subject_inputs[subject_id],
                subject_results_dir=subject_results_dir,
            )

        trial_tables[subject_id] = trial_table
        condition_tables[subject_id] = condition_table
        trial_summary_rows.append(
            {
                "subject": str(subject_id),
                "n_trial_rows": int(len(trial_table)),
                "n_condition_rows": int(len(condition_table)),
            }
        )

        progress_bar.update(1)
        return (trial_table, condition_table), used_saved_result

    skipped_subjects_df = process_subjects(
        subject_ids=[str(subject_id) for subject_id in subject_ids],
        progress_total=1,
        log_label="Encoding pattern-expression run",
        log_path=log_path,
        experiment_name="encoding",
        process_one_subject=process_one_subject,
    )

    if len(trial_tables) == 0:
        skipped_summary = (
            skipped_subjects_df
            if len(skipped_subjects_df) > 0
            else pd.DataFrame(columns=["subject", "reason"])
        )
        raise RuntimeError(
            "No subjects were successfully exported.\n"
            f"Failure summary:\n{skipped_summary.to_string(index=False)}"
        )

    trial_summary_df = pd.DataFrame(trial_summary_rows).sort_values("subject").reset_index(drop=True)

    trial_expression_df = pd.concat(list(trial_tables.values()), ignore_index=True)
    trial_expression_df = trial_expression_df.sort_values(
        ["subject", "condition", "effect", "trial_index", "time"]
    ).reset_index(drop=True)

    condition_expression_df = pd.concat(list(condition_tables.values()), ignore_index=True)
    condition_expression_df = condition_expression_df.sort_values(
        ["subject", "condition", "effect", "time"]
    ).reset_index(drop=True)

    return {
        "trial_summary_df": trial_summary_df,
        "skipped_subjects_df": skipped_subjects_df,
        "trial_expression_df": trial_expression_df,
        "condition_expression_df": condition_expression_df,
    }



def export_encoding_outputs(
    run_output: dict[str, object],
    results_dir: str | Path,
) -> None:
    """Save group-level pattern-expression output tables for one completed run."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trial_summary_df = run_output["trial_summary_df"]
    skipped_subjects_df = run_output["skipped_subjects_df"]
    trial_expression_df = run_output["trial_expression_df"]
    condition_expression_df = run_output["condition_expression_df"]

    trial_expression_df.to_csv(results_dir / LEGACY_OUTPUT_FILES["trial_expression"], index=False)
    condition_expression_df.to_csv(results_dir / LEGACY_OUTPUT_FILES["condition_expression"], index=False)
    trial_summary_df.to_csv(results_dir / LEGACY_OUTPUT_FILES["trial_summary"], index=False)

    if len(skipped_subjects_df) > 0:
        skipped_subjects_df.to_csv(results_dir / LEGACY_OUTPUT_FILES["skipped_subjects"], index=False)

    write_pattern_expression_readme(results_dir)



def _run_single_subject_export(
    subject_id: str,
    subject_input: dict[str, object],
    subject_results_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save pattern-expression tables for one subject."""

    required_keys = {"condition_labels", "times", "expression_by_effect"}
    missing_keys = sorted(required_keys.difference(subject_input.keys()))
    if len(missing_keys) > 0:
        raise KeyError(
            f"Subject {subject_id} input is missing required keys: {missing_keys}"
        )

    trial_table = build_trial_pattern_expression_table(
        subject=str(subject_id),
        condition_labels=subject_input["condition_labels"],
        times=subject_input["times"],
        expression_by_effect=subject_input["expression_by_effect"],
        trial_index=subject_input.get("trial_index"),
    )
    condition_table = build_condition_average_pattern_expression_table(trial_table)

    save_subject_results(
        output_dir=subject_results_dir,
        subject_id=subject_id,
        trial_table=trial_table,
        condition_table=condition_table,
    )

    return trial_table, condition_table
