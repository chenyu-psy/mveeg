"""Subject-level execution helpers for preprocessing workflows."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .epochs import keep_epochs_by_metadata_value, reset_epoch_event_samples


def load_subject_streams(flow, subject_number: str):
    """Load one subject's EEG, optional eye data, and optional behavior table."""
    pre = flow._require_preprocessor()
    eeg, eeg_events = pre.import_eeg(subject_number, overwrite=True)
    eye = None
    eye_events = None
    has_eye_data = True
    try:
        eye, eye_events = pre.import_eyetracker(subject_number, overwrite=True)
    except FileNotFoundError as err:
        has_eye_data = False
        print(f"Skipping eyetracking import for subject {subject_number}: {err}")

    behavior_data = None
    if flow.behavior_name_pattern is None:
        return eeg, eeg_events, eye, eye_events, has_eye_data, behavior_data

    pre.import_behavior(subject_number, name_pattern=flow.behavior_name_pattern)

    behavior_data = pre.load_behavior_table(subject_number, name_pattern=flow.behavior_name_pattern)
    if flow.pre_filter_rules is not None:
        if "trial_types" in flow.pre_filter_rules:
            behavior_data = behavior_data[behavior_data["trial_type"].isin(flow.pre_filter_rules["trial_types"])]
        if "rejection" in flow.pre_filter_rules:
            behavior_data = behavior_data[behavior_data["rejection"].eq(flow.pre_filter_rules["rejection"])]
    behavior_data = behavior_data.reset_index(drop=True)
    return eeg, eeg_events, eye, eye_events, has_eye_data, behavior_data


def build_subject_epochs(
    flow,
    subject_number: str,
    eeg: mne.io.BaseRaw,
    eeg_events: np.ndarray,
    eye: mne.io.BaseRaw | None,
    eye_events: np.ndarray | None,
    *,
    has_eye_data: bool,
) -> mne.Epochs:
    """Build epochs from imported recording streams for one subject."""
    pre = flow._require_preprocessor()
    manual_trial_exclusions = flow._manual_trial_exclusions()

    eeg.load_data()
    if flow.reref_channels is not None:
        reref_index = mne.pick_channels(eeg.ch_names, list(flow.reref_channels))
        if len(reref_index) == 0:
            raise ValueError(f"Could not find rereference channels: {flow.reref_channels}")
        eeg.apply_function(
            pre.rereference_to_average,
            picks=["eeg"],
            reref_values=np.squeeze(eeg.get_data()[reref_index]),
        )
    eeg.filter(*pre.filter_freqs, n_jobs=-1, verbose="ERROR")
    if has_eye_data:
        return pre.make_and_sync_epochs(
            eeg,
            eeg_events,
            eye,
            eye_events,
            eeg_trials_drop=manual_trial_exclusions["eeg"].get(subject_number, []),
            eye_trials_drop=manual_trial_exclusions["eye"].get(subject_number, []),
        )
    return pre.make_eeg_epochs(
        eeg,
        eeg_events,
        eeg_trials_drop=manual_trial_exclusions["eeg"].get(subject_number, []),
    )


def keep_aligned_experimental_trials(
    flow,
    subject_number: str,
    epochs: mne.Epochs,
    behavior_data: pd.DataFrame | None,
) -> mne.Epochs:
    """Align epochs to behavior rows and keep requested trial types."""
    if behavior_data is None:
        return epochs

    pre = flow._require_preprocessor()

    def mark_epochs_to_keep(matched_behavior_data: pd.DataFrame) -> pd.DataFrame:
        """Mark aligned rows that should remain in the saved epochs."""
        matched_behavior_data = matched_behavior_data.copy()
        if flow.post_filter_rules is not None and "keep_trial_type" in flow.post_filter_rules:
            keep_epoch = matched_behavior_data["trial_type"].eq(flow.post_filter_rules["keep_trial_type"])
        else:
            keep_epoch = pd.Series(True, index=matched_behavior_data.index)
        matched_behavior_data["keep_epoch"] = keep_epoch
        return matched_behavior_data

    return pre.exclude_practice_trials(
        subject_number,
        epochs,
        name_pattern=flow.behavior_name_pattern,
        behavior=behavior_data,
        matched_behavior_filter=mark_epochs_to_keep,
    )


def prepare_subject_epochs(flow, subject_number: str) -> mne.Epochs:
    """Import, epoch, and optionally behavior-align one raw recording."""
    eeg, eeg_events, eye, eye_events, has_eye_data, behavior_data = flow.load_subject_streams(subject_number)
    epochs = flow.build_subject_epochs(
        subject_number,
        eeg,
        eeg_events,
        eye,
        eye_events,
        has_eye_data=has_eye_data,
    )
    if behavior_data is None:
        return epochs
    return flow.keep_aligned_experimental_trials(subject_number, epochs, behavior_data)


def run_subject_qc(flow, epochs: mne.Epochs) -> dict[str, object]:
    """Run the stored QC pipeline on one subject's epochs."""
    from . import qc as preprocess_qc

    pre = flow._require_preprocessor()
    reject_qc, review_qc, autoreject_cfg = flow._require_qc_config()
    return preprocess_qc.run_subject_artifact_qc(
        pre,
        epochs,
        reject_qc=reject_qc,
        review_qc=review_qc,
        autoreject_cfg=autoreject_cfg,
    )


def run_subject_reject_qc(flow, epochs: mne.Epochs, progress_callback=None) -> dict[str, object]:
    """Run the hard-rejection QC stage on one subject's epochs."""
    from . import qc as preprocess_qc

    pre = flow._require_preprocessor()
    reject_qc, _, _ = flow._require_qc_config()
    return preprocess_qc.run_subject_reject_qc(
        pre,
        epochs,
        reject_qc=reject_qc,
        progress_callback=progress_callback,
    )


def run_subject_autoreject(
    flow,
    epochs: mne.Epochs,
    reject_result: dict[str, object],
    progress_callback=None,
) -> dict[str, object]:
    """Run only autoreject for one subject and return aligned trial flags."""
    from . import qc as preprocess_qc

    pre = flow._require_preprocessor()
    _, _, autoreject_cfg = flow._require_qc_config()
    return preprocess_qc.run_subject_autoreject(
        pre,
        epochs,
        reject_result=reject_result,
        autoreject_cfg=autoreject_cfg,
        progress_callback=progress_callback,
    )


def run_subject_soft_review_qc(
    flow,
    epochs: mne.Epochs,
    reject_result: dict[str, object],
    *,
    autoreject_bad_epoch: np.ndarray | None = None,
    autoreject_interp_channels: np.ndarray | None = None,
    progress_callback=None,
) -> dict[str, object]:
    """Run only soft-review QC using supplied autoreject trial decisions."""
    from . import qc as preprocess_qc

    pre = flow._require_preprocessor()
    reject_qc, review_qc, _ = flow._require_qc_config()
    return preprocess_qc.run_subject_soft_review_qc(
        pre,
        epochs,
        reject_result=reject_result,
        review_qc=review_qc,
        hard_bad_channel_limit=reject_qc["eeg"]["bad_channels"],
        autoreject_bad_epoch=autoreject_bad_epoch,
        autoreject_interp_channels=autoreject_interp_channels,
        progress_callback=progress_callback,
    )


def summarize_trial_rejection(
    flow,
    *,
    stage: str = "auto",
    hard_stage: str = "prepared",
    top_n_reasons: int = 3,
) -> None:
    """Print a trial-rejection summary using this workflow's data directory."""
    from . import qc as preprocess_qc

    preprocess_qc.summarize_trial_rejection(
        flow.data_dir,
        stage=stage,
        hard_stage=hard_stage,
        top_n_reasons=top_n_reasons,
    )


def log_subject_qc_summary(flow, subject_number: str) -> None:
    """Print detailed QC summary for one saved subject on demand."""
    from . import qc as preprocess_qc

    if not flow.has_subject_reject_checkpoint(subject_number):
        print(f"{subject_number}: reject checkpoint not found.")
        return
    if not flow.has_subject_autoreject_checkpoint(subject_number):
        print(f"{subject_number}: autoreject checkpoint not found.")
        return

    reject_result = flow.load_checkpoint(subject_number, kind="reject")
    autoreject_checkpoint = flow.load_checkpoint(subject_number, kind="autoreject")
    review_result = flow.run_subject_soft_review_qc(
        autoreject_checkpoint["epochs_for_soft_qc"],
        reject_result,
        autoreject_bad_epoch=autoreject_checkpoint["autoreject_bad_epoch"],
        autoreject_interp_channels=autoreject_checkpoint["autoreject_interp_channels"],
    )
    preprocess_qc.log_subject_qc_summary(review_result, flow.reject_qc, flow.review_qc)


def save_subject_data(
    flow,
    subject_number: str,
    epochs: mne.Epochs,
    artifact_labels: np.ndarray,
    trial_qc: pd.DataFrame,
) -> None:
    """Save one subject's epochs and QC outputs using the stored preprocessor."""
    pre = flow._require_preprocessor()
    pre.save_all_data(subject_number, epochs, artifact_labels, trial_qc)
    flow._save_final_trial_state(subject_number, trial_qc)


def build_subject_intermediate_epochs(
    flow,
    subject_number: str,
    subject_assembly_plan: dict[str, str],
    *,
    stage: str = "prepared",
    overwrite: bool = True,
) -> Path:
    """Prepare one subject and save intermediate epochs before trial QC."""
    epochs = flow.prepare_subject_epochs(subject_number)
    if len(subject_assembly_plan) > 0:
        epochs = flow.assemble_subject_epochs(
            subject_number=subject_number,
            base_epochs=epochs,
            assembly_plan=subject_assembly_plan,
        )
    epochs = reset_epoch_event_samples(epochs)
    return flow.save_intermediate_epochs(subject_number, epochs, stage=stage, overwrite=overwrite)


def build_all_intermediate_epochs(
    flow,
    *,
    subject_assembly_plans: dict[str, dict[str, str]] | None = None,
    stage: str = "prepared",
    overwrite: bool = True,
) -> None:
    """Build and save intermediate epochs for all selected subjects."""
    overwrite = flow.should_overwrite(overwrite)
    subject_dirs = flow.subject_dirs
    if not overwrite:
        subject_dirs = [
            subject_dir
            for subject_dir in subject_dirs
            if not flow.has_saved_intermediate_epochs(subject_dir.name, stage=stage)
        ]
    total_subjects = len(subject_dirs)
    if total_subjects == 0:
        print("No subjects selected for preprocessing. Check overwrite settings and selected_subjects.")
        return

    subject_assembly_plans = {} if subject_assembly_plans is None else subject_assembly_plans
    subject_bar = tqdm(
        subject_dirs,
        total=total_subjects,
        desc="Build intermediate epochs",
        unit="subject",
    )

    for subject_path in subject_bar:
        subject_number = subject_path.name
        subject_assembly_plan = subject_assembly_plans.get(subject_number, {})
        subject_bar.set_postfix_str(subject_number)
        flow.build_subject_intermediate_epochs(
            subject_number,
            subject_assembly_plan,
            stage=stage,
            overwrite=overwrite,
        )
    subject_bar.close()


def finish_subject_from_intermediate(
    flow,
    subject_number: str,
    *,
    stage: str = "prepared",
    reuse_saved_reject: bool = True,
    reject_progress_callback=None,
) -> dict[str, object]:
    """Load intermediate epochs, run QC, and save final subject outputs."""
    from . import qc as preprocess_qc

    epochs = flow.load_intermediate_epochs(subject_number, stage=stage)
    if reuse_saved_reject and flow.has_subject_reject_checkpoint(subject_number, stage=stage):
        reject_result = flow.load_checkpoint(subject_number, kind="reject", stage=stage)
    else:
        reject_result = flow.run_subject_reject_qc(
            epochs,
            progress_callback=reject_progress_callback,
        )
        flow.save_subject_reject_checkpoint(
            subject_number,
            epochs,
            reject_result,
            stage=stage,
        )
    autoreject_result = flow.run_subject_autoreject(epochs, reject_result)
    qc_result = flow.run_subject_soft_review_qc(
        autoreject_result["epochs_for_soft_qc"],
        reject_result,
        autoreject_bad_epoch=autoreject_result["autoreject_bad_epoch"],
        autoreject_interp_channels=autoreject_result["autoreject_interp_channels"],
    )
    epochs = qc_result["epochs"]
    artifact_labels = qc_result["artifact_labels"]
    trial_qc = qc_result["trial_qc"]
    preprocess_qc.log_subject_qc_summary(qc_result, flow.reject_qc, flow.review_qc)
    flow.save_subject_data(subject_number, epochs, artifact_labels, trial_qc)
    return qc_result


def assemble_subject_epochs(
    flow,
    *,
    subject_number: str,
    base_epochs: mne.Epochs,
    assembly_plan: dict[str, str],
    condition_column: str = "condition",
) -> mne.Epochs:
    """Assemble one saved subject from one or more source recordings."""
    if not flow.subject_dir_map:
        raise ValueError("assemble_subject_epochs requires subject_dir_map when donor recordings are used.")
    if base_epochs.metadata is None or condition_column not in base_epochs.metadata.columns:
        raise ValueError(f"Base epochs metadata must contain '{condition_column}' for subject assembly.")

    epoch_parts = []
    condition_values = [
        str(value)
        for value in base_epochs.metadata[condition_column].dropna().drop_duplicates().tolist()
    ]
    keep_map = {condition_value: subject_number for condition_value in condition_values}
    keep_map.update(assembly_plan)

    print(f"Assembly plan for {subject_number}: {keep_map}")
    for condition_value, source_subject in keep_map.items():
        if source_subject == subject_number:
            source_epochs = base_epochs.copy()
        else:
            source_epochs = flow.prepare_subject_epochs(source_subject)

        source_epochs = keep_epochs_by_metadata_value(source_epochs, condition_column, condition_value)
        epoch_parts.append(source_epochs)
        print(
            f"Using {len(source_epochs)} {condition_value} trials from {source_subject} "
            f"for saved subject {subject_number}."
        )

    return mne.concatenate_epochs(epoch_parts, add_offset=False)
