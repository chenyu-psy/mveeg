"""Dataset orchestration and transactional persistence for subspace comparison."""

from __future__ import annotations

import json
import secrets
import warnings

import duckdb
from tqdm.auto import tqdm

from .._provenance import fingerprint
from .._results.store import (
    delete_subject,
    insert_frame,
    mark_complete,
    mark_failed,
    mark_pending,
    require_equal_frame,
    subject_state,
    table_exists,
)
from ._comparison import compare_subject, condition_weights
from ._prepare import validate_groups
from ._storage import initialize_store, read_analysis, result_path

SUBJECT_TABLES = (
    "trials",
    "trial_groups",
    "model_designs",
    "pair_strengths",
    "strengths",
    "comparisons",
    "covariance_diagnostics",
)


def compare_pipeline(
    pipeline, *, models, target, conditions, file, groups, nuisance, weights, recompute, progress
):
    from ._pipeline import _input_fingerprint, _prepare_subject, _version

    pipeline._refresh_dataset()
    if len(models) < 2 or any(not isinstance(k, str) or not k.strip() for k in models):
        raise ValueError("models must contain at least two nonempty named formulas.")
    if any(not isinstance(f, str) or not f.strip() for f in models.values()):
        raise ValueError("Every model must be a nonempty RHS formula (use '1' for the null space).")
    if nuisance is not None and (not isinstance(nuisance, str) or not nuisance.strip()):
        raise ValueError("nuisance must be a nonempty RHS formula or None.")
    if groups is not None and (not isinstance(groups, str) or not groups.strip()):
        raise ValueError("groups must name an independent-group metadata column or be None.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must name a metadata column.")
    if recompute not in {"never", "changed", "all"}:
        raise ValueError("recompute must be 'never', 'changed', or 'all'.")
    condition_map, _ = validate_groups(conditions, None)
    normalized_weights = condition_weights(condition_map, weights)
    path = result_path(file)
    existing = read_analysis(path)
    if (
        existing is not None
        and json.loads(existing["config"]).get("analysis_kind") != "subspace_strength"
    ):
        raise ValueError("Use a separate file for subspace-strength comparison results.")
    seed = pipeline._cv["seed"]
    if seed is None:
        seed = int(existing["seed"]) if existing is not None else secrets.randbelow(2**63 - 1)
    config = {
        "analysis_kind": "subspace_strength",
        "estimator_version": 2,
        "minimum_noise_df": 3,
        "mveeg_version": _version(),
        "dataset": str(pipeline.dataset),
        "metadata_variables": list(pipeline._metadata_variables),
        "trial_selection": pipeline._trial_selection,
        "epoch_preparation": pipeline._epoch_preparation,
        "models": dict(models),
        "model_order": list(models),
        "nuisance": nuisance,
        "target": target,
        "conditions": condition_map,
        "weights": normalized_weights,
        "groups": groups,
        "cv": {"folds": pipeline._cv["folds"], "seed": seed},
        "precision": "leave_two_groups_out_full_channel_correlation_shrinkage",
        "normalization": "condition_weights_sum_one_no_rank_divisor",
    }
    analysis_fingerprint = fingerprint(config)
    reset = existing is not None and existing["fingerprint"] != analysis_fingerprint
    if reset and recompute != "all":
        raise ValueError(
            "Existing comparison uses incompatible settings; use another file or recompute='all'."
        )
    initialize_store(
        path,
        mveeg_version=_version(),
        seed=seed,
        config=config,
        fingerprint=analysis_fingerprint,
        reset=reset,
    )
    saved = subject_state(path)
    if recompute != "never":
        for subject in pipeline._excluded_subjects:
            if subject in saved:
                delete_subject(path, subject, subject_tables=SUBJECT_TABLES)
    included = [s for s in pipeline.subject_indices if s not in pipeline._excluded_subjects]
    if not included:
        warnings.warn("All subjects are excluded; no subjects to compare.", stacklevel=2)
        return None
    for subject in tqdm(included, desc="Subspace comparison", disable=not progress, unit="subject"):
        input_fingerprint = _input_fingerprint(pipeline, subject)
        status, previous = saved.get(subject, ("missing", None))
        changed = input_fingerprint != previous
        if status == "complete" and recompute != "all" and not (recompute == "changed" and changed):
            if changed:
                warnings.warn(
                    f"Input changed for subject {subject}, "
                    "but recompute='never' reuses the result.",
                    stacklevel=2,
                )
            continue
        mark_pending(path, subject, input_fingerprint, subject_tables=SUBJECT_TABLES)
        try:
            prepared = _prepare_subject(
                pipeline,
                subject=subject,
                target=target,
                conditions=condition_map,
                expression=condition_map,
            )
            metadata = prepared["metadata"]
            if groups is not None and groups not in metadata:
                raise ValueError(f"Independent-group column {groups!r} is missing.")
            tables = compare_subject(
                subject=subject,
                data=prepared["data"],
                metadata=metadata,
                labels=prepared["training_labels"],
                trial_ids=prepared["trials"]["epoch_index"].to_numpy(),
                times=prepared["time_bins"]["time"].to_numpy(),
                conditions=list(condition_map),
                models=models,
                nuisance=nuisance,
                weights=normalized_weights,
                groups=None if groups is None else metadata[groups].to_numpy(),
                folds=pipeline._cv["folds"],
                seed=seed,
            )
            tables["trials"] = prepared["trials"]
            with duckdb.connect(str(path)) as connection:
                connection.execute("BEGIN TRANSACTION")
                try:
                    for name in ("channels", "time_bins"):
                        if table_exists(connection, name):
                            require_equal_frame(connection, name, prepared[name])
                        else:
                            insert_frame(connection, name, prepared[name])
                    for name, frame in tables.items():
                        insert_frame(connection, name, frame)
                    mark_complete(connection, subject, input_fingerprint)
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
        except Exception as error:
            mark_failed(path, subject, input_fingerprint, str(error))
            raise RuntimeError(
                f"Subspace comparison failed for subject {subject}: {error}"
            ) from error
    return None
