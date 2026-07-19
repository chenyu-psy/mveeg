"""Focused tests for canonical artifact sidecars and review sessions."""

import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from mveeg.prep.artifacts import (
    ARTIFACT_COLUMNS,
    build_artifact_table,
    project_artifact_summary,
    read_artifact_table,
    validate_artifact_table,
    write_artifact_table,
)
from mveeg.prep.review import ReviewSession


def _all_review_table(n_epochs: int = 5) -> pd.DataFrame:
    """Build a small sidecar whose epochs all need initial review."""
    return build_artifact_table(
        "4001",
        range(n_epochs),
        ["Cz"],
        epoch_review=np.ones(n_epochs, dtype=bool),
        epoch_reasons={"large_p2p": np.ones(n_epochs, dtype=bool)},
    )


def test_ignored_channels_remain_labeled_but_do_not_aggregate():
    """Ignored channel cells should not change trial status or summary reasons."""
    rejected = {
        "large_p2p": np.array(
            [
                [False, True],
                [True, False],
            ]
        )
    }

    table = build_artifact_table(
        "4001",
        [0, 1],
        ["Cz", "VEOG"],
        rejected_reasons=rejected,
        ignore_channels=["VEOG"],
    )

    assert table["initial_status"].tolist() == ["accepted", "rejected"]
    assert pd.isna(table.loc[0, "epoch_reasons"])
    assert table.loc[0, "channel_VEOG"] == "large_p2p"
    assert table.loc[1, "epoch_reasons"] == "large_p2p"


def test_explicit_epoch_masks_are_authoritative_over_channel_flags():
    """Aggregated threshold masks must prevent one channel from deciding a trial."""
    rejected = {"large_p2p": np.array([[True, False], [False, False]])}
    review = {"large_step": np.array([[False, False], [True, False]])}

    table = build_artifact_table(
        "4001",
        [0, 1],
        ["Cz", "Pz"],
        rejected_reasons=rejected,
        review_reasons=review,
        epoch_rejected=np.array([False, True]),
        epoch_review=np.array([False, False]),
    )

    assert table["initial_status"].tolist() == ["accepted", "rejected"]
    assert table["epoch_reasons"].tolist() == ["large_p2p", "large_step"]


def test_relabeling_matches_keys_and_preserves_only_reviewed_final_status():
    """Reruns should never preserve an unreviewed or position-matched decision."""
    previous = _all_review_table(3)
    previous.loc[0, ["final_status", "reviewed"]] = ["accepted", True]

    relabeled = build_artifact_table(
        "4001",
        [2, 0, 1],
        ["Cz"],
        epoch_rejected=np.array([True, True, True]),
        previous=previous,
    )

    by_epoch = relabeled.set_index("epoch_index")
    assert by_epoch.loc[0, "final_status"] == "accepted"
    assert bool(by_epoch.loc[0, "reviewed"])
    assert by_epoch.loc[1, "final_status"] == "rejected"
    assert by_epoch.loc[2, "final_status"] == "rejected"


def test_relabeling_rejects_missing_previous_keys():
    """Previous decisions cannot be aligned by position or partial key overlap."""
    previous = _all_review_table(2)

    with pytest.raises(ValueError, match="must match exactly"):
        build_artifact_table(
            "4001",
            [0, 2],
            ["Cz"],
            epoch_review=np.array([True, True]),
            previous=previous,
        )


def test_artifact_builder_rejects_missing_subject_identity():
    """A missing subject must not be stringified into a seemingly valid key."""
    with pytest.raises(ValueError, match="subject_index cannot be missing"):
        build_artifact_table(None, [0], ["Cz"])


def test_artifact_validation_rejects_unreviewed_manual_change_and_bad_reason():
    """The sidecar boundary should protect status and reason invariants."""
    table = _all_review_table(1)
    table.loc[0, "final_status"] = "accepted"
    with pytest.raises(ValueError, match="Unreviewed epochs"):
        validate_artifact_table(table)

    table = _all_review_table(1)
    table.loc[0, "channel_Cz"] = "Large P2P"
    with pytest.raises(ValueError, match="snake_case"):
        validate_artifact_table(table)


def test_artifact_io_preserves_subject_text_and_summary_boundary(tmp_path):
    """TSV I/O should keep leading zeros and exclude review/channel detail downstream."""
    table = build_artifact_table(
        "004001",
        [0],
        ["Cz"],
        review_reasons={"large_step": np.array([[True]])},
    )
    path = tmp_path / "artifacts.tsv"

    write_artifact_table(table, path)
    loaded = read_artifact_table(path)
    summary = project_artifact_summary(loaded)

    assert loaded.loc[0, "subject_index"] == "004001"
    assert summary.columns.tolist() == [
        "subject_index",
        "epoch_index",
        "initial_status",
        "final_status",
        "epoch_reasons",
    ]


def test_review_group_contract_and_metadata_grouping():
    """Review groups should come from fixed fields or strictly keyed metadata."""
    table = _all_review_table(3)
    metadata = pd.DataFrame(
        {
            "subject_index": ["4001", "4001", "4001"],
            "epoch_index": [2, 0, 1],
            "condition": ["other", "target", "target"],
        }
    )

    session = ReviewSession(
        table,
        subject_index="4001",
        metadata=metadata,
        group_by="condition",
        label="target",
    )

    assert session.target_epoch_indices == (0, 1)
    with pytest.raises(ValueError, match="supplied together"):
        ReviewSession(table, subject_index="4001", group_by="initial_status")


def test_review_commit_marks_only_visited_union_and_saves_staged_edits():
    """Displayed review rows default rejected and only visited rows are saved."""
    table = _all_review_table(5)
    table.loc[0, "reviewed"] = True
    saved = []
    session = ReviewSession(
        table,
        subject_index="4001",
        group_by="initial_status",
        label="review",
        window_size=2,
        on_commit=saved.append,
    )

    assert session.start_index == 1
    first_window = session.mark_displayed(window_start=session.start_index)
    assert first_window["epoch_index"].tolist() == [1, 2]
    assert first_window["final_status"].tolist() == ["rejected", "rejected"]
    assert session.working_table().loc[3, "final_status"] == "review"
    session.mark_displayed(indices=[0])
    session.set_status(2, "accepted")
    committed = session.commit()

    assert session.visited_epoch_indices == (0, 1, 2)
    assert committed["reviewed"].tolist() == [True, True, True, False, False]
    assert committed.loc[1, "final_status"] == "rejected"
    assert committed.loc[2, "final_status"] == "accepted"
    assert committed.loc[3, "final_status"] == "review"
    assert len(saved) == 1


def test_review_close_discards_pending_edits_without_writing():
    """Closing without commit must not invoke the writer or mutate the table."""
    saved = []
    session = ReviewSession(
        _all_review_table(2),
        subject_index="4001",
        on_commit=saved.append,
    )
    session.mark_displayed(window_start=0)
    session.set_status(0, "accepted")

    session.close()

    assert saved == []
    assert session.artifacts.loc[0, "final_status"] == "review"
    assert session.artifacts["reviewed"].tolist() == [False, False]


def test_review_path_commit_resumes_at_first_unreviewed_epoch(tmp_path):
    """A new session should resume after windows explicitly saved with ``w``."""
    path = tmp_path / "artifacts.tsv"
    write_artifact_table(_all_review_table(5), path)
    session = ReviewSession.from_path(
        path,
        subject_index="4001",
        group_by="initial_status",
        label="review",
        window_size=2,
    )
    session.mark_displayed(window_start=0)
    session.commit()

    resumed = ReviewSession.from_path(
        path,
        subject_index="4001",
        group_by="initial_status",
        label="review",
        window_size=2,
    )

    assert resumed.start_index == 2
    assert resumed.current_epoch_index == 2


def test_review_rejects_editing_an_epoch_that_was_not_displayed():
    """Manual status edits must be limited to epochs the user actually saw."""
    session = ReviewSession(_all_review_table(2), subject_index="4001")

    with pytest.raises(ValueError, match="displayed"):
        session.set_status(0, "accepted")


def _make_review_epochs(n_epochs=3):
    """Return a keyed two-channel EpochsArray for review GUI tests."""
    import mne

    info = mne.create_info(["Cz", "Pz"], 100, ["eeg", "eeg"])
    metadata = pd.DataFrame({"subject_index": ["4001"] * n_epochs, "epoch_index": range(n_epochs)})
    events = np.column_stack(
        (
            np.arange(n_epochs) * 30,
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        )
    )
    return mne.EpochsArray(
        np.zeros((n_epochs, 2, 20)),
        info,
        events=events,
        event_id={"stimulus": 1},
        tmin=-0.1,
        metadata=metadata,
        verbose=False,
    )


def test_matplotlib_review_visuals_and_direct_controls():
    """One integration check covers the lightweight GUI's visible contract."""
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgba

    from mveeg.prep.review import open_review_figure

    plt.switch_backend("Agg")
    epochs = _make_review_epochs()
    review_mask = np.zeros((3, 2), dtype=bool)
    review_mask[0, 0] = True
    review_mask[1, 1] = True
    artifacts = build_artifact_table(
        "4001",
        [0, 1, 2],
        ["Cz", "Pz"],
        review_reasons={"large_step": review_mask},
        epoch_review=np.array([True, False, False]),
    )
    session = ReviewSession(artifacts, subject_index="4001", window_size=2)

    browser = open_review_figure(session, epochs, show=False)

    assert session.visited_epoch_indices == (0, 1)
    assert browser.axes.get_position().bounds == pytest.approx((0.08, 0.10, 0.915, 0.75))
    assert browser.axes.get_xlim() == pytest.approx((0, 2 * browser._samples_per_epoch))
    assert session.get_status(0) == "rejected"
    assert [tick.get_text() for tick in browser.axes.get_yticklabels()] == ["Cz", "Pz"]
    assert not hasattr(browser, "_selected_epoch")
    assert all("*" not in tick.get_text() for tick in browser.axes.get_xticklabels())
    black = [line for line in browser.axes.lines if to_hex(line.get_color()) == "#000000"]
    red = [line for line in browser.axes.lines if to_hex(line.get_color()) == "#ff0000"]
    traces = [line for line in black if line.get_linewidth() == 0.75]
    boundaries = [line for line in black if line.get_linewidth() == 3]
    assert len(traces) == 2
    assert len(boundaries) == 1 and boundaries[0].get_linestyle() == "-"
    assert len(red) == 1 and red[0].get_linewidth() == 1.0
    assert any(
        np.allclose(patch.get_facecolor(), to_rgba("#edb74a", alpha=0.4))
        for patch in browser.axes.patches
    )
    assert any(text.get_text() == "Progress 2 / 3" for text in browser.axes.texts)
    trial_labels = [text for text in browser.axes.texts if text.get_text().startswith("Trial ")]
    assert [text.get_text() for text in trial_labels] == [
        "Trial 0\nstimulus\nrejected",
        "Trial 1\nstimulus\naccepted",
    ]
    assert not browser.axes.get_xticks().size
    default_key_handler = browser.figure.canvas.manager.key_press_handler_id
    key_callbacks = browser.figure.canvas.callbacks.callbacks.get("key_press_event", {})
    assert default_key_handler not in key_callbacks

    first_center = browser._samples_per_epoch / 2
    browser._on_click(SimpleNamespace(inaxes=browser.axes, xdata=first_center, button=1))
    assert session.get_status(0) == "accepted"
    assert any(text.get_text() == "Trial 0\nstimulus\naccepted" for text in browser.axes.texts)
    assert not browser.axes.patches
    assert not [line for line in browser.axes.lines if to_hex(line.get_color()) == "#ff0000"]

    browser._flag_checkbox.set_active(0)
    red = [line for line in browser.axes.lines if to_hex(line.get_color()) == "#ff0000"]
    assert len(red) == 2
    assert session.get_status(0) == "accepted"

    browser._on_key(SimpleNamespace(key="r"))
    reason_labels = [
        text
        for text in browser.axes.texts
        if text.get_text() == "large_step" and text.get_position()[1] == 0.98
    ]
    assert len(reason_labels) == 2
    assert all(
        text.get_position()[1] > 1
        for text in browser.axes.texts
        if text.get_text().startswith("Trial ")
    )
    assert [tick.get_text() for tick in browser.axes.get_yticklabels()] == ["Cz", "Pz"]
    browser._on_click(SimpleNamespace(inaxes=browser.axes, xdata=first_center, button=1))
    assert session.get_status(0) == "rejected"
    assert browser.axes.patches
    time_lock_lines = [line for line in browser.axes.lines if line.get_label() == "time-lock event"]
    assert len(time_lock_lines) == 1
    assert to_hex(time_lock_lines[0].get_color()) == "#ff00ff"
    assert time_lock_lines[0].get_linestyle() == "-"

    base_spacing = browser._base_spacing
    initial_ylim = browser.axes.get_ylim()
    epochs._data[2] = 1_000
    browser._on_key(SimpleNamespace(key="right"))
    assert browser._base_spacing == base_spacing
    assert browser.axes.get_ylim() == initial_ylim
    browser._on_key(SimpleNamespace(key="+"))
    assert browser._base_spacing == base_spacing
    assert browser.axes.get_ylim()[1] > initial_ylim[1]
    browser._on_key(SimpleNamespace(key="-"))
    assert np.allclose(browser.axes.get_ylim(), initial_ylim)
    plt.close(browser.figure)


def test_review_close_releases_preloaded_epochs_and_gui_resources():
    """Closing should make a retained browser lightweight and be idempotent."""
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import CloseEvent

    from mveeg.prep.review import open_review_figure

    plt.switch_backend("Agg")
    epochs = _make_review_epochs()
    session = ReviewSession(
        build_artifact_table("4001", range(3), ["Cz", "Pz"]),
        subject_index="4001",
    )
    browser = open_review_figure(session, epochs, show=False)
    figure = browser.figure
    epochs_ref = weakref.ref(epochs)
    session_ref = weakref.ref(session)
    figure_ref = weakref.ref(figure)

    figure.canvas.callbacks.process("close_event", CloseEvent("close_event", figure.canvas))
    browser._on_close(None)
    plt.close(figure)
    del epochs, session, figure
    gc.collect()

    assert browser.epochs is None
    assert browser.session is None
    assert browser.figure is None
    assert browser.axes is None
    assert browser._flag_checkbox is None
    assert browser._flag_callback_id is None
    assert browser._canvas_callback_ids == []
    assert epochs_ref() is None
    assert session_ref() is None
    assert figure_ref() is None


def test_review_uses_02_epoch_guides_and_top_trial_labels():
    """Five visible epochs use the 0.2 guide and annotation hierarchy."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from mveeg.prep.review import open_review_figure

    plt.switch_backend("Agg")
    epochs = _make_review_epochs(5)
    artifacts = build_artifact_table(
        "4001",
        range(5),
        ["Cz", "Pz"],
        epoch_rejected=np.zeros(5, dtype=bool),
        epoch_review=np.zeros(5, dtype=bool),
    )
    browser = open_review_figure(
        ReviewSession(artifacts, subject_index="4001", window_size=5),
        epochs,
        show=False,
    )

    boundaries = [
        line
        for line in browser.axes.lines
        if to_hex(line.get_color()) == "#000000" and line.get_linewidth() == 3
    ]
    time_lock_lines = [line for line in browser.axes.lines if to_hex(line.get_color()) == "#ff00ff"]
    trial_labels = [
        text.get_text() for text in browser.axes.texts if text.get_text().startswith("Trial ")
    ]
    assert len(boundaries) == 4
    assert len(time_lock_lines) == 5
    assert all(line.get_linestyle() == "-" for line in time_lock_lines)
    assert trial_labels == [f"Trial {epoch}\nstimulus\naccepted" for epoch in range(5)]
    assert not browser.axes.get_xticks().size
    plt.close(browser.figure)


def test_review_preserves_epoch_width_when_window_is_not_full():
    """A short window keeps five epoch slots and ignores clicks in empty slots."""
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from mveeg.prep.review import open_review_figure

    plt.switch_backend("Agg")
    epochs = _make_review_epochs(3)
    artifacts = build_artifact_table(
        "4001",
        range(3),
        ["Cz", "Pz"],
        epoch_rejected=np.zeros(3, dtype=bool),
        epoch_review=np.zeros(3, dtype=bool),
    )
    session = ReviewSession(artifacts, subject_index="4001", window_size=5)
    browser = open_review_figure(session, epochs, show=False)

    samples = browser._samples_per_epoch
    traces = [
        line
        for line in browser.axes.lines
        if to_hex(line.get_color()) == "#000000" and line.get_linewidth() == 0.75
    ]
    time_lock_lines = [
        line for line in browser.axes.lines if to_hex(line.get_color()) == "#ff00ff"
    ]
    trial_labels = [
        text for text in browser.axes.texts if text.get_text().startswith("Trial ")
    ]
    statuses = [session.get_status(epoch) for epoch in range(3)]

    assert browser.axes.get_xlim() == pytest.approx((0, 5 * samples))
    assert all(len(line.get_xdata()) == 3 * samples for line in traces)
    assert len(time_lock_lines) == 3
    assert len(trial_labels) == 3

    browser._on_click(
        SimpleNamespace(inaxes=browser.axes, xdata=3.5 * samples, button=1)
    )
    assert [session.get_status(epoch) for epoch in range(3)] == statuses
    plt.close(browser.figure)


def test_review_distinguishes_bad_and_interpolated_autoreject_channels():
    """Interpolated channels use a distinct blue dashed overlay."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from mveeg.prep.review import open_review_figure

    plt.switch_backend("Agg")
    epochs = _make_review_epochs()
    bad = np.zeros((3, 2), dtype=bool)
    interpolated = np.zeros((3, 2), dtype=bool)
    bad[0] = True
    interpolated[0, 1] = True
    artifacts = build_artifact_table(
        "4001",
        [0, 1, 2],
        ["Cz", "Pz"],
        rejected_reasons={
            "autoreject_bad_channel": bad,
            "autoreject_interpolated": interpolated,
        },
        epoch_rejected=np.array([True, False, False]),
    )
    browser = open_review_figure(
        ReviewSession(artifacts, subject_index="4001"),
        epochs,
        show=False,
    )

    overlays = [line for line in browser.axes.lines if line.get_linewidth() == 1.0]
    red = [line for line in overlays if to_hex(line.get_color()) == "#ff0000"]
    blue = [line for line in overlays if to_hex(line.get_color()) == "#0072b2"]
    assert len(red) == 1 and red[0].get_linestyle() == "-"
    assert len(blue) == 1 and blue[0].get_linestyle() == "--"
    plt.close(browser.figure)


@pytest.mark.parametrize(
    ("group_by", "label", "conditions", "expected"),
    [
        (
            "initial_status",
            "review",
            None,
            "initial_status:\n  accepted: 1\n  review: 1\n  rejected: 1\n",
        ),
        (
            "condition",
            "target",
            ["target", pd.NA, "target"],
            "condition:\n  target: 2\n  <NA>: 1\n",
        ),
        (None, None, None, "all: 3\n"),
    ],
)
def test_review_open_prints_all_group_counts(group_by, label, conditions, expected, capsys):
    """GUI opening prints one complete artifact, metadata, or all count block."""
    import matplotlib.pyplot as plt

    from mveeg.prep.review import open_review_figure

    plt.switch_backend("Agg")
    epochs = _make_review_epochs()
    artifacts = build_artifact_table(
        "4001",
        [0, 1, 2],
        ["Cz", "Pz"],
        epoch_rejected=np.array([False, False, True]),
        epoch_review=np.array([False, True, False]),
    )
    metadata = epochs.metadata.copy()
    if conditions is not None:
        metadata["condition"] = conditions
    session = ReviewSession(
        artifacts,
        subject_index="4001",
        metadata=metadata,
        group_by=group_by,
        label=label,
    )

    browser = open_review_figure(session, epochs, show=False)

    assert capsys.readouterr().out == expected
    plt.close(browser.figure)


def test_review_complete_state_is_explicit_after_commit():
    session = ReviewSession(_all_review_table(2), subject_index="4001", window_size=2)
    session.mark_displayed(window_start=0)

    session.commit()

    assert session.is_complete


def test_schema_has_no_unrequested_review_audit_fields():
    """The fixed schema should keep only the agreed review marker."""
    assert tuple(ARTIFACT_COLUMNS) == (
        "subject_index",
        "epoch_index",
        "initial_status",
        "final_status",
        "epoch_reasons",
        "reviewed",
    )
