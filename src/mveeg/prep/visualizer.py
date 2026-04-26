"""Interactive reviewer for preprocessed EEG epochs and trial QC.

This module loads one subject's saved derivative files and opens a lightweight
Matplotlib browser for trial-by-trial inspection. The code is written to keep
the data flow explicit for research scripts: resolve files, load QC tables,
prepare plotting arrays, then handle keyboard and mouse review actions.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Slider
import pandas as pd
from pathlib import Path

from ..io.bids import build_derivative_stem, build_subject_label, normalize_subject_id


# For plotting
SLIDER_STEP = 5
TRIAL_CATEGORY_CODES = {"accepted": 1, "unclear": 2, "rejected": 3}


EYETRACK_SCALE = 1e-6  # default downscale factor for eyetracking channels
CHAN_OFFSET = 0.00005


def summarize_manual_qc_status(
    parent_dir: str | Path,
    *,
    experiment_name: str,
    subject_prefix: str = "sub",
    derivative_dirname: str = "derivatives",
    derivative_label: str = "preprocessed",
    datatype: str = "eeg",
) -> pd.DataFrame:
    """Summarize which subjects already have saved manual QC decisions.

    Parameters
    ----------
    parent_dir : str | Path
        Root BIDS directory that contains the derivatives folder.
    experiment_name : str
        Task label used in derivative filenames.
    subject_prefix : str, optional
        Subject prefix used in saved folders and filenames.
    derivative_dirname : str, optional
        Name of the folder that stores saved derivatives.
    derivative_label : str, optional
        Label written after ``desc-`` in derivative filenames.
    datatype : str, optional
        Datatype folder that contains the saved trial QC tables.

    Returns
    -------
    pd.DataFrame
        One row per subject with columns that describe whether manual QC has
        already been saved.
    """
    parent_dir = Path(parent_dir)
    derivative_root = parent_dir / derivative_dirname
    subject_rows = []

    for subject_dir in sorted(derivative_root.glob(f"{subject_prefix}-*")):
        if not subject_dir.is_dir():
            continue

        subject_label = subject_dir.name
        subject_id = normalize_subject_id(subject_label, subject_prefix=subject_prefix)
        stem = build_derivative_stem(
            subject_id,
            experiment_name=experiment_name,
            subject_prefix=subject_prefix,
            derivative_label=derivative_label,
        )
        trial_qc_path = subject_dir / datatype / f"{stem}_trial_qc.tsv"
        if not trial_qc_path.exists():
            subject_rows.append(
                {
                    "subject": subject_label,
                    "manual_qc_status": "missing_trial_qc",
                    "manual_qc_done": False,
                }
            )
            continue

        trial_qc = pd.read_csv(trial_qc_path, sep="\t", keep_default_na=False)
        manual_qc_done = "final_qc_category" in trial_qc.columns
        subject_rows.append(
            {
                "subject": subject_label,
                "manual_qc_status": "reviewed" if manual_qc_done else "not_reviewed",
                "manual_qc_done": manual_qc_done,
            }
        )

    return pd.DataFrame(subject_rows)


def print_manual_qc_status(
    parent_dir: str | Path,
    *,
    experiment_name: str,
    subject_prefix: str = "sub",
    derivative_dirname: str = "derivatives",
    derivative_label: str = "preprocessed",
    datatype: str = "eeg",
) -> pd.DataFrame:
    """Print a readable summary of manual QC completion across subjects.

    Parameters
    ----------
    parent_dir : str | Path
        Root BIDS directory that contains the derivatives folder.
    experiment_name : str
        Task label used in derivative filenames.
    subject_prefix : str, optional
        Subject prefix used in saved folders and filenames.
    derivative_dirname : str, optional
        Name of the folder that stores saved derivatives.
    derivative_label : str, optional
        Label written after ``desc-`` in derivative filenames.
    datatype : str, optional
        Datatype folder that contains the saved trial QC tables.

    Returns
    -------
    pd.DataFrame
        The same status table returned by ``summarize_manual_qc_status``.
    """
    status_table = summarize_manual_qc_status(
        parent_dir,
        experiment_name=experiment_name,
        subject_prefix=subject_prefix,
        derivative_dirname=derivative_dirname,
        derivative_label=derivative_label,
        datatype=datatype,
    )

    if len(status_table) == 0:
        print("No preprocessed subjects were found in derivatives.")
        return status_table

    reviewed_subjects = status_table.loc[status_table["manual_qc_done"], "subject"].tolist()
    pending_subjects = status_table.loc[~status_table["manual_qc_done"], "subject"].tolist()

    print("Manual visualization QC status:")
    print(f"  Reviewed: {len(reviewed_subjects)}")
    print(f"  Pending: {len(pending_subjects)}")

    if len(pending_subjects) > 0:
        print("  Subjects still needing manual QC:")
        for subject in pending_subjects:
            print(f"    {subject}")

    return status_table


def build_manual_qc_visualizer(
    subject: str,
    *,
    parent_dir: str | Path,
    experiment_name: str,
    subject_prefix: str = "sub",
    srate: float | None = None,
    rejection_time: list[float | None] | None = None,
    downscale: dict[str, float] | None = None,
    channels_drop: list[str] | tuple[str, ...] | None = None,
    channels_ignore: list[str] | tuple[str, ...] | None = None,
    trial_category: str | int | None = None,
) -> "Visualizer":
    """Build a Visualizer with explicit manual-QC plotting settings.

    Parameters
    ----------
    subject : str
        Subject entered by the user.
    parent_dir : str | Path
        Root BIDS directory that contains the derivatives folder.
    experiment_name : str
        Task label used in derivative filenames.
    subject_prefix : str, optional
        Subject prefix used in saved folders and filenames.
    srate : float | None, optional
        Sampling rate for plotting. If ``None``, use the saved epochs value.
    rejection_time : list[float | None] | None, optional
        Start and end of the artifact rejection window in seconds.
    downscale : dict[str, float] | None, optional
        Optional per-channel-type plotting scale factors.
    channels_drop : list[str] | tuple[str, ...] | None, optional
        Channels dropped from the browser before plotting.
    channels_ignore : list[str] | tuple[str, ...] | None, optional
        Channels shown in the browser but excluded from manual rejection flags.
    trial_category : str | int | None, optional
        Optional QC group filter shown in the browser.

    Returns
    -------
    Visualizer
        Configured visualizer ready for preprocessing and plotting.
    """
    return Visualizer(
        subject,
        parent_dir=parent_dir,
        experiment_name=experiment_name,
        subject_prefix=subject_prefix,
        srate=srate,
        rejection_time=rejection_time,
        downscale=downscale or {"eyegaze": 1e-6, "misc": 1e-4, "eeg": 1, "eog": 1},
        channels_drop=channels_drop,
        channels_ignore=channels_ignore,
        trial_category=trial_category,
    )


def print_flagged_channels(viz: "Visualizer", *, min_rejections: int = 10) -> None:
    """Print channels that were flagged often enough to deserve quick review.

    Parameters
    ----------
    viz : Visualizer
        Loaded visualizer object for one subject.
    min_rejections : int, optional
        Minimum number of flagged trials required before a channel is printed.

    Returns
    -------
    None
        The function prints one line per frequently flagged channel.
    """
    rejection_sums = viz.rej_chans.sum(axis=0)
    sort_ix = np.argsort(rejection_sums)[::-1]

    print(f"Channels flagged in at least {min_rejections} trials:")
    found_channel = False
    for chan_ix in sort_ix:
        if rejection_sums[chan_ix] <= min_rejections:
            continue
        print(f"  {viz.chan_labels[chan_ix]}: {int(rejection_sums[chan_ix])}")
        found_channel = True

    if not found_channel:
        print("  None")


def run_manual_qc_session(
    *,
    parent_dir: str | Path,
    experiment_name: str,
    subject_prefix: str = "sub",
    srate: float | None = None,
    rejection_time: list[float | None] | tuple[float | None, float | None] | None = None,
    min_rejections: int = 10,
    input_subject: str | None = None,
    input_trial_category: str | int | None = None,
) -> "Visualizer":
    """Run one manual-visualization QC session with standard script prompts.

    Parameters
    ----------
    parent_dir : str | Path
        Root BIDS directory that contains the derivatives folder.
    experiment_name : str
        Task label used in derivative filenames.
    subject_prefix : str, optional
        Subject prefix used in saved folders and filenames.
    srate : float | None, optional
        Sampling rate for plotting. If ``None``, use the saved epochs value.
    rejection_time : list[float | None] | tuple[float | None, float | None] | None, optional
        Start and end of the artifact-rejection window in seconds.
    min_rejections : int, optional
        Minimum channel rejection count shown by ``print_flagged_channels``.
    input_subject : str | None, optional
        Subject ID to open directly. If ``None``, prompt in the script.
    input_trial_category : str | int | None, optional
        Trial category filter. If ``None``, prompt in the script and allow
        blank input for all trials.

    Returns
    -------
    Visualizer
        Opened visualizer instance for interactive manual QC.

    Example
    -------
    run_manual_qc_session(
        parent_dir=preprocess_flow.data_dir,
        experiment_name=preprocess_flow.experiment_name,
        subject_prefix=preprocess_flow.subject_prefix,
        srate=500,
        rejection_time=(-0.25, 3.0),
    )
    """
    print_manual_qc_status(
        parent_dir,
        experiment_name=experiment_name,
        subject_prefix=subject_prefix,
    )

    subject = input_subject
    if subject is None:
        subject = input("Enter subject number: ").strip()

    trial_category = input_trial_category
    if trial_category is None:
        trial_category_raw = input(
            'Enter trial category ("accepted"/1, "unclear"/2, "rejected"/3, or leave blank for all): '
        ).strip()
        if trial_category_raw == "":
            trial_category = None
        elif trial_category_raw.isdigit():
            trial_category = int(trial_category_raw)
        else:
            trial_category = trial_category_raw

    viz = build_manual_qc_visualizer(
        subject,
        parent_dir=parent_dir,
        experiment_name=experiment_name,
        subject_prefix=subject_prefix,
        srate=srate,
        rejection_time=list(rejection_time) if rejection_time is not None else None,
        trial_category=trial_category,
    )

    print_flagged_channels(viz, min_rejections=min_rejections)
    viz.preprocess_data_for_plot()
    viz.open_figure()
    return viz


class Visualizer:
    def __init__(
        self,
        sub,
        parent_dir: str,
        experiment_name: str,
        subject_prefix: str = "sub",
        derivative_dirname: str = "derivatives",
        derivative_label: str = "preprocessed",
        datatype: str = "eeg",
        srate: float | None = None,
        rejection_time: list[float | None] | None = None,
        win_step: int = SLIDER_STEP,
        downscale: dict | None = None,
        chan_offset: float = CHAN_OFFSET,
        channels_drop: list | None = None,
        channels_ignore: list | None = None,
        load_flags: bool = True,
        trial_category: str | None = None,
        port_codes_show: list | None = None,  # list of port codes to show, if None then all are shown
    ):
        """Load one subject's preprocessed epochs and QC labels for manual review.

        Parameters
        ----------
        sub : str
            Subject label entered by the user, for example ``"1001"``,
        parent_dir : str
            Root BIDS directory that contains the ``derivatives`` folder.
        experiment_name : str
            Task label used in BIDS filenames.
        subject_prefix : str, optional
            Subject prefix used in saved folders and filenames, for example
            ``"sub"`` for BIDS labels like ``"sub-1001"``.
        derivative_dirname : str, optional
            Name of the folder that stores saved derivatives.
        derivative_label : str, optional
            Label written after ``desc-`` in derivative filenames.
        datatype : str, optional
            Datatype folder that contains the saved epochs and sidecars.
        srate : float | None, optional
            Sampling rate for plotting. If ``None``, use the epochs metadata.
        rejection_time : list[float | None] | None, optional
            Start and end of the artifact rejection window in seconds.
        win_step : int, optional
            Number of trials shown per view.
        downscale : dict | None, optional
            Per-channel-type scale factors used during plotting.
        chan_offset : float, optional
            Vertical offset between channels in the plot.
        channels_drop : list | None, optional
            Channels to drop before plotting and label display.
        channels_ignore : list | None, optional
            Channels that should never be marked as manually rejected.
        load_flags : bool, optional
            If ``True``, load previously saved manual rejection flags.
        trial_category : str | int | None, optional
            If provided, keep only trials from one QC group:
            ``"accepted"`` or ``1``, ``"unclear"`` or ``2``, and
            ``"rejected"`` or ``3``.
        port_codes_show : list | None, optional
            Event labels or codes to display in the browser.

        Returns
        -------
        None
            The initializer prepares data arrays used by the plotting methods.
        """

        self.subject_prefix = subject_prefix
        self.sub = self._normalize_subject_id(sub)
        self.parent_dir = parent_dir
        self.experiment_name = experiment_name
        self.derivative_dirname = derivative_dirname
        self.derivative_label = derivative_label
        self.datatype = datatype
        self.win_step = win_step
        self.trial_category_filter = trial_category

        self.rejection_time = [None, None] if rejection_time is None else list(rejection_time)
        self.downscale = self._normalize_downscale(downscale)
        self.chan_offset = chan_offset

        self.epochs_fpath = self._resolve_epochs_file()
        self.epochs_obj = mne.read_epochs(self.epochs_fpath)

        self.srate = self.epochs_obj.info["sfreq"] if srate is None else srate

        # get trial start and end time from epochs object
        self.trial_start = self.epochs_obj.times[0]
        self.trial_end = self.epochs_obj.times[-1]
        self.epoch_len = len(self.epochs_obj.times)
        self.rejection_time[0] = self.trial_start if self.rejection_time[0] is None else self.rejection_time[0]
        self.rejection_time[1] = self.trial_end if self.rejection_time[1] is None else self.rejection_time[1]

        self.events = pd.read_csv(self._resolve_related_file("events", ".tsv"), sep="\t")

        events_show = self._resolve_events_to_show(port_codes_show)
        self.all_port_codes, self.all_portcode_times = self._build_port_code_lists(events_show)

        self.rej_chans, self.rej_reasons = self._load_artifact_labels()
        self.trial_qc = self._load_trial_qc()
        self.trial_indices = self._load_trial_indices()

        if channels_drop is not None:  # drop ignored channels
            channels_drop = [ch for ch in channels_drop if ch in self.epochs_obj.ch_names]
            if len(channels_drop) > 0:
                keep_chans = ~np.isin(self.epochs_obj.ch_names, channels_drop)
                self.rej_chans = self.rej_chans[:, keep_chans]
                self.rej_reasons = self.rej_reasons[:, keep_chans]
                self.epochs_obj.drop_channels(channels_drop)

        self.info = self.epochs_obj.info
        self.chan_types = np.array(self.info.get_channel_types())
        self.chan_labels = np.array(self.epochs_obj.ch_names)

        self.channels_ignore = channels_ignore  # make a mask for channels we ignore
        ignore = [] if channels_ignore is None else channels_ignore
        self.ignored_channels_mask = np.isin(self.chan_labels, ignore)

        if self.rej_chans.shape[1] != self.ignored_channels_mask.shape[0]:
            raise ValueError(
                f"There are {self.rej_chans.shape[1]} channels in the rejection labels,\
                             but {self.ignored_channels_mask.shape[0]} channels in the data. \
                             Please make sure that any channels without artifact labels are dropped"
            )

        if channels_ignore is not None:  # never reject ignored channels (eg EOG)
            self.rej_chans[:, self.ignored_channels_mask] = False
            self.rej_reasons[:, self.ignored_channels_mask] = None

        self.rej_manual_full = np.array(self._load_manual_flags(load_flags), dtype=bool, copy=True)
        self.rej_manual = np.array(self.rej_manual_full, dtype=bool, copy=True)

        if self.trial_category_filter is not None:
            self._apply_trial_subset(self.trial_category_filter)

        self.win_step = min(self.win_step, len(self.epochs_obj))

        self.epochs_raw = self.epochs_obj.get_data(copy=True)
        self.epochs_pre = None  # initialized when we preprocess

        self.offset_dict = None
        self.xlim = (0, self.epoch_len * self.win_step)

        self.ylim = [None, None]
        self.stack = False

        self.pos = 0
        self.extra_chan_scale = 1

        self.rej_reasons_on = False
        self.port_codes_on = False

    def _normalize_downscale(self, downscale: dict | None) -> dict[str, float]:
        """Fill in plotting scale factors for each supported channel type.

        Parameters
        ----------
        downscale : dict | None
            Optional user-provided scale factors keyed by channel type.

        Returns
        -------
        dict[str, float]
            Complete mapping for EEG, EOG, eyegaze, pupil, and misc channels.

        Notes
        -----
        EEG is kept in Volts, while eyetracking channels are strongly scaled
        down so they can be shown in the same browser window.
        """
        default_downscale = {
            "eeg": 1.0,
            "eog": 1.0,
            "eyegaze": EYETRACK_SCALE,
            "pupil": 1.0,
            "misc": 1.0,
        }
        if downscale is None:
            return default_downscale

        merged = default_downscale.copy()
        merged.update(downscale)
        return merged

    def _subject_eeg_dir(self) -> Path:
        """Return the derivative EEG directory for the requested subject."""
        return Path(self.parent_dir) / self.derivative_dirname / self._subject_label() / self.datatype

    def _file_stem(self) -> str:
        """Return the shared filename stem for saved derivative EEG files."""
        return build_derivative_stem(
            self.sub,
            self.experiment_name,
            subject_prefix=self.subject_prefix,
            derivative_label=self.derivative_label,
        )

    def _subject_label(self) -> str:
        """Build the standard BIDS subject label for saved derivatives.

        Parameters
        ----------
        None
            The method uses the normalized subject ID stored on the instance.

        Returns
        -------
        str
            Subject label used in derivative folders and filenames, for example
            ``"sub-1001"``.
        """
        return build_subject_label(self.sub, subject_prefix=self.subject_prefix)

    def _build_sidecar_path(self, suffix: str, extension: str) -> Path:
        """Build a sidecar path next to the loaded epochs file.

        Parameters
        ----------
        suffix : str
            New filename suffix, for example ``"events"``.
        extension : str
            New filename extension including the leading dot.

        Returns
        -------
        Path
            Sidecar path that shares the epochs filename prefix.
        """
        return self._subject_eeg_dir() / f"{self._file_stem()}_{suffix}{extension}"

    def _resolve_epochs_file(self) -> Path:
        """Return the saved epochs file for a subject.

        Parameters
        ----------
        None
            The method uses the subject/task values already stored on the
            instance.

        Returns
        -------
        Path
            Full path to the epochs FIF file.

        Raises
        ------
        FileNotFoundError
            If no saved epochs file can be found.

        """
        epochs_path = self._subject_eeg_dir() / f"{self._file_stem()}_epo.fif"
        if epochs_path.exists():
            return epochs_path

        raise FileNotFoundError(
            "Could not find any saved epochs FIF file for "
            f"subject {self.sub}. Expected a file like "
                f'"{self._subject_label()}_{self.experiment_name}_desc-preprocessed_epo.fif". '
            "Run the preprocessing save step first "
            '(`pre.save_all_data(subject_number, epochs, rej_reasons)`). '
            f"Searched {self._subject_eeg_dir()}."
        )

    def _resolve_related_file(self, suffix: str, extension: str, required: bool = True) -> Path | None:
        """Return a saved sidecar file next to the epochs file.

        Parameters
        ----------
        suffix : str
            Filename suffix to search for, for example ``"events"``.
        extension : str
            Filename extension including the leading dot.
        required : bool, optional
            If ``True``, raise an error when no file is found.

        Returns
        -------
        Path | None
            Matching file path, or ``None`` when the file is optional and no
            match is found.
        """
        preferred_path = self._build_sidecar_path(suffix, extension)
        if preferred_path.exists():
            return preferred_path

        if required:
            raise FileNotFoundError(
                f"Could not find {preferred_path}."
            )
        return None

    def _normalize_subject_id(self, sub: str) -> str:
        """Convert subject input into the BIDS subject identifier.

        Parameters
        ----------
        sub : str
            Subject label entered by the user, for example ``"1001"``,
            ``"sub1001"``, or ``"sub-1001"``.

        Returns
        -------
        str
            Normalized subject identifier without the leading ``sub`` prefix.
        """
        return normalize_subject_id(sub, subject_prefix=self.subject_prefix)

    def _resolve_events_to_show(self, port_codes_show: list | None) -> list:
        """Resolve requested event labels/codes to canonical event labels."""
        event_id = self.epochs_obj.event_id
        if port_codes_show is None:
            return list(event_id.keys())

        resolved = []
        for code in port_codes_show:
            if code in event_id:
                resolved.append(code)
            elif code in event_id.values():
                label = list(event_id.keys())[list(event_id.values()).index(code)]
                resolved.append(label)
        return resolved

    def _build_port_code_lists(self, events_show: list) -> tuple[list, list]:
        """Build per-trial lists of numeric event codes and sample indices."""
        all_port_codes = []
        all_portcode_times = []
        for _, row in self.epochs_obj.metadata.iterrows():
            keep_events = np.intersect1d(events_show, row.index)
            row = row[keep_events].dropna()
            row = row[np.logical_and(row > self.trial_start, row < self.trial_end)]
            row.sort_values(inplace=True)
            all_port_codes.append([self.epochs_obj.event_id[ev] for ev in row.index])
            all_portcode_times.append(
                np.rint((row.to_numpy().astype(float) - self.trial_start) * self.srate).astype(int).tolist()
            )
        return all_port_codes, all_portcode_times

    def _load_artifact_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Load artifact labels and convert non-empty cells to channel flags.

        Parameters
        ----------
        None
            The method uses the subject/task values already stored on the
            instance.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Boolean rejection mask and string rejection labels, both shaped as
            ``(n_trials, n_channels)``.
        """
        rej = pd.read_csv(self._resolve_related_file("artifacts", ".tsv"), sep="\t", keep_default_na=False)
        rej_chans = rej.ne("").to_numpy()
        rej_reasons = rej.to_numpy()
        return rej_chans, rej_reasons

    def _load_trial_qc(self) -> pd.DataFrame | None:
        """Load the saved trial-level QC table when it is available.

        Parameters
        ----------
        None
            The method uses the subject/task values already stored on the
            instance.

        Returns
        -------
        pd.DataFrame | None
            Trial-level QC table aligned with the epochs order, or ``None``
            when the file is not available.
        """
        trial_qc_path = self._resolve_related_file("trial_qc", ".tsv", required=False)
        if trial_qc_path is None:
            return None
        trial_qc = pd.read_csv(trial_qc_path, sep="\t", keep_default_na=False)
        if "trial_qc_category" in trial_qc.columns:
            trial_qc["trial_qc_category"] = trial_qc["trial_qc_category"].replace(
                {"needs_manual_review": "unclear"}
            )
        if "final_qc_category" in trial_qc.columns:
            trial_qc["final_qc_category"] = trial_qc["final_qc_category"].replace(
                {"needs_manual_review": "unclear"}
            )
        return trial_qc

    def _load_trial_indices(self) -> np.ndarray:
        """Return the original trial indices for the loaded epochs table.

        Parameters
        ----------
        None
            The method uses the loaded trial QC table when available.

        Returns
        -------
        np.ndarray
            Integer vector with one original trial index per currently loaded
            epoch.
        """
        if self.trial_qc is not None and "trial_index" in self.trial_qc.columns:
            return self.trial_qc["trial_index"].to_numpy(dtype=int)
        return np.arange(len(self.epochs_obj), dtype=int)

    def _default_manual_flags(self) -> np.ndarray:
        """Build the initial manual rejection vector for the current subject.

        Parameters
        ----------
        None
            The method uses already-loaded QC labels on the instance.

        Returns
        -------
        np.ndarray
            Boolean vector with one value per trial.

        Notes
        -----
        Before manual review, both ``"rejected"`` and ``"unclear"`` trials are
        treated as rejected by default. This keeps borderline trials out of the
        accepted set until the reviewer explicitly keeps them.
        """
        if self.trial_qc is not None and "trial_qc_category" in self.trial_qc.columns:
            return np.array(
                self.trial_qc["trial_qc_category"].isin(["rejected", "unclear"]).to_numpy(),
                dtype=bool,
                copy=True,
            )
        return np.array(self.rej_chans.any(1), dtype=bool, copy=True)

    def _load_manual_flags(self, load_flags: bool) -> np.ndarray:
        """Load saved manual flags when they match the full trial set.

        Parameters
        ----------
        load_flags : bool
            If ``True``, try to load a previously saved flag file.

        Returns
        -------
        np.ndarray
            Boolean vector with one flag per trial in the full loaded dataset.

        Notes
        -----
        Older saved flag files may contain only a filtered subset of trials.
        Those files cannot be aligned safely to the full epoch list, so this
        method falls back to the default flags instead of guessing.
        """
        default_flags = self._default_manual_flags()
        if not load_flags:
            return default_flags

        flags_fpath = self._resolve_related_file("rejection_flags", ".npy", required=False)
        if flags_fpath is None:
            print("No saved annotations found, resetting to default.")
            return default_flags

        loaded_flags = np.load(flags_fpath)
        if len(loaded_flags) != len(default_flags):
            print(
                "Saved annotations do not match the current full trial count. "
                "Resetting to default flags instead."
            )
            return default_flags

        print("You have saved annotations already. Loading these.")
        return np.array(loaded_flags, dtype=bool, copy=True)

    def _apply_trial_subset(self, trial_category: str) -> None:
        """Keep only trials from one saved QC category.

        Parameters
        ----------
        trial_category : str | int
            Category name or code to keep. Must match the saved
            ``trial_qc_category`` or ``trial_qc_code`` column.

        Returns
        -------
        None
            The method subsets the epochs, events, QC labels, and manual flags
            in place.
        """
        if self.trial_qc is None or "trial_qc_category" not in self.trial_qc.columns:
            raise FileNotFoundError(
                "No trial_qc.tsv file was found for this subject, so trial-category filtering is unavailable."
            )

        if isinstance(trial_category, (int, np.integer)):
            keep_trials = self.trial_qc["trial_qc_category"].map(TRIAL_CATEGORY_CODES).eq(int(trial_category)).to_numpy()
        else:
            keep_trials = self.trial_qc["trial_qc_category"].eq(str(trial_category)).to_numpy()
        if not keep_trials.any():
            raise ValueError(f'No trials were labeled "{trial_category}" for subject {self.sub}.')

        self.epochs_obj = self.epochs_obj[keep_trials]
        self.events = self.events.loc[keep_trials].reset_index(drop=True)
        self.rej_chans = self.rej_chans[keep_trials]
        self.rej_reasons = self.rej_reasons[keep_trials]
        self.rej_manual = self.rej_manual[keep_trials]
        self.trial_indices = self.trial_indices[keep_trials]
        self.trial_qc = self.trial_qc.loc[keep_trials].reset_index(drop=True)
        self.all_port_codes = [codes for keep, codes in zip(keep_trials, self.all_port_codes) if keep]
        self.all_portcode_times = [times for keep, times in zip(keep_trials, self.all_portcode_times) if keep]

    def get_rejection_reason(self, trial):
        """Return human-readable artifact labels for one trial.

        Parameters
        ----------
        trial : int
            Trial index in the currently loaded dataset.

        Returns
        -------
        list[str]
            One string per flagged channel, formatted as
            ``"<channel>: <reason>"``.
        """
        reasons = []
        for ch in np.where(self.rej_chans[trial])[0]:
            reasons.append(f"{self.chan_labels[ch]}: {self.rej_reasons[trial, ch]}")
        return reasons

    def preprocess_data_for_plot(self, downscale=None, chan_offset=None):
        """Build stacked and unstacked plotting arrays for the browser.

        Parameters
        ----------
        downscale : dict | None, optional
            Optional per-channel-type scale factors. If ``None``, reuse the
            values stored on the instance.
        chan_offset : float | None, optional
            Vertical separation between channels. If ``None``, reuse the
            current instance value.

        Returns
        -------
        None
            The method stores plotting arrays and axis limits on the instance.
        """

        if downscale is None:
            downscale = self.downscale
        if chan_offset is None:
            chan_offset = self.chan_offset

        self.offset_dict = self._build_offset_dict(chan_offset)

        epochs_raw = self.epochs_raw.copy()

        epochs_raw *= self.extra_chan_scale

        self.epochs_pre = np.full(epochs_raw.shape, np.nan)

        self.ys = []
        for ichan in range(epochs_raw.shape[1]):
            extra_offset = self.offset_dict[self.chan_types[ichan]]
            downscale_factor = downscale[self.chan_types[ichan]]
            self.epochs_pre[:, ichan] = (epochs_raw[:, ichan] * downscale_factor) - chan_offset * ichan - extra_offset
            self.ys.append(-1 * chan_offset * ichan - extra_offset)
        self.ys = np.array(self.ys)

        self.ylim = (
            np.nanpercentile(self.epochs_pre[:, -1], 5)
            - chan_offset * 1.5,  # 5th percentile of lowest channel + 2 chans
            np.nanpercentile(self.epochs_pre[:, 0], 95) + chan_offset * 1.5,  # 95th percentile of highest + 2 chans
        )

        self.offset_dict_stacked = self._build_stacked_offset_dict()

        self.epochs_stacked = np.full(epochs_raw.shape, np.nan)
        self.ys_stacked = []
        for ichan in range(epochs_raw.shape[1]):

            if self.chan_types[ichan] == "eyegaze":
                if "x" in self.chan_labels[ichan]:
                    extra_offset = self.offset_dict_stacked["eyegaze_x"]
                elif "y" in self.chan_labels[ichan]:
                    extra_offset = self.offset_dict_stacked["eyegaze_y"]
                else:
                    raise ValueError('Eyegaze channels must be labeled with "x" or "y"')
            else:
                extra_offset = self.offset_dict_stacked[self.chan_types[ichan]]

            downscale_factor = downscale[self.chan_types[ichan]]
            self.epochs_stacked[:, ichan] = (epochs_raw[:, ichan] * downscale_factor) - extra_offset
            self.ys_stacked.append(-extra_offset)
        self.ys_stacked = np.array(self.ys_stacked)
        self.ylim_stacked = (
            np.nanpercentile(self.epochs_stacked[:, -1], 5)
            - chan_offset * 1.5,  # 5th percentile of lowest channel + 2 chans
            np.nanpercentile(self.epochs_stacked[:, 0], 95) + chan_offset * 1.5,  # 95th percentile of highest + 2 chans
        )

    def _build_offset_dict(self, chan_offset: float) -> dict[str, float]:
        """Return vertical offsets for the standard multi-channel view.

        Parameters
        ----------
        chan_offset : float
            Base vertical spacing between channels.

        Returns
        -------
        dict[str, float]
            Offset for each supported channel type.
        """
        return {
            "eeg": 0.0,
            "eog": chan_offset,
            "eyegaze": chan_offset * 2,
            "pupil": chan_offset * 3,
            "misc": chan_offset * 5,
        }

    def _build_stacked_offset_dict(self) -> dict[str, float]:
        """Return vertical offsets for the stacked browser view.

        Parameters
        ----------
        None
            The method uses the current channel layout and y coordinates.

        Returns
        -------
        dict[str, float]
            Offsets that group similar channels together in stacked mode.
        """
        offset_dict_stacked = {
            "eeg": -self.ys[self.chan_types == "eeg"][-3],
        }
        if np.sum(self.chan_types == "eog") > 0:
            offset_dict_stacked["eog"] = -self.ys[self.chan_types == "eog"].mean()
        if np.sum(self.chan_types == "eyegaze") > 0:
            offset_dict_stacked["eyegaze_x"] = -self.ys[self.chan_types == "eyegaze"][0]
            offset_dict_stacked["eyegaze_y"] = -self.ys[self.chan_types == "eyegaze"][-1]
        if np.sum(self.chan_types == "pupil") > 0:
            offset_dict_stacked["pupil"] = -self.ys[self.chan_types == "pupil"].mean()
        if np.sum(self.chan_types == "misc") > 0:
            offset_dict_stacked["misc"] = -self.ys[self.chan_types == "misc"][-1]
        return offset_dict_stacked

    def open_figure(self, color="white"):
        """Open the interactive Matplotlib window for manual review.

        Parameters
        ----------
        color : str, optional
            Background color for the slider axis.

        Returns
        -------
        None
            The method creates the figure, slider, and keyboard bindings.
        """

        self.rej_reasons_on = False
        self.port_codes_on = False

        self.stack = False
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(f"EEG Viewer - Subject {self.sub} (press H for help)")

        self.plot_pos(0)
        axis_position = plt.axes([0.2, -0.1, 0.65, 0.03], facecolor=color)
        self.slider = Slider(axis_position, "Pos", 0, self.epochs_pre.shape[0], valstep=1)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress_event)
        self.fig.canvas.mpl_connect("button_press_event", self.click_toggle)

        # make a new axis for the help window

        self.help_ax = plt.axes()

        self.help_ax.set_title("Keyboard Shortcuts", size=40)
        self.help_ax.text(
            0.5,
            1,
            "\n h: Hide and show this window \n"
            + "[ and ]: Change window size \n"
            + "+ and -: Change channel scale \n"
            + "r: Show rejection reasons \n"
            + "p: Show port codes \n"
            + "c: Stack channels \n"
            + "w: Save annotations \n",
            horizontalalignment="center",
            verticalalignment="top",
            transform=self.help_ax.transAxes,
            size=20,
        )
        self.help_ax.set_axis_off()
        self.help_ax.set_visible(False)

    def plot_channels(self, epochs, pos):
        """Plot one browser window of channel traces.

        Parameters
        ----------
        epochs : np.ndarray
            Plot-ready data with shape ``(n_trials, n_channels, n_times)``.
        pos : int
            First trial index shown in the current browser window.

        Returns
        -------
        None
            The traces and trial annotations are drawn on ``self.ax``.
        """
        self.ax.plot(
            np.concatenate(epochs[pos : pos + self.win_step, ~self.ignored_channels_mask], 1).T,
            color="#000000",
            linewidth=0.75,
        )  # good channels

        self.ax.plot(
            np.concatenate(epochs[pos : pos + self.win_step, self.ignored_channels_mask], 1).T,
            color="#666666",
            linewidth=0.75,
        )  # ignored channels in gray

        for i, epoch in enumerate(range(pos, pos + self.win_step)):
            # annotate with condition labels

            self.ax.annotate(
                self._trial_annotation_text(epoch),
                (
                    i * self.epoch_len + self.epoch_len / 2,
                    self.ylim[1] + 1.05 * CHAN_OFFSET,
                ),
                annotation_clip=False,
                ha="center",
            )
            if self.rej_manual[epoch]:
                self.ax.plot(
                    np.arange(i * self.epoch_len, (i + 1) * self.epoch_len),
                    epochs[epoch, self.rej_chans[epoch]].T,
                    color="#FF0000",
                    linewidth=1,
                )
                self.ax.fill_between(
                    [i * self.epoch_len, (i + 1) * self.epoch_len],
                    [self.ylim[0]],
                    [self.ylim[1]],
                    color="#edb74a",
                    alpha=0.4,
                    zorder=-10,
                )

    def plot_helper_lines(self):
        """Draw trial and rejection-window guides for the current view."""
        self.ax.vlines(
            np.arange(self.epoch_len, self.epoch_len * self.win_step, self.epoch_len),
            -1,
            1,
            "#000000",
            linewidths=3,
        )  # Divide Epochs
        self.ax.vlines(
            np.arange(
                (self.rejection_time[0] - self.trial_start) * self.srate,
                self.epoch_len * self.win_step,
                self.epoch_len,
            ),
            -1,
            1,
            "#0000FF",
            linewidths=1.5,
        )  # baseline start
        self.ax.vlines(
            np.arange(
                (-self.trial_start) * self.srate,
                self.epoch_len * self.win_step,
                self.epoch_len,
            ),
            -1,
            1,
            "#FF00FF",
        )  # Task start
        self.ax.vlines(
            np.arange(
                (self.rejection_time[1] - self.trial_start) * self.srate,
                self.epoch_len * self.win_step,
                self.epoch_len,
            ),
            -1,
            1,
            "#0000FF",
            linewidths=1.5,
        )  # end of delay

    def _trial_annotation_text(self, epoch: int) -> str:
        """Build the per-trial label shown above each plotted epoch.

        Parameters
        ----------
        epoch : int
            Trial index in the currently loaded set.

        Returns
        -------
        str
            Multi-line label with trial number, condition, and QC category when
            available.
        """
        trial_number = epoch
        category = None
        if self.trial_qc is not None:
            if "trial_index" in self.trial_qc.columns:
                trial_number = int(self.trial_qc.loc[epoch, "trial_index"])
            if "trial_qc_category" in self.trial_qc.columns:
                category = self.trial_qc.loc[epoch, "trial_qc_category"]

        label = f"Trial {trial_number}\n{self.events['trial_type'][epoch]}"
        if category:
            label += f"\n{category}"
        return label

    def plot_pos(self, pos):
        """Draw the current browser window at one slider position.

        Parameters
        ----------
        pos : int
            First trial index shown in the browser window.

        Returns
        -------
        None
            The current axis is updated in place.
        """

        self.plot_helper_lines()

        if self.stack:
            self.plot_channels(self.epochs_stacked, pos)
            self.ax.set_ylim(*self.ylim_stacked)
            self.ax.set_yticks([y * -1 for y in self.offset_dict_stacked.values()], self.offset_dict_stacked.keys())

        else:
            self.plot_channels(self.epochs_pre, pos)
            self.ax.set_ylim(*self.ylim)
            self.ax.set_yticks(self.ys, self.chan_labels)

        self.ax.set_xlim(*self.xlim)
        self._set_time_axis_ticks()

    def _set_time_axis_ticks(self) -> None:
        """Show rejection-window time labels on the x-axis when informative."""
        if self.rejection_time[0] == self.trial_start and self.rejection_time[1] == self.trial_end:
            self.ax.set_xticks([])
            return

        tick_positions = []
        tick_labels = []

        if self.rejection_time[0] != self.trial_start:
            tick_positions.extend(
                np.arange(
                    (self.rejection_time[0] - self.trial_start) * self.srate,
                    self.epoch_len * self.win_step,
                    self.epoch_len,
                )
            )
            tick_labels.extend([int(self.rejection_time[0] * 1000)] * self.win_step)

        if self.rejection_time[1] != self.trial_end:
            tick_positions.extend(
                np.arange(
                    (self.rejection_time[1] - self.trial_start) * self.srate,
                    self.epoch_len * self.win_step,
                    self.epoch_len,
                )
            )
            tick_labels.extend([int(self.rejection_time[1] * 1000)] * self.win_step)

        tick_order = np.argsort(tick_positions)
        tick_positions = np.asarray(tick_positions)[tick_order]
        tick_labels = np.asarray(tick_labels, dtype=int)[tick_order]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels.tolist())

    def show_rejection_reasons(self):
        """Annotate flagged channels for the trials on screen."""
        self.rej_annotations = []

        for i in range(self.win_step):
            trial = self.slider.val + i

            for ch in np.where(self.rej_chans[trial])[0]:
                an = self.ax.annotate(
                    f"{self.chan_labels[ch]}: {self.rej_reasons[trial,ch]}",
                    (i * self.epoch_len, self.ys[ch]),
                    backgroundcolor="white",
                    annotation_clip=False,
                )
                self.rej_annotations.append(an)

    def show_port_codes(self):
        """Annotate event codes and sample times for the trials on screen."""

        self.code_annotations = []
        all_codes = []
        all_times = []
        for i in range(self.win_step):
            times = self.all_portcode_times[self.slider.val + i]
            all_times.extend([t + i * self.epoch_len for t in times])
            all_codes.extend(self.all_port_codes[self.slider.val + i])

        self.code_lines = self.ax.vlines(all_times, *self.ylim, color="g")

        for code, time in zip(all_codes, all_times):
            an = self.ax.annotate(code, (time, self.ylim[1] + 5e-6), ha="center", annotation_clip=False)
            self.code_annotations.append(an)

    def update(self, force=False):
        """Redraw the browser after a state change.

        Parameters
        ----------
        force : bool, optional
            If ``True``, trigger a full canvas redraw. Otherwise, use a faster
            axis-only blit update.

        Returns
        -------
        None
            The current figure is updated in place.
        """
        pos = self.slider.val
        if pos < 0:
            self.slider.set_val(0)
            self.update()
        elif pos > self.epochs_pre.shape[0] - self.win_step:
            self.slider.set_val(self.epochs_pre.shape[0] - self.win_step)
            self.update()
        else:
            self.ax.clear()
            self.plot_pos(pos)
            if self.rej_reasons_on:
                self.show_rejection_reasons()
            if self.port_codes_on:
                self.show_port_codes()
            if force:
                self.fig.canvas.draw_idle()
            else:
                self.fig.canvas.blit(self.ax.bbox)

    def keypress_event(self, ev):
        """Handle keyboard shortcuts for navigation and annotation."""
        match ev.key:
            case "right":
                self.slider.set_val(self.slider.val + self.win_step)
                self.update()
            case "left":
                self.slider.set_val(self.slider.val - self.win_step)
                self.update()
            case ".":
                self.slider.set_val(self.slider.val + 1)
                self.update()
            case ",":
                self.slider.set_val(self.slider.val - 1)
                self.update()
            case "[":
                self._change_window_size(-1)
            case "]":
                self._change_window_size(1)

            case "+":
                self._change_channel_scale(0.2, 2)
            case "-":
                self._change_channel_scale(-0.2, 0.5)
            case "w":
                self.save_annotations()

            case "r":
                self._toggle_rejection_reasons()

            case "p":
                self._toggle_port_codes()

            case "c":
                self.stack = not self.stack
                self.update(force=True)
            case "h":
                self.help_ax.set_visible(not self.help_ax.get_visible())
                self.ax.set_visible(not self.ax.get_visible())
                self.fig.canvas.draw_idle()
            case _:
                print(f"key not recognized: {ev.key}. Press h for help.")

    def click_toggle(self, ev):
        """Toggle one trial's manual rejection state with a mouse click."""

        if ev.button is MouseButton.LEFT and ev.xdata is not None:
            pos = self.slider.val

            epoch_index = np.arange(pos, pos + self.win_step)[int(ev.xdata // self.epoch_len)]
            self.rej_manual[epoch_index] = not self.rej_manual[epoch_index]
            self.update(force=True)

    def _change_window_size(self, step_change: int) -> None:
        """Increase or decrease the number of trials shown at once."""
        new_win_step = self.win_step + step_change
        if new_win_step < 1:
            return
        self.win_step = new_win_step
        self.xlim = (0, self.epoch_len * self.win_step)
        self.update(force=True)

    def _change_channel_scale(self, scale_change: float, offset_factor: float) -> None:
        """Update channel scaling and rebuild the plotting arrays."""
        self.extra_chan_scale += scale_change
        self.chan_offset *= offset_factor
        self.preprocess_data_for_plot()
        self.update(force=True)

    def _toggle_rejection_reasons(self) -> None:
        """Show or hide text labels that explain channel rejections."""
        if self.rej_reasons_on:
            for an in self.rej_annotations:
                an.remove()
            self.rej_reasons_on = False
            self.fig.canvas.draw_idle()
            return

        self.show_rejection_reasons()
        self.rej_reasons_on = True
        self.fig.canvas.draw_idle()

    def _toggle_port_codes(self) -> None:
        """Show or hide event-code markers for the visible trials."""
        if self.port_codes_on:
            for an in self.code_annotations:
                an.remove()
            self.code_lines.remove()
            self.port_codes_on = False
            self.fig.canvas.draw_idle()
            return

        self.port_codes_on = True
        self.show_port_codes()
        self.fig.canvas.draw_idle()

    def save_annotations(self):
        """Save manual rejection flags aligned to the full trial list.

        Parameters
        ----------
        None
            The method uses the currently loaded subject and manual review
            state.

        Returns
        -------
        None
            The updated boolean rejection vector is written to disk.
        """
        save_path = self._build_sidecar_path("rejection_flags", ".npy")
        self.rej_manual_full[self.trial_indices] = self.rej_manual
        print(
            f'{np.sum(self.rej_manual_full)}/{len(self.rej_manual_full)} trials rejected. Saving annotations as "{save_path}"'
        )
        np.save(save_path, self.rej_manual_full)
        self._save_reviewed_trial_qc()

    def _save_reviewed_trial_qc(self) -> None:
        """Update the saved trial QC table with the reviewed final decision.

        Parameters
        ----------
        None
            The method uses the current full-length manual rejection vector.

        Returns
        -------
        None
            The function updates ``trial_qc.tsv`` in place when it exists.
        """
        trial_qc_path = self._resolve_related_file("trial_qc", ".tsv", required=False)
        if trial_qc_path is None:
            return

        trial_qc_full = pd.read_csv(trial_qc_path, sep="\t", keep_default_na=False)
        if "trial_qc_category" in trial_qc_full.columns:
            trial_qc_full["trial_qc_category"] = trial_qc_full["trial_qc_category"].replace(
                {"needs_manual_review": "unclear"}
            )

        final_qc_category = np.where(self.rej_manual_full, "rejected", "accepted")
        trial_qc_full["final_manual_reject"] = self.rej_manual_full.astype(int)
        trial_qc_full["final_qc_category"] = final_qc_category
        trial_qc_full["final_qc_code"] = np.where(self.rej_manual_full, 3, 1)
        trial_qc_full.to_csv(trial_qc_path, sep="\t", index=False)

        trial_state_path = self._resolve_related_file("trial_state", ".tsv", required=False)
        if trial_state_path is not None:
            trial_state = pd.read_csv(trial_state_path, sep="\t", keep_default_na=False)
            trial_state["soft_reject"] = trial_qc_full["trial_qc_category"].astype(str).eq("unclear").to_numpy(dtype=bool)
            trial_state["final_keep"] = ~self.rej_manual_full
            trial_state["final_qc_category"] = final_qc_category
            trial_state["state_stage"] = "final"
            trial_state.to_csv(trial_state_path, sep="\t", index=False)
