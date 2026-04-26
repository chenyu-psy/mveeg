import mne_bids.read
import numpy as np
import mne
import mne_bids
import os
import pandas as pd
import json
import shutil
import subprocess
import warnings
from pathlib import Path
from dataclasses import dataclass
from ..io.bids import build_derivative_stem, build_subject_label, build_task_stem, find_subject_dir, normalize_subject_id


FILTER_FREQS = (0, 80)  # low, high


@dataclass
class IOConfig:
    """Input/output settings for locating raw files and writing BIDS output."""
    data_dir: str | Path
    root_dir: str | Path
    experiment_name: str | None = None
    subject_prefix: str = "sub"
    derivative_dirname: str = "derivatives"
    derivative_label: str = "preprocessed"
    raw_datatype: str = "eeg"
    derivative_datatype: str = "eeg"
    drop_channels: list | tuple | None = None


@dataclass
class EpochConfig:
    trial_start: float
    trial_end: float
    srate: int | None = None
    baseline_time: tuple[float, float] | None = None
    rejection_time: tuple[float, float] | None = None
    reject_between_codes: tuple | None = None
    filter_freqs: tuple[None | int | float, None | int | float] = FILTER_FREQS


@dataclass
class EventConfig:
    event_dict: dict
    event_code_dict: dict
    timelock_ix: int | dict | None = None
    event_names: dict | None = None


class Preprocess:

    def __init__(
        self,
        *,
        io_config: IOConfig,
        epoch_config: EpochConfig,
        event_config: EventConfig,
    ):
        """Initialize preprocessing options from grouped config objects."""
        # Keep constructor shallow: each helper owns one concern.
        self._warned_missing_templates = set()
        self._warned_manual_sidecar = False
        self._apply_io_config(io_config)
        self._apply_epoch_config(epoch_config)
        self._apply_event_config(event_config)
        self.timelock_ix = self._build_timelock_index(event_config.timelock_ix)
        self._configure_rejection_window(epoch_config.rejection_time, epoch_config.reject_between_codes)

    def _apply_io_config(self, io_config: IOConfig):
        """Store input/output settings from ``IOConfig`` on the preprocessor."""
        self.data_dir = io_config.data_dir
        self.root_dir = io_config.root_dir
        self.experiment_name = io_config.experiment_name
        self.subject_prefix = io_config.subject_prefix
        self.derivative_dirname = io_config.derivative_dirname
        self.derivative_label = io_config.derivative_label
        self.raw_datatype = io_config.raw_datatype
        self.derivative_datatype = io_config.derivative_datatype
        self.drop_channels = io_config.drop_channels

    def _normalize_subject_id(self, subject_number: str) -> str:
        """Convert a raw subject label into a BIDS-safe subject identifier.

        Parameters
        ----------
        subject_number : str
            Subject label used in raw folders or caller input, for example
            ``"1001"``, ``"sub1001"``, or ``"sub-1001"``.

        Returns
        -------
        str
            Normalized BIDS subject identifier without the leading ``sub``.
        """
        return normalize_subject_id(subject_number, subject_prefix=self.subject_prefix)

    def _resolve_subject_dir(self, subject_number: str) -> Path:
        """Find the raw-data folder for a subject.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or a BIDS-style ID.

        Returns
        -------
        Path
            Raw-data folder that contains the subject files.

        Raises
        ------
        FileNotFoundError
            If no matching subject folder exists in ``root_dir``.
        """
        return find_subject_dir(self.root_dir, subject_number, subject_prefix=self.subject_prefix)

    def _derivative_subject_label(self, subject_number: str) -> str:
        """Build the standard BIDS subject label used in saved outputs.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.

        Returns
        -------
        str
            Subject label used in imported and derivative outputs, for example
            ``"sub-1001"``.
        """
        return build_subject_label(subject_number, subject_prefix=self.subject_prefix)

    def _raw_datatype_dir(self, subject_number: str, datatype: str) -> Path:
        """Return the imported raw-data directory for one datatype.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.
        datatype : str
            BIDS datatype folder name such as ``"eeg"`` or ``"beh"``.

        Returns
        -------
        Path
            Directory inside ``data_dir`` for that subject and datatype.
        """
        return Path(self.data_dir) / self._derivative_subject_label(subject_number) / datatype

    def _raw_file_stem(self, subject_number: str) -> str:
        """Build the common filename stem for imported raw task files.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.

        Returns
        -------
        str
            Filename stem shared by raw task files written during import, for
            example ``"sub-1001_task-experiment1"``.
        """
        return build_task_stem(
            subject_number,
            self.experiment_name,
            subject_prefix=self.subject_prefix,
        )

    def _raw_file_path(
        self,
        subject_number: str,
        datatype: str,
        suffix: str,
        extension: str,
        split: int | None = None,
    ) -> Path:
        """Build one imported raw-data file path.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.
        datatype : str
            Output datatype folder such as ``"eeg"``, ``"beh"``, or
            ``"eyetracking"``.
        suffix : str
            BIDS suffix used in the filename, for example ``"eeg"`` or
            ``"events"``.
        extension : str
            Filename extension including the leading dot.
        split : int | None, optional
            Optional split index used when one raw recording is saved across
            multiple files.

        Returns
        -------
        Path
            Full output path inside ``data_dir``.

        Notes
        -----
        Raw task files in this project follow the standard BIDS
        ``sub-<id>_task-<task>_<suffix>`` pattern. Dataset-level sidecars such
        as ``electrodes.tsv`` keep the shorter ``sub-<id>_<suffix>`` pattern.
        """
        datatype_dir = self._raw_datatype_dir(subject_number, datatype)
        split_label = "" if split is None else f"_split-{split:02d}"
        stem = self._derivative_subject_label(subject_number)
        if suffix in {"beh", "eeg", "events", "eyetracking"}:
            stem = self._raw_file_stem(subject_number)
        return datatype_dir / f"{stem}{split_label}_{suffix}{extension}"

    def _raw_eeg_bids_path(
        self,
        subject_number: str,
        suffix: str,
        extension: str,
        space: str | None = None,
    ) -> Path:
        """Build a raw EEG-side BIDS path using MNE-BIDS naming rules.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.
        suffix : str
            BIDS suffix to locate, for example ``"eeg"``, ``"events"``, or
            ``"electrodes"``.
        extension : str
            Filename extension including the leading dot.
        space : str | None, optional
            Optional BIDS space entity, used for files such as
            ``space-CapTrak_electrodes.tsv``.

        Returns
        -------
        Path
            Full path to the requested EEG-side BIDS file.

        Notes
        -----
        This helper is only for files written by ``mne_bids.write_raw_bids`` in
        the EEG folder. It keeps the naming rules explicit instead of guessing
        filenames from glob patterns.
        """
        bids_path = mne_bids.BIDSPath(
            subject=self._normalize_subject_id(subject_number),
            task=self.experiment_name,
            root=self.data_dir,
            datatype=self.raw_datatype,
            suffix=suffix,
            extension=extension,
            space=space,
            check=False,
        )
        if suffix == "electrodes":
            bids_path = bids_path.update(task=None)
        return bids_path.fpath

    def _derivative_eeg_dir(self, subject_number: str) -> Path:
        """Return the derivative EEG directory for one subject.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.

        Returns
        -------
        Path
            Output directory for saved derivative EEG files.
        """
        return (
            Path(self.data_dir)
            / self.derivative_dirname
            / self._derivative_subject_label(subject_number)
            / self.derivative_datatype
        )

    def _derivative_file_stem(self, subject_number: str) -> str:
        """Build the common filename stem for derivative outputs.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or folder name.

        Returns
        -------
        str
            Filename stem shared by all derivative EEG sidecars.
        """
        return build_derivative_stem(
            subject_number,
            self.experiment_name,
            subject_prefix=self.subject_prefix,
            derivative_label=self.derivative_label,
        )

    def _find_subject_files(self, subject_dir: Path, subject_number: str, extension: str) -> list[str]:
        """Find subject files with a given extension.

        Parameters
        ----------
        subject_dir : Path
            Folder containing one subject's raw files.
        subject_number : str
            Subject label from the caller or folder name.
        extension : str
            File extension including the leading dot, for example ``".vhdr"``.

        Returns
        -------
        list[str]
            Matching filenames relative to ``subject_dir``, sorted
            alphabetically.
        """
        subject_id = self._normalize_subject_id(subject_number)
        subject_labels = {
            f"{self.subject_prefix}{subject_id}",
            f"{self.subject_prefix}-{subject_id}",
            subject_id,
        }
        matches = []
        for path in sorted(subject_dir.glob(f"*{extension}")):
            if any(label in path.stem for label in subject_labels):
                matches.append(path.name)
        return matches

    def _apply_epoch_config(self, epoch_config: EpochConfig):
        self.srate = epoch_config.srate
        self.trial_start_t = epoch_config.trial_start
        self.trial_end_t = epoch_config.trial_end
        self.baseline_time = epoch_config.baseline_time
        self.filter_freqs = epoch_config.filter_freqs

    def _apply_event_config(self, event_config: EventConfig):
        self.event_dict = event_config.event_dict
        self.event_dict_inv = {v: k for k, v in self.event_dict.items()}  # invert for later use
        self.event_code_dict = event_config.event_code_dict
        # Copy mappings so caller-owned dicts are not mutated inside this class.
        self.event_names = (
            dict(event_config.event_names)
            if event_config.event_names is not None
            else dict(self.event_dict)
        )
        self.event_names.update(
            {
                "New Segment/": 99999,
                "New Segment/LostSamples: 2": 10001,
            }
        )

    def _base_bids_template_path(self, filename: str) -> Path:
        """Return the expected path to a bundled BIDS template file.

        Parameters
        ----------
        filename : str
            Template filename inside the optional ``base_bids_files`` folder.

        Returns
        -------
        Path
            Full path to the requested template file.

        Notes
        -----
        Some project copies may not include these templates. Callers should
        check ``path.exists()`` before using the returned path.
        """
        return Path(__file__).parent.joinpath("../base_bids_files", filename)

    def _copy_sidecar_template(self, output_path: Path, template_name: str):
        """Copy a sidecar template when available, otherwise keep processing.

        Parameters
        ----------
        output_path : Path
            Destination sidecar path in the BIDS dataset.
        template_name : str
            Template filename expected inside ``base_bids_files``.

        Returns
        -------
        None
            The function copies the template when present.

        Notes
        -----
        Eyetracking and events sidecars are optional in this project setup.
        Missing templates should not stop preprocessing.
        """
        template_path = self._base_bids_template_path(template_name)
        if not template_path.exists():
            if template_name not in self._warned_missing_templates:
                print(
                    f"WARNING: Missing BIDS template {template_name}. "
                    f"Please review {output_path.name} manually if needed."
                )
                self._warned_missing_templates.add(template_name)
            return

        shutil.copy(template_path, output_path)

    def _build_timelock_index(self, timelock_ix: int | dict | None) -> dict:
        # Support either a single shared lock index or per-condition indices.
        if isinstance(timelock_ix, dict):
            return timelock_ix
        if isinstance(timelock_ix, int):
            return {k: timelock_ix for k in self.event_code_dict.keys()}
        if timelock_ix is None:
            # Default: lock to the first occurrence of each condition code in its sequence.
            return {k: v.index(k) for k, v in self.event_code_dict.items()}
        raise TypeError(
            "timelock_ix must be either an int, a dict with event codes as keys and indices as values, or None"
        )

    def _configure_rejection_window(
        self,
        rejection_time: tuple[float, float] | None,
        reject_between_codes: tuple | None,
    ):
        # Rejection can be defined by time-window OR event-code window, but never both.
        if rejection_time is not None and reject_between_codes is not None:
            raise ValueError("You cannot specify both rejection_time and reject_between_codes")

        if rejection_time is not None:
            self.rejection_time = self._normalize_rejection_time(rejection_time)
            self.rejection_codes = None
            return

        if reject_between_codes is not None:
            if len(reject_between_codes) != 2:
                raise ValueError("reject_between_codes must be a tuple of length 2")
            if (
                reject_between_codes[0] not in self.event_dict.values()
                or reject_between_codes[1] not in self.event_dict.values()
            ):
                raise ValueError("inputs to reject_between_codes must be valid event codes")
            self.rejection_codes = reject_between_codes
            self.rejection_time = None
            return

        self.rejection_time = (self.trial_start_t, self.trial_end_t)
        self.rejection_codes = None

    def _normalize_rejection_time(self, rejection_time):
        # Allow open-ended bounds via None and fill from epoch boundaries.
        if len(rejection_time) != 2:
            raise ValueError("rejection_time must be a tuple of length 2")

        if rejection_time[0] is None and rejection_time[1] is not None:
            normalized = (self.trial_start_t, rejection_time[1])
        elif rejection_time[0] is not None and rejection_time[1] is None:
            normalized = (rejection_time[0], self.trial_end_t)
        elif rejection_time[0] is None and rejection_time[1] is None:
            normalized = (self.trial_start_t, self.trial_end_t)
        else:
            normalized = rejection_time

        if normalized[0] >= normalized[1]:
            raise ValueError("rejection_time[0] must be less than rejection_time[1]")
        if normalized[0] < self.trial_start_t or normalized[1] > self.trial_end_t:
            raise ValueError("rejection_time must be within the trial time range")
        return normalized

    def rereference_to_average(self, data, reref_values):
        """
        re reference data to the average of the offline reference and the given data from another channel
        """
        assert data.shape == reref_values.shape
        return data - (0.5 * reref_values)

    def _read_raw_brainvision_quiet(self, vhdr_path, preload=False):
        """Read BrainVision files while hiding expected metadata warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"Online software filter detected.*")
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=r"Channels contain different highpass filters.*"
            )
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=r"Channels contain different lowpass filters.*"
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"Not setting positions of .* eog/misc channels found in montage:.*",
            )
            return mne.io.read_raw_brainvision(
                vhdr_path,
                eog=["HEOG", "VEOG"],
                misc=["StimTrak"],
                preload=preload,
                verbose="ERROR",
            )

    def _write_raw_bids_quiet(self, eegdata, bids_path, overwrite, events, event_id):
        """Write BIDS EEG while hiding expected BrainVision conversion warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"Converting data files to BrainVision format",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r"Encountered unsupported non-voltage units:.*",
            )
            mne_bids.write_raw_bids(
                eegdata,
                bids_path,
                overwrite=overwrite,
                events=events,
                event_id=event_id,
                verbose=False,
                allow_preload=True,
                format="BrainVision",
            )

    def _warn_manual_sidecar_once(self):
        """Print one shared sidecar-edit reminder per preprocessing run."""
        if self._warned_manual_sidecar:
            return
        print("WARNING: YOU WILL HAVE TO MODIFY THE SIDECAR FILE YOURSELF")
        self._warned_manual_sidecar = True

    def import_eeg(self, subject_number, overwrite=False):
        """
        function to import raw eeg data and convert it to a bids object

        Args:
            subject_number: the subject number (data should be in a folder with this name)
            overwrite (bool): if the data already exists, should it be just read from disc (false) or rewritten (true)
        Returns:
            bids object (saved from raw EEG)
            mne-python raw dataset
            events array (as a dataframe)
        """

        subject_dir = self._resolve_subject_dir(subject_number)
        eeg_path = self._raw_file_path(subject_number, "eeg", "eeg", ".vhdr")
        events_path = self._raw_file_path(subject_number, "eeg", "events", ".tsv")

        if not overwrite:
            try:
                eegdata = self._read_raw_brainvision_quiet(eeg_path, preload=False)
                events = pd.read_csv(events_path, sep="\t")
                return eegdata, events
            except FileNotFoundError as e:
                print(e)
                print("Could not find EEG data in your directory. I will try to import it from the raw data")

        vhdr_file = self._find_subject_files(subject_dir, subject_number, ".vhdr")

        if len(vhdr_file) == 0:
            raise FileNotFoundError("No vhdr files in subject directory")
        elif len(vhdr_file) == 1:
            eegfile = os.path.join(subject_dir, vhdr_file[0])  # search for vhdr file
            eegdata = self._read_raw_brainvision_quiet(eegfile, preload=False)  # read into mne.raw structure
        elif len(vhdr_file) > 1:
            raws = []

            for file in vhdr_file:
                print(
                    "More than 1 vhdr file present in subject directory. They will be concatenated in alphabetical order"
                )

                eegfile = os.path.join(subject_dir, file)  # search for vhdr file
                raws.append(self._read_raw_brainvision_quiet(eegfile, preload=False))
            eegdata = mne.concatenate_raws(raws, verbose="ERROR")

        events, event_dict = mne.events_from_annotations(eegdata, verbose="ERROR")
        boundaries = {k: v for k, v in event_dict.items() if "New Segment" in k}
        self.event_names.update(boundaries)

        # drop artifactual extra conditions
        events_to_keep = np.isin(events[:, 2], list(self.event_names.values()))
        if any(~events_to_keep):
            uq_events, counts = np.unique(events[~events_to_keep, 2], return_counts=True)
            print(
                f"WARNING: Dropping {np.sum(counts)} events that are not in the event list:"
                + "\n".join([f"{uq_events[i]} ({counts[i]})" for i in range(len(uq_events))])
            )
        events = events[events_to_keep]

        eegdata.set_annotations(None)

        self._write_raw_bids_quiet(
            eegdata,
            mne_bids.BIDSPath(
                subject=self._normalize_subject_id(subject_number),
                task=self.experiment_name,
                root=self.data_dir,
                datatype="eeg",
                suffix="eeg",
                extension=".vhdr",
            ),
            overwrite,
            events,
            self.event_names,
        )

        self._copy_sidecar_template(self._raw_file_path(subject_number, "eeg", "eeg", ".json"), "TEMPLATE_eeg.json")
        events = pd.read_csv(events_path, sep="\t")

        return eegdata, events

    def import_behavior(self, subject_number, suffix="_beh.csv"):
        """Import behavioral data into a BIDS-compatible TSV file.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or raw-data folder.
        suffix : str, default "_data.csv"
            Required filename ending for the behavior file after any optional
            prefix match.

        Returns
        -------
        None
            The function writes a BIDS behavior TSV into ``data_dir``.
        """
        subject_dir = self._resolve_subject_dir(subject_number)
        beh_path = self._raw_file_path(subject_number, "beh", "beh", ".tsv")
        beh_path.parent.mkdir(parents=True, exist_ok=True)
        csv_files = self._find_subject_files(subject_dir, subject_number, ".csv")
        for f in csv_files:
            f = os.path.join(subject_dir, f)
            if f.endswith(suffix):
                pd.read_csv(f).to_csv(beh_path, sep="\t")

    def load_behavior_table(self, subject_number, suffix="_beh.csv"):
        """Load one subject's behavioral CSV file.

        Parameters
        ----------
        subject_number : str
            Subject label from the caller or raw-data folder.
        suffix : str, optional
            Required filename ending for the behavior file.

        Returns
        -------
        pandas.DataFrame
            Behavioral trial table from the raw-data folder.

        Raises
        ------
        FileNotFoundError
            If no matching behavior CSV is found.
        """
        subject_dir = self._resolve_subject_dir(subject_number)
        csv_files = self._find_subject_files(subject_dir, subject_number, ".csv")
        for filename in csv_files:
            full_path = os.path.join(subject_dir, filename)
            if full_path.endswith(suffix):
                return pd.read_csv(full_path)
        raise FileNotFoundError(f"Could not find a behavior file ending in {suffix} for subject {subject_number}.")

    def _load_behavior_table(self, subject_number, suffix="_beh.csv"):
        """Backward-compatible wrapper for ``load_behavior_table``."""
        return self.load_behavior_table(subject_number, suffix=suffix)

    def _normalize_epoch_trial_codes(self, event_codes):
        """Map epoch event codes back to their base condition codes.

        Parameters
        ----------
        event_codes : array-like
            Trial codes stored in ``epochs.events[:, 2]``.

        Returns
        -------
        np.ndarray
            Integer array with rejected-trial offset codes mapped back to their
            underlying condition code.
        """
        normalized = np.asarray(event_codes, dtype=int).copy()
        valid_codes = set(self.event_names.values())
        for i, code in enumerate(normalized):
            if code in valid_codes:
                continue
            if (code - 1000) in valid_codes:
                normalized[i] = code - 1000
        return normalized

    def _get_order_only_trial_codes(self):
        """Return trial codes that should align by order rather than condition.

        Returns
        -------
        set[int]
            Event codes whose condition must be inherited from behavior order.

        Notes
        -----
        Your current preprocessing script does not use any order-only trial
        codes. This helper remains available for datasets that need them.
        """
        order_only = set()
        for label, code in self.event_names.items():
            if "EARLY_TRIAL_REJECT" in label:
                order_only.add(code)
        return order_only

    def _align_behavior_rows_to_epochs(self, behavior_codes, epoch_codes):
        """Align behavior rows to epochs using longest common subsequence matching.

        Parameters
        ----------
        behavior_codes : array-like
            Condition codes from the behavior file in chronological order.
        epoch_codes : array-like
            Condition codes from the epoched EEG data in chronological order.

        Returns
        -------
        np.ndarray
            Integer indices into the behavior table, one per epoch.

        Raises
        ------
        RuntimeError
            If not all epochs can be aligned to behavior rows.

        Notes
        -----
        The behavior file can contain extra rows from practice, warm-up, or
        repeated trials. Longest common subsequence matching lets us skip those
        extra rows while preserving the order of the actual experiment. If a
        dataset defines any order-only trial codes, those codes match any
        behavior condition and inherit their label later.
        """
        behavior_codes = np.asarray(behavior_codes, dtype=int)
        epoch_codes = np.asarray(epoch_codes, dtype=int)
        order_only_codes = self._get_order_only_trial_codes()
        n_behavior = len(behavior_codes)
        n_epoch = len(epoch_codes)

        lcs = np.zeros((n_behavior + 1, n_epoch + 1), dtype=np.uint16)
        for i in range(n_behavior):
            for j in range(n_epoch):
                if epoch_codes[j] in order_only_codes or behavior_codes[i] == epoch_codes[j]:
                    lcs[i + 1, j + 1] = lcs[i, j] + 1
                else:
                    lcs[i + 1, j + 1] = max(lcs[i, j + 1], lcs[i + 1, j])

        matched_pairs = []
        i = n_behavior
        j = n_epoch
        while i > 0 and j > 0:
            if epoch_codes[j - 1] in order_only_codes or behavior_codes[i - 1] == epoch_codes[j - 1]:
                matched_pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif lcs[i - 1, j] >= lcs[i, j - 1]:
                i -= 1
            else:
                j -= 1

        matched_pairs.reverse()
        if len(matched_pairs) != n_epoch:
            raise RuntimeError(
                "Could not align every epoch to a behavior row. "
                f"Matched {len(matched_pairs)} of {n_epoch} epochs."
            )

        return np.array([behavior_ix for behavior_ix, _ in matched_pairs], dtype=int)

    def exclude_practice_trials(
        self,
        subject_number,
        epochs,
        suffix="_beh.csv",
        behavior=None,
        matched_behavior_filter=None,
    ):
        """Align epochs to cleaned behavior rows, then filter matched rows together.

        Parameters
        ----------
        subject_number : str
            Subject label used to find the raw behavior CSV.
        epochs : mne.Epochs
            Epoched EEG or EEG-plus-eye-tracking data.
        suffix : str, optional
            Required filename ending for the behavior file.
        behavior : pandas.DataFrame | None, optional
            Pre-filtered behavior table prepared by the caller. When
            ``None``, the method loads the subject's behavior CSV and applies
            the default ``exp``/``pra`` plus ``rejection == "no"`` filter.
        matched_behavior_filter : callable | None, optional
            Function applied after behavior rows have been aligned one-to-one
            with EEG epochs. It must accept a matched behavior table and return
            a table with a boolean ``keep_epoch`` column.

        Returns
        -------
        mne.Epochs
            Copy of ``epochs`` with only experimental trials kept after
            alignment to behavior-approved rows.

        Notes
        -----
        The behavior file in this project includes warm-up and practice rows in
        addition to the main experimental trials, and it also contains a manual
        per-trial rejection column. To keep EEG and behavior aligned to the same
        cleaned trial list, we first limit the behavior table to rows where
        ``trial_type`` is either ``"exp"`` or ``"pra"`` and
        ``rejection == "no"``. After the one-to-one alignment step, an optional
        matched-row filter can further drop practice rows or other conditions
        using the same indices in both datasets.
        """
        if behavior is None:
            behavior = self.load_behavior_table(subject_number, suffix=suffix)
            behavior = behavior[
                behavior["trial_type"].isin(["exp", "pra"]) & behavior["rejection"].eq("no")
            ].reset_index(drop=True)
        else:
            behavior = behavior.copy().reset_index(drop=True)

        behavior = behavior[behavior["label"].isin(self.event_names.keys())].reset_index(drop=True)
        behavior_codes = behavior["label"].map(self.event_names).to_numpy(dtype=int)

        order_only_codes = self._get_order_only_trial_codes()
        original_epoch_codes = np.asarray(epochs.events[:, 2], dtype=int)
        epoch_codes = self._normalize_epoch_trial_codes(original_epoch_codes)
        matched_behavior_ix = self._align_behavior_rows_to_epochs(behavior_codes, epoch_codes)
        behavior = behavior.iloc[matched_behavior_ix].reset_index(drop=True)
        behavior_codes = behavior_codes[matched_behavior_ix]
        if matched_behavior_filter is not None:
            behavior = matched_behavior_filter(behavior.copy().reset_index(drop=True))

        comparable = ~np.isin(original_epoch_codes, list(order_only_codes))
        if not np.array_equal(epoch_codes[comparable], behavior_codes[comparable]):
            raise RuntimeError("Behavior condition codes do not match epoch condition codes after alignment.")

        epochs_aligned = epochs.copy()
        early_reject_mask = np.isin(original_epoch_codes, list(order_only_codes))
        if np.any(early_reject_mask):
            relabeled_codes = behavior_codes[early_reject_mask] + 1000
            epochs_aligned.events[early_reject_mask, 2] = relabeled_codes
            print(f"Relabeled {int(early_reject_mask.sum())} early rejected trials using behavior order.")

        if "keep_epoch" in behavior.columns:
            keep_trials = behavior["keep_epoch"].to_numpy(dtype=bool)
        else:
            keep_trials = behavior["trial_type"].eq("exp").to_numpy()
        n_removed = int((~keep_trials).sum())
        epochs_filtered = epochs_aligned.copy()
        if n_removed > 0:
            # Use explicit drop with quiet logging to avoid noisy script output.
            drop_ix = np.flatnonzero(~keep_trials).tolist()
            epochs_filtered.drop(drop_ix, verbose="ERROR")

        aligned_behavior = behavior.loc[keep_trials].reset_index(drop=True)
        metadata = (
            epochs_filtered.metadata.reset_index(drop=True)
            if epochs_filtered.metadata is not None
            else pd.DataFrame(index=np.arange(len(epochs_filtered)))
        )
        for col in aligned_behavior.columns:
            out_col = col if col not in metadata.columns else f"beh_{col}"
            metadata[out_col] = aligned_behavior[col].to_numpy()
        # MNE logs an INFO line when replacing metadata; keep this step quiet.
        use_log_level = getattr(mne, "use_log_level", None)
        if use_log_level is None:
            epochs_filtered.metadata = metadata
        else:
            with use_log_level("ERROR"):
                epochs_filtered.metadata = metadata

        return epochs_filtered

    def _convert_edf_to_asc(self, subject_dir, subject_number):
        """Convert EyeLink EDF files to ASC files when ``edf2asc`` is available.

        Parameters
        ----------
        subject_dir : str | Path
            Raw-data folder for one subject.
        subject_number : str
            Subject label used to match files inside ``subject_dir``.

        Returns
        -------
        list[str]
            Names of ``.asc`` files now present in ``subject_dir`` after any
            conversion step.

        Raises
        ------
        FileNotFoundError
            If no ``.asc`` files exist and no matching ``.edf`` file is found.

        RuntimeError
            If ``edf2asc`` is available but conversion fails.

        Notes
        -----
        This helper keeps the workflow simple for the script user: if the raw
        export is still in EyeLink's binary ``.edf`` format, try converting it
        automatically before giving up.
        """
        asc_files = self._find_subject_files(subject_dir, subject_number, ".asc")
        if len(asc_files) > 0:
            return asc_files

        edf_files = self._find_subject_files(subject_dir, subject_number, ".edf")
        if len(edf_files) == 0:
            raise FileNotFoundError("No .asc or .edf file found for eyetracking import.")

        converter = self._find_edf2asc()
        if converter is None:
            error_msg = (
                "No .asc file was found for eyetracking import, and the EyeLink "
                '`edf2asc` converter is not installed or not on your PATH. '
                "Install `edf2asc`, or manually convert the .edf file to .asc and rerun preprocessing."
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)

        print("No .asc file found. Converting EyeLink .edf file(s) with edf2asc.")
        for edf_name in edf_files:
            edf_path = Path(subject_dir) / edf_name
            result = subprocess.run(
                [converter, str(edf_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            conversion_succeeded = (
                result.returncode == 0
                or self._edf2asc_created_output(edf_path)
                or self._edf2asc_reported_success(result)
            )
            if not conversion_succeeded:
                raise RuntimeError(
                    f"edf2asc failed for {edf_path.name}.\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

        asc_files = self._find_subject_files(subject_dir, subject_number, ".asc")
        if len(asc_files) == 0:
            raise FileNotFoundError(
                "edf2asc ran, but no .asc file was created in the subject directory."
            )
        return asc_files

    def _find_edf2asc(self) -> str | None:
        """Return the path to the EyeLink ``edf2asc`` converter if available.

        Parameters
        ----------
        None
            The method only checks the current environment.

        Returns
        -------
        str | None
            Full path to ``edf2asc`` when it is installed and visible on the
            shell PATH, otherwise ``None``.
        """
        return shutil.which("edf2asc")

    def _edf2asc_created_output(self, edf_path: Path) -> bool:
        """Check whether ``edf2asc`` wrote the expected ASCII file.

        Parameters
        ----------
        edf_path : Path
            Path to the source EyeLink EDF file.

        Returns
        -------
        bool
            ``True`` when an ``.asc`` file with the same stem exists in the
            same folder, otherwise ``False``.
        """
        return edf_path.with_suffix(".asc").exists()

    def _edf2asc_reported_success(self, result: subprocess.CompletedProcess) -> bool:
        """Check converter stdout for EyeLink success messages.

        Parameters
        ----------
        result : subprocess.CompletedProcess
            Result object returned by ``subprocess.run``.

        Returns
        -------
        bool
            ``True`` when stdout contains a known success message, otherwise
            ``False``.

        Notes
        -----
        Some Mac versions of ``edf2asc`` appear to finish the conversion but
        still return a non-zero exit code. We therefore treat the converter's
        own success text as an additional success signal.
        """
        stdout_lower = result.stdout.lower()
        success_markers = ["converted successfully", "conversion done"]
        return any(marker in stdout_lower for marker in success_markers)

    def _read_eyelink_ascii(self, asc_path):
        """Read an EyeLink ASCII file, with a fallback parser for noisy exports.

        Parameters
        ----------
        asc_path : str | Path
            Path to an EyeLink ``.asc`` file.

        Returns
        -------
        mne.io.BaseRaw
            Raw eye-tracking data with gaze and pupil channels plus message
            annotations that can be converted into events.

        Notes
        -----
        This project mainly needs the gaze samples and the task-message markers.
        The fallback parser reads those directly and is more robust for the
        EyeLink exports collected here. If that fallback ever fails, the method
        tries MNE's native EyeLink reader as a secondary option.
        """
        asc_path = Path(asc_path)
        try:
            return self._read_eyelink_ascii_fallback(asc_path)
        except ValueError as err:
            print(f"Fallback ASCII parser failed for {asc_path.name}. Trying MNE's native EyeLink reader.")
            print(err)
            return mne.io.read_raw_eyelink(
                asc_path,
                create_annotations=["blinks", "messages"],
                verbose="ERROR",
            )

    def _read_eyelink_ascii_fallback(self, asc_path):
        """Parse a simplified EyeLink ASCII export into an MNE ``RawArray``.

        Parameters
        ----------
        asc_path : str | Path
            Path to a cleaned EyeLink ``.asc`` file.

        Returns
        -------
        mne.io.RawArray
            Raw object containing left/right gaze and pupil channels.

        Raises
        ------
        ValueError
            If the file does not contain a usable sampling rate or any sample
            rows.

        Notes
        -----
        This fallback keeps the project readable and local to Python. It only
        extracts the pieces this project needs: binocular sample rows and task
        message markers whose last token is an integer event code.
        """
        asc_path = Path(asc_path)
        sfreq = None
        sample_rows = []
        event_onsets = []
        event_descs = []
        valid_codes = set()
        if self.event_names is not None:
            valid_codes = {str(code) for code in self.event_names.values()}

        with asc_path.open("r") as f:
            for line in f:
                stripped = line.strip()
                if stripped == "":
                    continue

                if stripped.startswith("SAMPLES"):
                    parts = stripped.split()
                    if "RATE" in parts:
                        sfreq = float(parts[parts.index("RATE") + 1])
                    continue

                if stripped.startswith("MSG"):
                    parts = stripped.split(maxsplit=2)
                    if len(parts) < 3:
                        continue
                    msg_time = int(parts[1])
                    msg_text = parts[2].strip()
                    msg_parts = msg_text.split()
                    if len(msg_parts) == 0:
                        continue
                    last_token = msg_parts[-1]
                    if valid_codes and last_token not in valid_codes:
                        continue
                    if not last_token.isdigit():
                        continue
                    event_onsets.append(msg_time)
                    event_descs.append(msg_text)
                    continue

                parts = stripped.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    try:
                        sample_rows.append(
                            [
                                int(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
                                float(parts[5]),
                                float(parts[6]),
                            ]
                        )
                    except ValueError:
                        continue

        if sfreq is None:
            raise ValueError(f"Could not find a sampling rate in {asc_path}.")
        if len(sample_rows) == 0:
            raise ValueError(f"Could not find any usable sample rows in {asc_path}.")

        sample_array = np.asarray(sample_rows, dtype=float)
        sample_times = sample_array[:, 0].astype(int)
        data = sample_array[:, 1:].T

        info = mne.create_info(
            ["xpos_left", "ypos_left", "pupil_left", "xpos_right", "ypos_right", "pupil_right"],
            sfreq=sfreq,
            ch_types=["eyegaze", "eyegaze", "pupil", "eyegaze", "eyegaze", "pupil"],
        )
        eye = mne.io.RawArray(data, info, verbose="ERROR")

        if len(event_onsets) > 0:
            event_ix = np.searchsorted(sample_times, np.asarray(event_onsets), side="left")
            event_ix = np.clip(event_ix, 0, len(sample_times) - 1)
            onsets_sec = event_ix / sfreq
            annotations = mne.Annotations(
                onset=onsets_sec,
                duration=np.zeros(len(onsets_sec), dtype=float),
                description=event_descs,
            )
            eye.set_annotations(annotations)

        return eye

    def import_eyetracker(self, subject_number, keyword=None, overwrite=False):
        """Load one subject's eye-tracking data and convert events to BIDS form.

        Parameters
        ----------
        subject_number : str
            Subject label used in the raw-data folder structure.
        keyword : str | None, optional
            Optional prefix used before sync messages in the EyeLink file. When
            provided, only messages starting with this prefix are converted into
            events.
        overwrite : bool, optional
            If ``False``, reuse previously exported BIDS-side eye-tracking files
            when available. If ``True``, rebuild them from the raw source files.

        Returns
        -------
        eye : mne.io.BaseRaw
            MNE raw object with the eye-tracking samples.
        eye_events : pandas.DataFrame
            Event table whose codes match the EEG event definitions used for
            syncing.

        Raises
        ------
        FileNotFoundError
            If no eye-tracking source file can be found.

        Notes
        -----
        The preferred input is an EyeLink ``.asc`` file because MNE reads that
        format directly. If only an EyeLink ``.edf`` file is present and the SR
        Research ``edf2asc`` converter is installed, this method will convert it
        automatically before import.
        """

        subject_dir = self._resolve_subject_dir(subject_number)
        eye_path = self._raw_file_path(subject_number, "eyetracking", "eyetracking", ".asc")
        eye_events_path = self._raw_file_path(subject_number, "eyetracking", "events", ".tsv")

        if not overwrite:
            try:
                asc_file = self._find_subject_files(eye_path.parent, subject_number, ".asc")
                if len(asc_file) == 0:
                    raise FileNotFoundError("No .asc file. I will try to re-import the data")
                elif len(asc_file) == 1:
                    eye = self._read_eyelink_ascii(eye_path)

                else:
                    print(
                        "More than 1 asc file present in subject directory. They will be concatenated in alphabetical order"
                    )
                    raws = []
                    for ifile, file in enumerate(asc_file):
                        file = os.path.join(eye_path.parent, file)
                        raws.append(self._read_eyelink_ascii(file))
                    eye = mne.concatenate_raws(raws, verbose="ERROR")

                # events should be already saved regardless
                eye_events = pd.read_csv(eye_events_path, sep="\t", index_col=False)

                return eye, eye_events

            except FileNotFoundError as e:
                print(e)
                print("Could not find Eyetracking data in your directory. I will try to import it from the raw data")

        eye_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the EyeLink ASCII file as-is. In this project the exported
        # eyetracker files already use the no-space format MNE can read.

        asc_file = self._convert_edf_to_asc(subject_dir, subject_number)

        if len(asc_file) == 1:
            asc_file = os.path.join(subject_dir, asc_file[0])
            shutil.copy2(asc_file, eye_path)

            # load in eye tracker data
            eye = self._read_eyelink_ascii(eye_path)

        else:  # more than one asc
            print("More than 1 asc file present in subject directory. They will be concatenated in alphabetical order")
            raws = []
            for ifile, file in enumerate(asc_file):
                file = os.path.join(subject_dir, file)
                ascpath = self._raw_file_path(subject_number, "eyetracking", "eyetracking", ".asc", split=ifile + 1)
                ascpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, ascpath)

                try:
                    raws.append(self._read_eyelink_ascii(ascpath))
                except ValueError as e:
                    print(
                        f"Error reading {ascpath}. This may be due to a bug in mne if your eyetracking file"
                        + "contains dropouts where the eye is lost. To fix this, manually re-generate the asc files"
                        + "with 'Block Flags Output' checked"
                    )
                    raise e
            eye = mne.concatenate_raws(raws, verbose="ERROR")

        keyword = "" if keyword is None else keyword
        regexp = f"^{keyword}(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$"
        et_events, et_event_dict = mne.events_from_annotations(eye, regexp=regexp, verbose="ERROR")

        # save sidecar
        self._copy_sidecar_template(self._raw_file_path(subject_number, "eyetracking", "eyetracking", ".json"), "TEMPLATE_eyetracking.json")
        self._warn_manual_sidecar_once()

        # convert events to match the EEG events

        et_events_dict_convert = {}
        for k, v in et_event_dict.items():
            new_k = int(k.split(" ")[-1])
            et_events_dict_convert[v] = new_k
        et_events_converted = et_events.copy()
        for code in et_events_dict_convert.keys():
            et_events_converted[:, 2][et_events[:, 2] == code] = et_events_dict_convert[code]

        # save events as TSV
        eye_events = pd.DataFrame(columns=["onset", "duration", "trial_type", "value", "sample"])
        eye_events["sample"] = et_events_converted[:, 0]
        eye_events["value"] = et_events_converted[:, 2]
        eye_events["onset"] = eye_events["sample"] / 1000
        event_names_inv = {v: k for k, v in self.event_names.items()}

        def get_events(trl):
            return event_names_inv[trl["value"]]

        eye_events["trial_type"] = eye_events.apply(get_events, axis=1)
        eye_events["duration"] = 0
        eye_events.to_csv(eye_events_path, sep=str("\t"), index=False)

        # sidecar events
        self._copy_sidecar_template(self._raw_file_path(subject_number, "eyetracking", "events", ".json"), "TEMPLATE_events.json")
        self._warn_manual_sidecar_once()

        return eye, eye_events

    def _convert_bids_events(self, events):
        """
        Converts a BIDS events file to a mne events array
        of the form [sample,0,value]
        """
        return events[["sample", "duration", "value"]].to_numpy().astype(int)

    def _filter_events(self, events):
        """Find trial sequences that match the configured event patterns.

        Parameters
        ----------
        events : np.ndarray
            MNE-style event array with columns ``[sample, duration, value]``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Timelock events and full matched event rows.

        Notes
        -----
        ``event_code_dict`` may map one trial code either to a single event
        sequence or to a list of allowed alternative sequences. The latter is
        useful when a trial can terminate early with ``TRIAL_REJECT`` at
        several possible points.
        """

        def get_sequence_options(sequence):
            """Return one or more allowed event sequences for a trial code."""
            if len(sequence) == 0:
                return []
            first_item = sequence[0]
            if isinstance(first_item, (list, tuple, np.ndarray)):
                return [list(option) for option in sequence]
            return [list(sequence)]

        class NoCode(Exception):
            pass

        # iterates through the list of event lists, requiring it to match the sequence in required_event_order
        # only picks out the first stimulus
        new_events = []
        new_times = []
        lock_events = []
        lock_times = []
        for i in range(len(events)):
            for code, sequence in self.event_code_dict.items():
                for sequence_option in get_sequence_options(sequence):
                    try:
                        for j, ev in enumerate(sequence_option):  # loop through the sequence and check if all events match
                            if events[i + j, 2] != ev:
                                raise NoCode
                    except NoCode:
                        continue  # not met, re loop
                    except IndexError:
                        break  # end of events array

                    else:
                        lock_events.append(code)  # good for eyetracking, only keep events we are locking to
                        lock_times.append(events[i + self.timelock_ix[code], 0])

                        cond_events = events[i : i + len(sequence_option), 2]
                        cond_events[self.timelock_ix[code]] = code  # replace the timelock event with the assigned code

                        new_events.append(cond_events)
                        new_times.append(events[i : i + len(sequence_option), 0])
                        break

        new_events = np.concatenate(new_events)
        new_times = np.concatenate(new_times)
        filtered_events = np.stack((new_times, np.zeros(len(new_times)), new_events), axis=1).astype(int)
        filtered_timelock_events = np.stack((lock_times, np.zeros(len(lock_times)), lock_events), axis=1).astype(int)

        return filtered_timelock_events, filtered_events

    def _make_metadata_from_events(self, events, srate):
        """Build trial-level metadata from matched event sequences.

        Parameters
        ----------
        events : np.ndarray
            MNE-style event array containing the matched trial sequences.
        srate : float
            Sampling rate used to convert event samples into times.

        Returns
        -------
        tuple[pandas.DataFrame, np.ndarray]
            Metadata table and the timelock event array returned by
            ``mne.epochs.make_metadata``.
        """

        sequence_values = []
        for sequence in self.event_code_dict.values():
            if len(sequence) == 0:
                continue
            first_item = sequence[0]
            if isinstance(first_item, (list, tuple, np.ndarray)):
                for option in sequence:
                    sequence_values.extend(option)
            else:
                sequence_values.extend(sequence)

        trial_keys = np.vectorize(self.event_dict_inv.get)(
            np.unique(np.asarray(sequence_values, dtype=int))
        )
        row_events = [self.event_dict_inv[k] for k in self.event_code_dict.keys()]

        metadata, metadata_events, meta_id = mne.epochs.make_metadata(
            events,
            self.event_dict,
            tmin=self.trial_start_t,
            tmax=self.trial_end_t * 2,  # double the time to account for delay end / long response
            sfreq=srate,
            row_events=row_events,
        )

        metadata = metadata[trial_keys]  # limit to events appearing in the trial

        return metadata, metadata_events
    
    def _get_tolerance(self,x, y):
        """
        helper function to find the first mismatch between two arrays of different lengths
        Args:
            x: shorter array
            y: longer array
        """
        mismatches = 0
        for itrial in range(len(x)):
            if x[itrial] != y[itrial]:
                mismatches += 1
        return mismatches
    def _find_extra_events(self, x, y, required_row_correct=10):
        """
        Helper function to identify sequences in one list that do not match the other.
        y is the longer list, and events will be removed form it until it matches x
        A better way to do this would use timestamps, but this is easier for now
        Args:
            x: shorter array
            y: longer array
            required_row_correct: how many events in a row must match to confirm the desync is fixed
        """
        reps = 0
        y_short = y.copy()
        events_to_delete = []
        while self._get_tolerance(x, y_short) > 0:
            reps += 1
            if reps > 20:
                raise RuntimeError("Could not find and fix all desyncs. Manual intervention is required.")
            # find the first point at which x and y differ
            for desync in range(len(x)):
                if x[desync] != y_short[desync]:
                    break
            for irm in range(abs(len(y) - len(x))):
                # check if removing irm events after the desync fixes the problem
                # defined as if the next required_row_correct events match
                tol = self._get_tolerance(
                    x[desync : desync + required_row_correct],
                    y_short[desync + irm + 1 : desync + irm + 1 + required_row_correct],
                )
                if tol == 0:
                    # events to delete based on index in original y list
                    mismatches = list(range(len(events_to_delete) + desync, len(events_to_delete) + desync + irm + 1))
                    events_to_delete.extend(mismatches)
                    y_short = np.delete(y, events_to_delete)
                    break
        return events_to_delete
    


    def _check_event_desyncs(self, eeg_events, eye_events, required_row_correct=10):
        """
        Check for desynchronization between EEG and eye-tracking events.
        Args:
            eeg_events: List of EEG events
            eye_events: List of eye-tracking events
            required_row_correct: Number of consecutive matching events required to confirm synchronization
            - only used if extra events are not at only the start or end
            note: this function assumes that the events are mostly synchronized, with only a few extra events in batches

        Returns:
            extra_eeg_events: Indices of extra EEG events to drop
            extra_eye_events: Indices of extra eye-tracking events to drop
        """

        if self._get_tolerance(eeg_events, eye_events) == 0:
            return [], []
        elif len(eeg_events) == len(eye_events):
            raise RuntimeError("Event lists are the same length but do not match. Manual intervention is required.")
        elif len(eeg_events) > len(eye_events):
            print("More EEG events than eye-tracking events. Checking for desyncs...")
            extra_eeg_events = len(eeg_events) - len(eye_events)

            # if extra eeg events at start remove those
            if self._get_tolerance(eeg_events[extra_eeg_events:], eye_events) == 0:
                print(f"Removing first {extra_eeg_events} extra EEG events at the start")
                return list(range(extra_eeg_events)), []
            # if extra eeg events at end remove those
            elif self._get_tolerance(eeg_events[:-extra_eeg_events], eye_events) == 0:
                print(f"Removing last {extra_eeg_events} extra EEG events at the end")
                return list(range(len(eeg_events) - extra_eeg_events, len(eeg_events))), []
            else:
                events_to_drop = self._find_extra_events(eye_events, eeg_events, required_row_correct)
                print(f"Removing extra EEG events at indices: {events_to_drop}")
                return events_to_drop, []
        elif len(eye_events) > len(eeg_events):
            print("More eye-tracking events than EEG events. Checking for desyncs...")
            extra_eye_events = len(eye_events) - len(eeg_events)
            # if extra eye events at start remove those
            if self._get_tolerance(eye_events[extra_eye_events:], eeg_events) == 0:
                print(f"Removing first {extra_eye_events} extra eye events at the start")
                return [], list(range(extra_eye_events))
            # if extra eye events at end remove those
            elif self._get_tolerance(eye_events[:-extra_eye_events], eeg_events) == 0:
                print(f"Removing last {extra_eye_events} extra eye events at the end")
                return [], list(range(len(eye_events) - extra_eye_events, len(eye_events)))
            else:
                events_to_drop = self._find_extra_events(eeg_events, eye_events, required_row_correct)
                print(f"Removing extra eye events at indices: {events_to_drop}")
                return [], events_to_drop
        else:
            raise RuntimeError("Could not identify desyncs. Manual intervention is required.")




    def make_eeg_epochs(self, eeg, eeg_events, eeg_trials_drop=None):
        """
        Function that handles epoching for EEG data. A replacement for make_and_sync_epochs when you don't want eyetracking

        Args:
            eeg: mne raw object containing EEG data
            eeg_events: events structure containing condition codes. These should match the eyetracking conditions
            eeg_trials_drop: trials to drop from EEG

        Returns:
            epochs: mne epochs object containing data
        """

        if eeg_trials_drop is None:
            eeg_trials_drop = []
        if self.srate is None:
            self.srate = eeg.info["sfreq"]

        # convert event dataframe to mne format (array of sample, duration, value)
        eeg_events = self._convert_bids_events(eeg_events)
        _, eeg_events = self._filter_events(eeg_events)
        metadata, metadata_events = self._make_metadata_from_events(eeg_events, eeg.info["sfreq"])

        # get EEG epochs object
        assert eeg.info["sfreq"] % self.srate == 0
        decim = eeg.info["sfreq"] / self.srate

        epochs = mne.Epochs(
            eeg,
            metadata_events,
            self.event_dict,
            tmin=self.trial_start_t,
            tmax=self.trial_end_t,
            on_missing="ignore",
            metadata=metadata,
            baseline=self.baseline_time,
            preload=True,
            decim=decim,
            verbose="ERROR",
        ).drop(
            eeg_trials_drop
        )  # set up epochs object
        if self.drop_channels is not None:
            epochs = epochs.drop_channels(self.drop_channels)

        return epochs

    def _make_and_prep_sync_of_epochs(
        self,
        eeg,
        eeg_events,
        eye,
        eye_events,
        eeg_trials_drop=None,
        eye_trials_drop=None,
    ):
        """
        Helper Function that does basic epoching
        converts EEG and eyetracking raw objects into epochs
        Returns the epochs for both EEG and eyetracking, along with their event codes, to confirm syncing

        Args:
            eeg: mne raw object containing EEG data
            eeg_events: events structure containing condition codes. These should match the eyetracking conditions
            eye: mne raw object containing eyetracking data
            eye_events: events structure containing condition codes. These should match the EEG conditions
            eeg_trials_drop: trials to drop from EEG
            eye_trials_drop: trials to drop from eyetracking

        Returns:
            eeg_epochs: mne epochs object containing eeg data
            eye_epochs: mne epochs object containing eye data
            eeg_events: events structure containing condition codes for EEG data
            eye_events: events structure containing condition codes for eye data

        """

        if eeg_trials_drop is None:
            eeg_trials_drop = []
        if eye_trials_drop is None:
            eye_trials_drop = []
        if self.srate is None:
            self.srate = eeg.info["sfreq"]

        # get our events list
        unmatched_codes = list(set(eeg_events["value"].unique()) ^ set(eye_events["value"].unique()))
        # convert event dataframe to mne format (array of sample, duration, value)
        eeg_events = self._convert_bids_events(eeg_events)
        eye_events = self._convert_bids_events(eye_events)
        eeg_events = eeg_events[~np.isin(eeg_events, unmatched_codes).any(axis=1)]
        eye_events = eye_events[~np.isin(eye_events, unmatched_codes).any(axis=1)]
        eeg_events_lock, eeg_events = self._filter_events(eeg_events)  # get all events
        eye_events, _ = self._filter_events(eye_events)  # only get events we timelock to


        extra_eeg_trials, extra_eye_trials = self._check_event_desyncs(eeg_events_lock[:, 2], eye_events[:, 2])
        eeg_trials_drop = list(set(eeg_trials_drop) | set(extra_eeg_trials))
        eye_trials_drop = list(set(eye_trials_drop) | set(extra_eye_trials))


        metadata, metadata_events = self._make_metadata_from_events(eeg_events, eeg.info["sfreq"])  # only from EEG data

        # get EEG epochs object
        assert eeg.info["sfreq"] % self.srate == 0
        decim = eeg.info["sfreq"] / self.srate

        eeg_epochs = mne.Epochs(
            eeg,
            metadata_events,
            self.event_dict,
            tmin=self.trial_start_t,
            tmax=self.trial_end_t,
            on_missing="ignore",
            metadata=metadata,
            baseline=self.baseline_time,
            preload=True,
            decim=decim,
            verbose="ERROR",
        ).drop(
            eeg_trials_drop
        )  # set up epochs object
        if self.drop_channels is not None:
            eeg_epochs = eeg_epochs.drop_channels(self.drop_channels)

        assert eye.info["sfreq"] % self.srate == 0
        decim = eye.info["sfreq"] / self.srate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"The measurement information indicates a low-pass frequency .* can cause aliasing artifacts\\.",
            )
            eye_epochs = mne.Epochs(
                eye,
                eye_events,
                self.event_dict,
                tmin=self.trial_start_t,
                tmax=self.trial_end_t,
                on_missing="ignore",
                baseline=self.baseline_time,
                reject=None,
                flat=None,
                reject_by_annotation=False,
                preload=True,
                decim=decim,
                verbose="ERROR",
            ).drop(eye_trials_drop)

        if "DIN" in eye_epochs.info["ch_names"]:
            eye_epochs.drop_channels(["DIN"])

        if len(eye_epochs) != len(
            eeg_epochs
        ):  # this happens if you abort recording mid trial. Trials should (normally) never be dropped
            dropped_eye_trials = [trial for trial, reason in enumerate(eye_epochs.drop_log) if len(reason) > 0]
            dropped_eeg_trials = [trial for trial, reason in enumerate(eeg_epochs.drop_log) if len(reason) > 0]
            print(
                f"WARNING: issue with trial count. EEG has {len(eeg_epochs)} trials, eyetracking has {len(eye_epochs)} trials\n.This is likely because you aborted the recording mid trial. If you did not do this, double check your event timings"
            )
            print(f"Dropping EEG trials: {dropped_eye_trials}")
            print(f"Dropping Eyetracking trials: {dropped_eeg_trials}")
            eeg_epochs.drop(dropped_eye_trials)
            eye_epochs.drop(dropped_eeg_trials)

        return eeg_epochs, eye_epochs, eeg_events, eye_events

    def make_and_sync_epochs(
        self,
        eeg,
        eeg_events,
        eye,
        eye_events,
        eeg_trials_drop=None,
        eye_trials_drop=None,
    ):
        """
        Function that does basic epoching
        converts EEG and eyetracking raw objects into epochs

        Args:
            eeg: mne raw object containing EEG data
            eeg_events: events structure containing condition codes. These should match the eyetracking conditions
            eye: mne raw object containing eyetracking data
            eye_events: events structure containing condition codes. These should match the EEG conditions
            eeg_trials_drop: trials to drop from EEG
            eye_trials_drop: trials to drop from eyetracking

        Returns:
            epochs: mne epochs object containing combined data

        """
        eeg_epochs, eye_epochs, _, _ = self._make_and_prep_sync_of_epochs(
            eeg, eeg_events, eye, eye_events, eeg_trials_drop, eye_trials_drop
        )

        try:
            epochs = eeg_epochs.copy()
            epochs.add_channels([eye_epochs], force_update_info=True)
        except ValueError as e:
            print(f"EEG has {len(eeg_epochs.info.ch_names)} channels and {len(eeg_epochs)} trials")
            print(f"Eyetracking has {len(eye_epochs.info.ch_names)} channels and {len(eye_epochs)} trials")
            raise e

        return epochs

    def _check_both_eyes(self, chan_labels, rej_chans):
        """
        If an artifact is found in eyetracking channels, ensures that it is found in both eyes
            chan_labels: list of all channel labels
            rej_chans: trials x channels matrix, boolean
        """
        if isinstance(chan_labels, list):
            chan_labels = np.array(chan_labels)
        if np.all(
            [eye_chan in chan_labels for eye_chan in ["xpos_right", "xpos_left", "ypos_right", "ypos_left"]]
        ):  # TODO: triple check this works
            x_chans = chan_labels[["xpos" in c for c in chan_labels]]
            y_chans = chan_labels[["ypos" in c for c in chan_labels]]

            rej_chans[:, np.isin(chan_labels, x_chans)] = rej_chans[:, np.isin(chan_labels, x_chans)].all(axis=1)[
                :, np.newaxis
            ]
            rej_chans[:, np.isin(chan_labels, y_chans)] = rej_chans[:, np.isin(chan_labels, y_chans)].all(axis=1)[
                :, np.newaxis
            ]
        return rej_chans

    def _get_data_from_rej_period(self, epochs):
        """
        grab the data from the rejection period, which might be smaller than epoch length
        """

        if self.rejection_time is not None:
            return epochs.get_data(copy=True)[
                :,
                :,
                np.logical_and(
                    epochs.times >= self.rejection_time[0],
                    epochs.times <= self.rejection_time[1],
                ),
            ]

        elif self.rejection_codes is not None:  # use rejection codes and return a masked array

            rejection_start_code = self.event_dict_inv[self.rejection_codes[0]]
            rejection_end_code = self.event_dict_inv[self.rejection_codes[1]]
            rejection_time_ix = ~np.logical_and(
                epochs.times[:, np.newaxis] > epochs.metadata[rejection_start_code].to_numpy(),
                epochs.times[:, np.newaxis] < epochs.metadata[rejection_end_code].to_numpy(),
            ).T  # trials x timepoints matrix of times to reject within. Inverted (because masking will mask these)

            epoch_data = epochs.get_data(copy=True)
            rejection_time_ix = np.broadcast_to(
                rejection_time_ix[:, np.newaxis, :], (epoch_data.shape)
            )  # broadcast to match shape. Yes this is necessary

            return np.ma.masked_array(epoch_data, mask=rejection_time_ix)

        else:
            return epochs.get_data(copy=True)

    def _conv_ms_to_samples(self, dur, epochs):
        """
        convert a duration in ms to timepoints
        """
        return int(np.floor(dur * epochs.info["sfreq"] / 1000))  # convert ms to timepoints

    def artreject_nan(self, epochs):
        """
        Rejects trials that contain any nan values
        """
        eegdata = self._get_data_from_rej_period(epochs)
        return np.any(np.isnan(eegdata), 2)

    def artreject_slidingP2P(self, epochs, rejection_criteria, win=200, win_step=100):
        """Reject channels whose peak-to-peak range exceeds a threshold.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data used for artifact checks.
        rejection_criteria : dict
            Mapping from channel type (for example ``"eeg"`` or
            ``"eyegaze"``) to a peak-to-peak threshold.
        win : int | str, optional
            Sliding-window size in milliseconds. Use ``"absolute"`` to apply
            the threshold across the full rejection interval.
        win_step : int, optional
            Step size between windows in milliseconds.

        Returns
        -------
        np.ndarray
            Boolean matrix with shape ``(n_trials, n_channels)`` marking
            channels that exceeded the threshold.

        Notes
        -----
        Some datasets in this project do not include every channel type listed
        in ``rejection_criteria``. Missing types are skipped so the script can
        use one clear configuration across subjects.
        """

        eegdata = self._get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())

        win = self._conv_ms_to_samples(win, epochs)
        win_step = self._conv_ms_to_samples(win_step, epochs)

        if isinstance(win, int):
            win_starts = np.arange(0, eegdata.shape[2] - win, win_step)
        elif win == "absolute":
            win_starts = [0]
            win = eegdata.shape[2]
        else:
            raise ValueError("win must be either an integer value or 'absolute'")

        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            if not np.any(chans):
                continue
            threshold = rejection_criteria[chan_type]
            for st in win_starts:
                data_min = np.nanmin(eegdata[:, chans, st : st + win], axis=2)
                data_max = np.nanmax(eegdata[:, chans, st : st + win], axis=2)
                reject = (data_max - data_min) > threshold
                if hasattr(reject, "mask"):
                    reject = reject.filled(False)  # fill masked values with False so they aren't rejected
                rej_chans[:, chans] = np.logical_or(rej_chans[:, chans], reject)

        # if both eyes are recorded, then ONLY mark artifacts if they appear in both eyes
        rej_chans = self._check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def artreject_value(self, epochs, rejection_criteria):
        """Reject channels whose absolute value exceeds a threshold.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data used for artifact checks.
        rejection_criteria : dict
            Mapping from channel type to an absolute-value threshold.

        Returns
        -------
        np.ndarray
            Boolean matrix with shape ``(n_trials, n_channels)`` marking
            channels that exceeded the threshold.

        Notes
        -----
        Missing channel types are skipped so the same artifact settings can be
        reused whether or not EOG channels were kept.
        """

        eegdata = self._get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            if not np.any(chans):
                continue
            threshold = rejection_criteria[chan_type]

            data_min = np.nanmin(eegdata[:, chans], axis=2)
            data_max = np.nanmax(eegdata[:, chans], axis=2)
            rej_chans[:, chans] = np.logical_or(data_max > threshold, data_min < -1 * threshold)

        # if both eyes are recorded, then ONLY mark artifacts if they appear in both eyes
        rej_chans = self._check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def artreject_step(self, epochs, rejection_criteria, win=80, win_step=10):
        """Reject channels that show abrupt step-like changes.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched data used for artifact checks.
        rejection_criteria : dict
            Mapping from channel type to the maximum allowed step size.
        win : int, optional
            Window size in milliseconds.
        win_step : int, optional
            Step size between windows in milliseconds.

        Returns
        -------
        np.ndarray
            Boolean matrix with shape ``(n_trials, n_channels)`` marking
            channels that exceeded the threshold.

        Notes
        -----
        Missing channel types are skipped so EEG-only and EEG-plus-eye-tracking
        datasets can share the same script code.
        """

        eegdata = self._get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())

        win = self._conv_ms_to_samples(win, epochs)
        win_step = self._conv_ms_to_samples(win_step, epochs)

        win_starts = np.arange(0, eegdata.shape[2] - win, win_step)
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            if not np.any(chans):
                continue
            threshold = rejection_criteria[chan_type]
            for st in win_starts:

                first_half = np.nanmean(eegdata[:, chans, st : st + win // 2], axis=2)
                last_half = np.nanmean(eegdata[:, chans, st + win // 2 : st + win], axis=2)
                reject = np.abs(first_half - last_half) > threshold
                if hasattr(reject, "mask"):
                    reject = reject.filled(False)  # fill masked values with False so they aren't rejected
                rej_chans[:, chans] = np.logical_or(rej_chans[:, chans], reject)

        # if both eyes are recorded, then ONLY mark artifacts if they appear in both eyes
        rej_chans = self._check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def artreject_linear(self, epochs, min_slope=75e-6, min_r2=0.3):
        """
        Rejects trials based on a linear regression fit

        Args:
            epochs: mne epochs object containing  data
            min_slope (int, optional): minimum slope to reject. Units are V/S
            min_r2 (float, optional): minimum r2 to reject at. Defaults to 0.3.
        """

        eegdata = self._get_data_from_rej_period(epochs)
        rej_linear = np.full((eegdata.shape[0:2]), False)  # assign to all channels first
        chans = np.array(epochs.info.get_channel_types()) == "eeg"  # TODO: make more flexible?
        eegdata = eegdata[:, chans, :]  # only keep EEG channels

        slopes = np.full((eegdata.shape[0:2]), 0, dtype=float)
        ssrs = np.full((eegdata.shape[0:2]), 0, dtype=float)

        for itrial in range(eegdata.shape[0]):
            trial_data = eegdata[itrial]
            if hasattr(trial_data, "mask"):
                trial_data = trial_data[
                    :, ~trial_data.mask.any(0)
                ].filled()  # take out masked timepoints bc don't play nice with linear regression
            xs = np.arange(trial_data.shape[1])
            A = np.vstack([xs, np.ones(len(xs))]).T
            (slopes[itrial], _), ssrs[itrial], _, _ = np.linalg.lstsq(A, trial_data.T, rcond=None)

        if hasattr(trial_data, "mask"):
            r2s = 1 - ssrs / (eegdata.shape[2] * eegdata.var(axis=2)).filled()  # double check value for n
        else:
            r2s = 1 - ssrs / (eegdata.shape[2] * eegdata.var(axis=2))  # double check value for n

        rej_linear[:, chans] = np.logical_and(r2s > min_r2, slopes > min_slope)
        return rej_linear

    def artreject_flatline(self, epochs, rejection_criteria, flatline_duration=200):
        """
        Rejects channels with flatline behavior (more than [flatline_duration] ms of the same value)
        You should probably only run this on EEG channels...
        Args:
            epochs: mne epochs object containing  data
            rejection_criteria: dict containing difference thresholds for each window
            flatline_duration: length of time in ms that a channel must be flat to be rejected
        Returns:
            rej_chans: indicates which electrodes match automated rejection criteria
        """

        duration = self._conv_ms_to_samples(flatline_duration, epochs)

        def get_flatline(non_flats, duration=duration):
            """
            function that finds at least [duration] subsequent timepoints of the same value
            Not pretty, but fairly optimized
            Args:
                non_flats: boolean array where True indicates a non-flat moment (large enough change in value b/w adjacent timepoints)
                duration (optional): number of subsequent timepoints that need to be flat
            Returns:
                boolean indicating if there is a flatline of at least [duration] timepoints
            """
            return np.any(  # see if the gap is bigger than duration at any timepoint
                np.diff(  # see if the gap between indices is bigger than the duration
                    np.where(  # find the indices of non-flat moments
                        np.concatenate(
                            # add True to the beginning and end of the array, to avoid unbounded runs of flats
                            ([True], non_flats, [True])
                        )
                    )[0]
                )
                >= duration
            )

        eegdata = self._get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            if np.sum(
                chans
            ):  # some datasets might not have all channel types (e.g. EOGs), which causes apply_along_axis to fail
                threshold = rejection_criteria[chan_type]

                diff = np.diff(eegdata[:, chans], axis=2)
                non_flats = (diff < -threshold) | (diff > threshold)
                if hasattr(non_flats, "mask"):

                    non_flats = non_flats.filled(
                        True
                    )  # converts masked values to True, so they are non considered flat

                rej_chans[:, chans] = rej_chans[:, chans] | np.apply_along_axis(
                    get_flatline, 2, non_flats
                )  # apply the function to each trial

        rej_chans = self._check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def deg2pix(self, eyeMoveThresh=1, distFromScreen=800, monitorWidth=532, screenResX=1920):
        """Converts degrees visual angle to a pixel value

        Args:
            eyeMoveThresh (int, optional): threshold (dva). Defaults to 1.
            distFromScreen (int, optional): distance from headrest to screen. Defaults to 900.
            monitorWidth (int, optional): Width of monitor. Defaults to 532.
            screenResX (int, optional): Monitor resolution. Defaults to 1920.

        Returns:
            pix: pixel value
        """

        pix_size_x = monitorWidth / screenResX
        mmfromfix = 2 * distFromScreen * np.tan(0.5 * np.deg2rad(eyeMoveThresh))
        pix = round(mmfromfix / pix_size_x)
        return pix

    def save_all_data(self, subject_number, epochs, rej_reasons, trial_qc=None):
        """Save preprocessed epochs and QC labels to the derivatives folder.

        Parameters
        ----------
        subject_number : str
            Subject number or label to save.
        epochs : mne.Epochs
            Preprocessed epochs. Any attached metadata are saved with the FIF
            file so trial-level QC labels remain available later.
        rej_reasons : np.ndarray
            Trial-by-channel string matrix. Non-empty cells mark channels with
            either hard rejection reasons or softer manual-review flags.
        trial_qc : pd.DataFrame | None, optional
            Optional trial-level QC table with one row per epoch. When
            provided, it is written as a sidecar TSV for downstream manual QC.
        """
        # file processing

        derivative_dir = self._derivative_eeg_dir(subject_number)
        derivative_dir.mkdir(parents=True, exist_ok=True)
        derivative_stem = self._derivative_file_stem(subject_number)

        def derivative_path(suffix: str, extension: str) -> Path:
            """Build one derivative file path from the shared stem."""
            return derivative_dir / f"{derivative_stem}_{suffix}{extension}"

        raw_events_json = self._raw_eeg_bids_path(subject_number, "events", ".json")
        raw_electrodes_tsv = self._raw_eeg_bids_path(subject_number, "electrodes", ".tsv", space="CapTrak")
        raw_eeg_json = self._raw_eeg_bids_path(subject_number, "eeg", ".json")

        # EVENTS

        events_final = pd.DataFrame(epochs.events, columns=["sample", "duration", "value"])

        def get_events(trl):
            return self.event_dict_inv[trl["value"]]

        events_final["trial_type"] = events_final.apply(get_events, axis=1)
        events_final["onset"] = events_final["sample"] / self.srate
        events_final = events_final[["onset", "duration", "trial_type", "value", "sample"]]

        events_final.to_csv(derivative_path("events", ".tsv"), sep="\t", index=False)

        # COPY EVENTS SIDECAR
        shutil.copy(raw_events_json, derivative_path("events", ".json"))

        # COPY ELECTRODES
        shutil.copy(raw_electrodes_tsv, derivative_path("electrodes", ".tsv"))

        # COPY SIDECAR AND CHANGE TO EPOCHED

        with open(raw_eeg_json) as f:
            sidecar_data = json.load(f)
        new_keys = {
            "RecordingType": "epoched",
            "EpochLength": round(self.trial_end_t - self.trial_start_t, 3),
            # Save the epoch sampling rate after any decimation so the
            # derivative metadata matches the saved epochs on disk.
            "SamplingFrequency": float(epochs.info["sfreq"]),
        }
        sidecar_data.update(new_keys)
        with open(derivative_path("eeg", ".json"), "w") as f:
            json.dump(sidecar_data, f, indent=4)

        ## BIDS NONCOMPLIANT FROM HERE ON. SUGGESTIONS WELCOME

        # Save one epochs file as the persistent EEG data source. Downstream
        # state transitions should be tracked in sidecar tables rather than a
        # second raw array export.
        epochs.save(derivative_path("epo", ".fif"), overwrite=True, verbose="ERROR")

        # ARTIFACTS
        pd.DataFrame(rej_reasons, columns=epochs.info["ch_names"]).to_csv(
            derivative_path("artifacts", ".tsv"), sep="\t", index=False
        )

        if trial_qc is not None:
            trial_qc.to_csv(derivative_path("trial_qc", ".tsv"), sep="\t", index=False)
