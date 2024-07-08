import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pint_pandas  # noqa: F401

from tracking_physmed.arena import RectangularArena
from tracking_physmed.tracking import Tracking, calculate_rectangle_cm_per_pixel


def _check_filename(
    filename: Path | str, label: str = "filename", is_file: bool = True
) -> Path:
    """Resolve filename full path and check it exists.

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename to check validity.
    label : str, optional
        Name of the variable passed to `_check_filename`, used in the error message.
        Default is "filename".
    is_file : bool, optional
        Whether or not to check if filename is an existing file. Default is ``True``.

    Returns
    -------
    pathlib.Path
        If successful, the filename resolved to its full path.

    Raises
    ------
    TypeError
        If `filename` cannot be cast to pathlib.Path.
    ValueError
        If `is_file` is ``True`` and `filename` does not exist.
    """
    try:
        filename = Path(filename)
    except TypeError as e:
        raise TypeError(
            f"{label} argument must be a pathlib.Path (or a type that supports"
            " casting to pathlib.Path, such as string)."
        ) from e

    filename = filename.expanduser().resolve()

    if is_file and not filename.is_file():
        raise ValueError(f"File not found: {filename}.")

    return filename


def _hdf52track(
    filename: Path, pkl_filename: Path | None = None, video_filename: Path | None = None
) -> Tracking:
    """Load ``.h5`` file (DeepLabCut output) into a `Tracking` instance.

    If `pkl_filename` is ``None``, the function will try to search for it in the same
    foldar as `filename`.

    Parameters
    ----------
    filename : pathlib.Path
        The ``.h5`` absolute file path.
    pkl_filename : pathlib.Path or None, optional
        The accompanying ``.pickle``. Default is ``None``.
    video_filename : pathlib.Path or None, optional
        The accompanying video file path. Default is ``None``.

    Returns
    -------
    Tracking
        The `Tracking` instance.
    """
    df = pd.read_hdf(filename)
    if pkl_filename is None:
        pkl_files = list(filename.parent.glob("*.pickle"))
        if len(pkl_files) != 1:
            warnings.warn(
                "Could not find exactly one pickle file containing behavioral metadata"
                f" in the same directory. Found [{len(pkl_files)}] *.pickle files.",
                UserWarning,
            )
        pkl_filename = pkl_files[0]
    metadata = pd.read_pickle(pkl_filename) if pkl_filename is not None else None

    motion_df = {}
    labels = []
    for c in df.columns:
        labels.append(c[1])
        col_name = "_".join(c[1:])
        motion_df[col_name] = df[c]

    dataframe = pd.DataFrame(motion_df)
    units = "pint[px]"
    per_px_ratio = 1.0
    if metadata is not None:
        corner_coords = metadata["data"]["corner_coords"]
        corner_coords = np.array(
            (
                corner_coords["top_left"],
                corner_coords["top_right"],
                corner_coords["bottom_left"],
                corner_coords["bottom_right"],
            )
        )
        per_px_ratio = calculate_rectangle_cm_per_pixel(corner_coords, 100, 100)

        x_columns = [col for col in dataframe.columns if col.endswith("_x")]
        dataframe[x_columns] -= corner_coords[0][0]
        y_columns = [col for col in dataframe.columns if col.endswith("_y")]
        dataframe[y_columns] -= corner_coords[0][1]

        xy_columns = [
            col for col in dataframe.columns if not col.endswith("_likelihood")
        ]
        dataframe[xy_columns] *= per_px_ratio
        units = "pint[cm]"

    for col in dataframe.columns:
        dataframe[col] = pd.Series(
            dataframe[col], dtype=units if not col.endswith("_likelihood") else "pint[]"
        )

    track = Tracking(
        dataframe,
        fps=50.0 if metadata is None else metadata["data"]["fps"],
        video_filename=video_filename,
        filename=filename,
    )
    track.space_units_per_pixel = per_px_ratio
    return track


def _tsv2track(
    filename: Path,
    channel_tsv_filename: Path | None = None,
    video_filename: Path | None = None,
) -> Tracking:
    """Load BIDS motion ``.tsv`` file into a `Tracking` instance.

    If `channel_tsv_filename` is ``None``, the function will try to search for the
    ``fps`` information in a ``.json`` sidecar or by trying to search for a matching
    ``_channels.tsv`` file.

    Parameters
    ----------
    filename : pathlib.Path
        The ``_motion.tsv`` absolute file path.
    channel_tsv_filename : pathlib.Path or None, optional
        The accompanying ``_channel.tsv``. Default is ``None``.
    video_filename : pathlib.Path or None, optional
        The accompanying video file path. Default is ``None``.

    Returns
    -------
    Tracking
        The `Tracking` instance.
    """
    dataframe = pd.read_csv(filename, sep="\t")
    if channel_tsv_filename is not None:
        metadata = pd.read_csv(channel_tsv_filename, sep="\t")
        fps = metadata["sampling_frequency"][0]
        units = metadata["units"]
        per_px = metadata["units/px"][0]
    else:
        fps, units, per_px = _try_get_fps_and_units(filename)
        if units is None:
            units = ["px"] * len(dataframe.columns)
            per_px = 1.0

    for col, unit in zip(dataframe.columns, units):
        unit = "pint[" + unit + "]" if "_likelihood" not in col else "pint[]"
        dataframe[col] = pd.Series(dataframe[col], dtype=unit)

    track = Tracking(
        data=dataframe,
        fps=50.0 if fps is None else fps,
        video_filename=video_filename,
        filename=filename,
    )

    if video_filename is not None:
        if video_filename.with_suffix(".json").is_file():
            with open(video_filename.with_suffix(".json")) as jf:
                video_sidecar = json.load(jf)
                corner_coords = np.array(
                    (
                        video_sidecar["PixelCoordinates"]["top_left"],
                        video_sidecar["PixelCoordinates"]["top_right"],
                        video_sidecar["PixelCoordinates"]["bottom_left"],
                        video_sidecar["PixelCoordinates"]["bottom_right"],
                    )
                )
        track.arena = RectangularArena(100, 100, "cm", *corner_coords)
        track.space_units_per_pixel = per_px
    return track


def _try_get_fps_and_units(filename: Path):
    maybe_bids = filename.stem.endswith("motion")
    if maybe_bids:
        channel_tsv_filename = filename.with_name(
            filename.name.replace("motion", "channels")
        )
        if channel_tsv_filename.exists():
            metadata = pd.read_csv(channel_tsv_filename, sep="\t")
            return (
                metadata["sampling_frequency"][0],
                metadata["units"],
                metadata["units/px"][0],
            )

    sidecar_path = filename.with_suffix(".json")
    if sidecar_path.exists():
        with open(sidecar_path) as fp:
            sidecar_motion = json.load(fp)
        if "SamplingFrequency" in sidecar_motion:
            return sidecar_motion["SamplingFrequency"], None, None


def load_tracking(
    filename: str | Path,
    metadata_filename: str | Path | None = None,
    video_filename: str | Path | None = None,
) -> Tracking:
    """Load tracking data.

    Parameters
    ----------
    filename : str or pathlib.Path
        Should resolve to a complete filename path. Either a ``.h5`` file directly from
        DeepLabCut output or a ``*_motion.tsv`` in a BIDS motion structure.
    metadata_filename : str or pathlib.Path, optional
        Should resolve to a complete filename path.

            * If `filename` is a ``.h5`` this should be a ``.pickle`` metadata filename.
            * If `filename` is a ``_motion.tsv`` this should be a ``*_channels.tsv``
              filename.

        If this is ``None``, we will use heuristics to try and find the associated
        metadata file. Default is ``None``.

    video_filename : str or pathlib.Path, optional
        Should resolve to a complete filename path. The ``.mp4`` file associated with
        the tracking data.

    Returns
    -------
    Tracking
        The `Tracking` instance.
    """
    filename = _check_filename(filename=filename)
    metadata_filename = (
        _check_filename(metadata_filename) if metadata_filename is not None else None
    )
    video_filename = (
        _check_filename(video_filename) if video_filename is not None else None
    )
    if filename.suffix == ".h5":
        track = _hdf52track(
            filename, pkl_filename=metadata_filename, video_filename=video_filename
        )
    elif filename.suffix == ".tsv":
        track = _tsv2track(
            filename,
            channel_tsv_filename=metadata_filename,
            video_filename=video_filename,
        )

    return track
