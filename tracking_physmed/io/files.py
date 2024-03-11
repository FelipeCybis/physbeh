from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

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
    df = pd.read_hdf(filename)
    metadata = pd.read_pickle(pkl_filename) if pkl_filename is not None else None

    motion_df = {}
    labels = []
    for c in df.columns:
        labels.append(c[1])
        col_name = "_".join(c[1:])
        motion_df[col_name] = df[c]

    dataframe = pl.from_dict(motion_df)
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

        dataframe = dataframe.with_columns(
            pl.col(c for c in dataframe.columns if "_x" in c) - corner_coords[0][0],
            pl.col(c for c in dataframe.columns if "_y" in c) - corner_coords[0][1],
        )
        dataframe = dataframe.with_columns(
            pl.col(c for c in dataframe.columns if "_likelihood" not in c)
            * per_px_ratio
        )

    track = Tracking(
        dataframe,
        fps=metadata["data"]["fps"],
        video_filename=video_filename,
        filename=filename,
    )

    return track


def _tsv2track(
    filename: Path,
    channel_tsv_filename: Path | None = None,
    video_filename: Path | None = None,
) -> Tracking:
    dataframe = pl.read_csv(filename, separator="\t")
    if channel_tsv_filename is not None:
        metadata = pl.read_csv(channel_tsv_filename, separator="\t")

    track = Tracking(
        data=dataframe,
        fps=metadata["sampling_frequency"][0],
        video_filename=video_filename,
        filename=filename,
    )
    return track


def load_tracking(filename, metadata_filename=None, video_filename=None):
    filename = _check_filename(filename=filename)
    metadata_filename = (
        _check_filename(metadata_filename) if metadata_filename is not None else None
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
