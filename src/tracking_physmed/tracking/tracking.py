import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal
from scipy.stats import multivariate_normal

from tracking_physmed.arena import BaseArena

from ..utils import (
    custom_2d_sigmoid,
    custom_sigmoid,
    get_gaussian_value,
    get_place_field_coords,
    get_value_from_hexagonal_grid,
    set_hexagonal_parameters,
)

SIGMOID_PARAMETERS = {
    "left": (-0.3, 20),
    "right": (0.3, 80),
    "top": (-0.3, 20),
    "bottom": (0.3, 80),
}


def to_tracking_time(arr, time, new_time, offset=0):
    """Returns upsampled array to match `new_time` array.

    Simply duplicates data to fill new array indices.

    Parameters
    ----------
    arr : _type_
        _description_
    time : _type_
        _description_
    new_time : _type_
        _description_
    offset : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    assert arr.shape[-1] == time.shape[-1], (
        "`arr` and `time` arguments must have the same length, but they are"
        f" {arr.shape[-1]} and {time.shape[-1]} respectively"
    )
    new_arr = np.zeros(arr.shape[:-1] + new_time.shape)
    fps = 1 / (new_time[1] - new_time[0])

    last_idx = 0
    for i, trig in enumerate(time):
        idx = min(int(np.ceil(trig * fps)), new_arr.shape[-1])
        arr_idx = min(i + offset, arr.shape[-1] - 1)

        new_arr[..., last_idx:idx] = np.repeat(
            arr[..., arr_idx][..., np.newaxis], idx - last_idx, axis=-1
        )
        last_idx = idx

    return new_arr


class Tracking:
    """A class to manipulate multi-label tracking data.


    Parameters
    ----------
    data : pandas.Dataframe
        The dataframe containing the tracking data.
    fps : float
        The frame per second parameter of the video used in the tracking.
    video_filename : str or pathlib.Path, optional
        The filename of the video associated with the tracking data.
    filename : str or pathlib.Path, optional
        The filename of the tracking data itself.
    """

    @property
    def video_filepath(self) -> Path | None:
        """Fullpath to labeled video of ``.h5`` file.

        Parameters
        ----------
        path : str or pathlib.Path
            File path to associated video.

        Returns
        -------
        pathlib.Path or None
            The fullpath to the labeled video if it exists.
        """
        return self._video_filepath

    @video_filepath.setter
    def video_filepath(self, str_path: str | Path):
        accepted_extensions = (".mp4", ".avi", ".mpg")
        if Path(str(str_path)).suffix not in accepted_extensions:
            self._video_filepath = None
            warnings.warn(
                f"Video file should have extensions {accepted_extensions} and not "
                f"{Path(str(str_path)).suffix}. Returning `NoneType`",
                category=UserWarning,
            )
        else:
            self._video_filepath = Path(str(str_path))

    @property
    def fps(self) -> float:
        """The frames per second from metadata of chosen analysis.

        Returns
        -------
        float
            The frames per second parameter.
        """
        return self._fps

    @property
    def space_units(self):
        """Units of coordinates in the dataframe."""
        return self.Dataframe.dtypes

    @property
    def space_units_per_pixel(self) -> float:
        """Per pixel ratio of the space units in the dataframe.

        Parameters
        ----------
        value : float
            The per pixel ratio.

        Returns
        -------
        float
            The space units per pixel ratio.
        """
        return self._space_units_per_pixel

    @space_units_per_pixel.setter
    def space_units_per_pixel(self, value: float):
        self._space_units_per_pixel = value

    @property
    def arena(self) -> BaseArena:
        return self._arena

    @arena.setter
    def arena(self, arena: BaseArena):
        self._arena = arena

    @property
    def time_units(self):
        """The units of the timing index ``.time``.

        Returns
        -------
        str
            The time units.
        """
        return "seconds"

    @property
    def time(self) -> npt.NDArray:
        """Timing index calculated using the ``fps`` property.

        Returns
        -------
        numpy.ndarray
            The timing array.
        """
        return np.arange(len(self.Dataframe)) / self.fps

    @property
    def pcutout(self) -> float:  # numpydoc ignore=PR02
        """The p-value cutout to consider an acceptable labeled frame.

        Parameters
        ----------
        value : int or float
            The p-value cutout.

        Returns
        -------
        float
            The p-value cutout.
        """
        return self._pcutout

    @pcutout.setter
    def pcutout(self, value: int | float):  # numpydoc ignore=GL08
        self._pcutout = float(value)

    @property
    def speed_smooth_window(self) -> npt.NDArray:  # numpydoc ignore=PR02
        """The gaussian window used to apply smoothing.

        Parameters
        ----------
        params : list or tuple of [float, float]
            The mean and sigma parameters for smoothing the speed data.

        Returns
        -------
        numpy.ndarray
            The gaussian window used to apply smoothing.
        """
        return self._speed_smooth_window

    @speed_smooth_window.setter
    def speed_smooth_window(
        self, params: tuple[float, float]
    ) -> None:  # numpydoc ignore=GL08
        assert (
            len(params) == 2
        ), "To set the gaussian window, a tuple with (length, std) should be passed."
        m, sigma = params
        self._speed_smooth_window = signal.windows.gaussian(M=m, std=sigma)
        self._speed_smooth_window /= sum(self._speed_smooth_window)

    @property
    def scan(self):
        """Related ``pythmed.scan.Scan`` object.

        Returns
        -------
        pythmed.scan.Scan
            The associated scan.
        """
        return self._scan

    def attach_scan(self, Scan):
        """Attach the corresponding Scan object to the Tracking.

        Parameters
        ----------
        Scan : pythmed.scan.Scan
            The associated scan.
        """
        self._scan = Scan

    def __init__(
        self,
        data: pd.DataFrame,
        fps: float,
        video_filename: Path | None = None,
        filename: Path | None = None,
    ):
        self.Dataframe = data

        self.labels = [
            label.replace("_x", "")
            .replace("_y", "")
            .replace("_z", "")
            .replace("_likelihood", "")
            for label in data.columns
        ]
        self.labels = sorted(set(self.labels), key=self.labels.index)

        self.filename = filename
        self._video_filepath = video_filename

        self.nframes = data.shape[0]
        self._fps = float(fps)

        self._arena = BaseArena()
        self._space_units_per_pixel = 1.0
        self._pcutout = 0.8
        self._speed_smooth_window = signal.windows.gaussian(M=101, std=6)
        self._speed_smooth_window /= sum(self._speed_smooth_window)

        self._scan = None

    def __repr__(self):
        infos = self.get_infos()
        return (
            f"Filename: {self.filename.name}\n"
            "-----------------------------------------------------------\n"
            f"Total time: {infos['total_time']:.2f} s\n"
            f"Time running: {infos['total_running_time']:.2f} s\n"
            f"Distance run: {infos['total_distance']:.2f} cm\n"
            f"Running ratio (running time / all time): {infos['running_ratio']:.2f}\n"
            "Exploration ratio (ratio of visited bins): "
            f"{infos['exploration_ratio']:.2f}\n"
            "Exploration std. (std of visits on each bin): "
            f"{infos['exploration_std']:.2f}\n"
            "Mean running speed (only running periods): "
            f"{infos['mean_running_speed']:.2f} cm/s\n"
            f"Mean speed: {infos['mean_speed']:.2f} cm/s\n"
            "-----------------------------------------------------------"
        )

    def set_ratio_coords(self, coord_list=[]):
        """Set the edge coordinates of a rectangle.

        If coord_list is not given, it calls Corner_Coords class GUI so the user can
        label the four corners and the Tracking class is able to calculate the ratio
        px/cm. It rights the corner coordinates in the metadata pickle file.

        Parameters
        ----------
        coord_list : list, optional
            Should be a list of ``[x, y]`` coordinates for the top left, top right,
            bottom left and bottom right corners such that ``coord_list = [[tl_x, tl_y],
            [tr_x, tr_y], ...]``. Default is ``[]``.
        """
        if coord_list:
            self._write_corner_coords(coord_list)
        else:
            from tracking_physmed.gui import Corner_Coords

            x_crop = self.metadata["data"]["cropping_parameters"][:2]
            y_crop = self.metadata["data"]["cropping_parameters"][2:]
            self.corner_coords = Corner_Coords(
                self.video_filepath,
                function_after_done=self._write_corner_coords,
                x_crop=x_crop,
                y_crop=y_crop,
            )

    def get_index(self, label: str, pcutout: float | None = None) -> npt.NDArray:
        """Get likelihood acceptability for `label` and threshold `pcutout`.

        Parameters
        ----------
        label : str
            The labels to extract the likelihood from.
        pcutout : float, optional
            Between 0 and 1. If `None`, uses the ``self.pcutout`` property. Default
            is ``None``.

        Returns
        -------
        array_like
            Array of ``True`` when ``index >= pcutout`` and ``False`` otherwise.
        """
        if pcutout is None:
            pcutout = self.pcutout

        return np.array(self.Dataframe[label + "_likelihood"] >= pcutout)

    def get_direction_array(
        self,
        label0: str = "neck",
        label1: str = "probe",
        mode: str = "degree",
        smooth: bool = False,
        only_running_bouts: bool = False,
    ):
        """Get direction vector ``'label0'->'label1'`` doing ``label1 - label0``.

        Default vector is ``'neck'->'probe'``. This can be used to get the head
        direction of the animal, for example.

        Parameters
        ----------
        label0 : str, optional
            Label where the vector will start. Default is ``'neck'``.
        label1 : str, optional
            Label where the vector will finish. Default is ``'probe'``.
        mode : str, optional
            Get direction data in ``"degrees"`` or ``"radians"``. Default is
            ``"degree"``.
        smooth : bool, optional
            Whether or not to smooth the direction data. Default is ``False``.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        direction array : numpy.ndarray
            In degrees, if mode asks for it, otherwise in radians.
        time_array : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where ``p-value > pcutout`` is ``True``, index is ``False`` otherwise.
        """

        hd_x, hd_y = self._get_vector_from_two_labels(label0=label0, label1=label1)
        index0 = self.get_index(label=label0)
        index1 = self.get_index(label=label1)
        index = index0 + index1

        resp_in_rad = np.arctan2(
            hd_y * (-1), hd_x
        )  # multiplication by -1 needed because of video x and y directions

        if smooth:
            resp_in_rad = np.unwrap(resp_in_rad)
            smooth_window = signal.windows.gaussian(M=101, std=10)
            smooth_window /= sum(smooth_window)

            resp_in_rad[~index] = 0
            resp_in_rad = np.convolve(smooth_window, resp_in_rad, "same")

            resp_in_rad = np.arctan2(np.sin(resp_in_rad), np.cos(resp_in_rad))

        resp_in_rad[resp_in_rad < 0] += 2 * np.pi
        resp = resp_in_rad
        if mode in ("deg", "degree"):
            resp = np.degrees(resp_in_rad)

        if only_running_bouts:
            resp_bouts = self._split_in_running_bouts(resp)
            index_bouts = self._split_in_running_bouts(index)

            return resp_bouts, self.time_bouts, index_bouts
        return resp, self.time, index

    def get_direction_angular_velocity(
        self, label0="neck", label1="probe", only_running_bouts=False
    ):
        """Get angular velocity using the vector between `label0` and `label1`.

        Parameters
        ----------
        label0 : str, optional
            Label where the vector will start. Default is ``'neck'``.
        label1 : str, optional
            Label where the vector will finish. Default is ``'probe'``.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        angular_velocity : numpy.ndarray
            The angular velocity array.
        time_array : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where ``p-value > pcutout`` is ``True``, index is ``False`` otherwise.
        """

        hd_x, hd_y = self._get_vector_from_two_labels(label0=label0, label1=label1)
        index = self.get_index(label=label1)

        resp_in_rad = np.arctan2(
            hd_y * (-1), hd_x
        )  # multiplication by -1 needed because of video x and y directions

        resp_in_rad = np.unwrap(
            resp_in_rad
        )  # unwrapping so smoothing and derivative can be done

        smooth_window = signal.windows.gaussian(M=101, std=10)
        smooth_window /= sum(smooth_window)
        resp_in_rad[~index] = 0
        resp_in_rad = np.convolve(smooth_window, resp_in_rad, "same")

        angular_velocity = np.gradient(resp_in_rad)

        if only_running_bouts:
            angular_velocity_bouts = self._split_in_running_bouts(angular_velocity)
            index_bouts = self._split_in_running_bouts(index)

            return angular_velocity_bouts, self.time_bouts, index_bouts

        return angular_velocity, self.time, index

    def get_degree_interval_hd(self, deg, only_running_bouts=False):
        """Get head direction array modulated by a gaussian function centered in `deg`.

        Parameters
        ----------
        deg : int or float
            Head direction in degrees, between 0 and 360.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        hd_array : numpy.ndarray
            The "activation" array for given `deg` head direction.
        time_array : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.
        """
        hd_deg, _, index = self.get_direction_array(
            label0="neck", label1="probe", mode="deg"
        )

        sigma = 10

        if deg < 60:
            hd_deg = np.where(hd_deg > 300, hd_deg - 360, hd_deg)

        if deg > 300:
            hd_deg = np.where(hd_deg < 60, hd_deg + 360, hd_deg)

        tmp = hd_deg - deg

        hd_array = get_gaussian_value(x=tmp, sigma=sigma)

        if only_running_bouts:
            hd_bouts = self._split_in_running_bouts(hd_array)
            index_bouts = self._split_in_running_bouts(index)
            return hd_bouts, self.time_bouts, index_bouts

        return hd_array, self.time, index

    def get_direction_histogram(
        self, bin_size: float = 4.0, label0: str = "neck", label1: str = "probe"
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the direction histogram between `label0` and `label1`.

        Parameters
        ----------
        bin_size : float, optional
            The size of the bins of the histogram, in degrees. Default is ``4.0``.
        label0 : str, optional
            Label where the vector will start. Default is ``"neck"``.
        label1 : str, optional
            Label where the vector will finish. Default is ``"probe"``.

        Returns
        -------
        hist : numpy.ndarray
            The values of the histogram.
        bin_edges : numpy.ndarray
            Return the bin edges ``(length(hist)+1)``.
        """

        bins = int(360 // bin_size)

        hd_deg, _, index = self.get_direction_array(
            label0=label0, label1=label1, mode="deg"
        )

        return np.histogram(hd_deg[index], bins=bins, range=(0, 360))

    def get_xy_coords(self, bodypart="body"):
        """Get array of ``x, y`` coordinates of the `bodypart`.

        Parameters
        ----------
        bodypart : str, optional
            The bodypart. Default is ``'body'``.

        Returns
        -------
        coords : numpy.ndarray
            Values in x and y multiplied by ``ratio_per_pixel`` for bodypart.
        time_array : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.
        """

        x, _, index = self.get_position_x(bodypart=bodypart, pcutout=self.pcutout)
        y, _, _ = self.get_position_y(bodypart=bodypart, pcutout=self.pcutout)

        coords = np.array([x, y]).T

        return coords, self.time, index

    def get_position_x(self, bodypart, pcutout=None):
        """Simple function to get x values for `bodypart`.

        Parameters
        ----------
        bodypart : str
            The bodypart.
        pcutout : float, optional
            Between 0 and 1. If ``None``, uses the ``self.pcutout`` property. Default is
            ``None``.

        Returns
        -------
        x_bp : numpy.ndarray
            Pixel values in x for bodypart.
        time_array : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.
        """
        index = self.get_index(bodypart, pcutout)
        if hasattr(self.Dataframe, "pint"):
            x_bp = self.Dataframe.pint.dequantify()[bodypart + "_x"].to_numpy()[:, 0]
        else:
            x_bp = self.Dataframe[bodypart + "_x"].to_numpy()

        return x_bp, self.time, index

    def get_position_y(self, bodypart, pcutout=None):
        """Simple function to get y values for bodypart.

        Parameters
        ----------
        bodypart : str
            The bodypart.
        pcutout : float, optional
            Between 0 and 1. If `None`, uses the self.pcutout property. Default is
            ``None``.

        Returns
        -------
        y_bp : numpy.ndarray
            Pixel values in y for bodypart.
        time_array : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.
        """

        index = self.get_index(bodypart, pcutout)
        if hasattr(self.Dataframe, "pint"):
            y_bp = self.Dataframe.pint.dequantify()[bodypart + "_y"].to_numpy()[:, 0]
        else:
            y_bp = self.Dataframe[bodypart + "_y"].to_numpy()
        return y_bp, self.time, index

    def get_speed(
        self,
        bodypart="body",
        axis="xy",
        euclidean_distance=False,
        smooth=True,
        speed_cutout=0,
        only_running_bouts=False,
    ):
        """Get speed for given ``bodypart``.

        When getting the distance between frames, the first index is hard set to be 0 so
        that the output array has the same length as the number of frames.

        Parameters
        ----------
        bodypart : str, optional
            Name of the label to get the speed from, by default 'body'.
        axis : str, optional
            To compute Vx, Vy or V, axis is ``'x'``, ``'y'`` or ``'xy'``, respectively.
            Default is ``'xy'``.
        euclidean_distance : bool, optional
            If ``axis`` is only one dimension, the distance can be the euclidean
            (absolute) or real. Default is ``False``.
        smooth : bool, optional
            If ``True`` a Gaussian window will convolve the speed array. The parameters
            of the Gaussian window can be set via the self.speed_smooth_window variable.
            Default is ``True``.
        speed_cutout : int, optional
            If given it will set the speed values under this threshold to 0. Default is
            ``0``.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        tuple
            - speed_array : (numpy.ndarray)
            - time_array : (numpy.ndarray)
            - index : (numpy.ndarray)
              Index where p-value > pcutout is True, index is False otherwise.
            - speed_units : (str)
              String telling the units of the speed_array
        """
        dist = self._get_distance_between_frames(
            bodypart=bodypart, axis=axis, euclidean=euclidean_distance
        )
        speed_array = dist * self.fps

        index = self.get_index(bodypart, self.pcutout)

        if smooth:
            speed_array[~index] = 0
            speed_array = np.convolve(self.speed_smooth_window, speed_array, "same")
            speed_array[np.abs(speed_array) < speed_cutout] = 0

        if only_running_bouts:
            speed_array = self._split_in_running_bouts(speed_array)
            index = self._split_in_running_bouts(index)
            return speed_array, self.time_bouts, index, "cm/s"

        return speed_array, self.time, index, "cm/s"

    def get_running_bouts(self, speed_array=None, time_array=None):
        """Get running bouts given certain parameters.

        Parameters used to determine running bouts are speed threshold,
        minimal duration of running bout and minimal duration of resting bout.

        Parameters
        ----------
        speed_array : array_like, optional
            Array to get automatic running bouts from, if ``None`` then
            `Tracking.get_speed()` is called. Default is ``None``.
        time_array : array_like, optional
            Time array to be passed as output. Default is ``None``.

        Returns
        -------
        running_bouts : numpy.ndarray
            Boolean array with True when the animal is running and False otherwise.
        time_array : numpy.ndarray
            Array mapping the index of the tracking data to time in seconds.
        """
        if speed_array is None or time_array is None:
            speed_array, time_array, _, _ = self.get_speed(bodypart="body", smooth=True)

        # getting True False array where speed is above 10 cm/s
        self.running_bouts = speed_array > 10
        # getting indices where this True False array changes from True to False or
        # False to True
        change_idx = np.where(np.diff(self.running_bouts))[0]
        # getting length from one change_idx to the next one
        bout_lengths = np.insert(np.diff(change_idx), 0, change_idx[0])

        # This for loop gets every False bout (no running) shorter than 15 seconds,
        # either in the beginning or between running bouts and sets them to True
        # (running)
        for i in range(len(bout_lengths)):
            if bout_lengths[i] < 750 and self.running_bouts[change_idx[i]] is False:
                if i == 0:
                    self.running_bouts[: change_idx[i] + 1] = True
                else:
                    self.running_bouts[change_idx[i - 1] : change_idx[i] + 1] = True

        # again, getting indices of change in the new running_bouts array
        temp_change_idx = np.where(np.diff(self.running_bouts))[0]
        # again, getting lengths from one temp_change_idx to the next one
        temp_bout_lengths = np.insert(np.diff(temp_change_idx), 0, temp_change_idx[0])

        # This for loop gets every True bout (running) shorter then 15 seconds,
        # either in the beginning or between no running bouts and sets them to False (no
        # running)
        for i in range(len(temp_bout_lengths)):
            if (
                temp_bout_lengths[i] < 750
                and self.running_bouts[temp_change_idx[i]] is True
            ):
                if i == 0:
                    self.running_bouts[: temp_change_idx[i] + 1] = False
                else:
                    self.running_bouts[
                        temp_change_idx[i - 1] : temp_change_idx[i] + 1
                    ] = False

        self.time_bouts = np.split(
            self.time[self.running_bouts],
            np.where(np.diff(np.where(self.running_bouts)[0]) > 1)[0] + 1,
        )

        return self.running_bouts, self.time_bouts

    def get_binned_position(
        self,
        bodypart: str = "body",
        bins: int | Sequence[int] = [10, 10],
        only_running_bouts: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get animal position in a 2D histogram.

        Parameters
        ----------
        bodypart : str, optional
            The specified bodypart. Default is ``"body"``.
        bins : int or [int, int], optional
            The bin specification:

            * If ``int``, the number of bins for the two dimensions (``nx=ny=bins``).
            * If ``[int, int]``, the number of bins in each dimension
                (``nx, ny = bins``).

        only_running_bouts : bool, optional
            Whether or not to use only running bouts. Default is ``False``.

        Returns
        -------
        H : numpy.ndarray, shape(nx, ny)
            The bi-dimensional histogram of samples `x` and `y`. Values in `x`
            are histogrammed along the first dimension and values in `y` are
            histogrammed along the second dimension.
        xedges : numpy.ndarray, shape(nx+1,)
            The bin edges along the first dimension.
        yedges : numpy.ndarray, shape(ny+1,)
            The bin edges along the second dimension.
        """

        x_pos, _, index = self.get_position_x(bodypart=bodypart)
        y_pos = self.get_position_y(bodypart=bodypart)[0]

        if only_running_bouts:
            if not hasattr(self, "running_bouts"):
                self.get_running_bouts()

            index = self.running_bouts

        return np.histogram2d(
            x_pos[index],
            y_pos[index],
            bins=bins,
            range=[[0, 100], [0, 100]],
        )

    def get_infos(self, bins: int = 10, bin_only_running_bouts: bool = False) -> dict:
        """Get general information about the tracking and return it in a dictionary.

        Parameters
        ----------
        bins : int, optional
            Number of bins to use when calculating the occupancy. Default is ``10``.
        bin_only_running_bouts : bool, optional
            Whether or not to use only running bouts when binning. Default is ``False``.

        Returns
        -------
        dict
            The dictionary containing general tracking information.
        """
        H = self.get_binned_position(
            "body", bins, only_running_bouts=bin_only_running_bouts
        )[0]

        info_dict = {}
        info_dict["total_time"] = self.nframes / self.fps

        speed = self.get_speed(bodypart="body", smooth=True)[0]
        speed_bouts = self.get_speed(
            bodypart="body", smooth=True, only_running_bouts=True
        )[0]
        bout_lenghts = [t_bout[-1] - t_bout[0] for t_bout in self.time_bouts]
        info_dict["total_running_time"] = sum(bout_lenghts)
        info_dict["total_distance"] = np.concatenate(speed_bouts).mean() * sum(
            bout_lenghts
        )
        info_dict["running_ratio"] = (
            info_dict["total_running_time"] / info_dict["total_time"]
        )
        info_dict["exploration_ratio"] = np.mean(H > 0)
        info_dict["exploration_std"] = np.std(H)
        info_dict["mean_running_speed"] = np.concatenate(speed_bouts).mean()
        info_dict["mean_speed"] = speed.mean()
        return info_dict

    def print_infos(self, bins: int = 10):
        """Print general informations of the tracking.

        Parameters
        ----------
        bins : int, optional
            Number of bins to use when calculating the occupancy. Default is ``10``.
        """
        info_dict = self.get_infos(bins=bins)
        spatial_units = self.space_units[self.labels[0] + "_x"].units
        print(
            "--------------------------------------------------------------\n"
            + f"Total tracking time: {info_dict['total_time']} {self.time_units}\n"
            + f"Total running time: {info_dict['total_running_time']:.2f}"
            + f" {self.time_units}\n"
            + f"Total distance run: {info_dict['total_distance']:.2f} {spatial_units}\n"
            + "Running time ratio (running time / all time): "
            f"{info_dict['running_ratio']:.2f}\n"
            + "Exploration ratio (ratio of visited bins): "
            f"{info_dict['exploration_ratio']:.3f}\n"
            + "Exploration std (std of visits on each bin): "
            f"{info_dict['exploration_std']:.3f}\n"
            + "Running speed (only running periods): "
            f"{info_dict['mean_running_speed']:.2f} {spatial_units}/{self.time_units}\n"
            + f"Mean running speed: {info_dict['mean_speed']:.2f} "
            f"{spatial_units}/{self.time_units}\n"
            "--------------------------------------------------------------"
        )

    def get_proximity_from_wall(
        self,
        wall="left",
        bodypart="probe",
        only_running_bouts=False,
    ):
        """Get a sigmoid activation for the label in relation to the specified wall.

        Parameters
        ----------
        wall : str or list of str or tuple of str, optional
            Wall to use for computations. Can be one of ("left", "right", "top",
            "bottom"). Default is ``"left"``.
        bodypart : str, optional
            Bodypart to use for computations. Default is ``"probe"``.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        tuple
            Tuple of ``wall_activation``, time and likelihood indices.

        Raises
        ------
        ValueError
            If wall parameter is not on of the possibilities to choose from.
        """
        if wall == "all":
            wall = ("left", "right", "top", "bottom")

        if not isinstance(wall, list | tuple):
            wall = [wall]

        if not set(wall).issubset(("left", "right", "top", "bottom")):
            raise ValueError(
                "wall parameter must be one of the following: left, right, top or"
                f" bottom, not {wall}."
            )

        coords, _, index = self.get_xy_coords(bodypart=bodypart)

        subset_wall_activation = []
        for w in wall:
            a, b = SIGMOID_PARAMETERS[w]
            if w in ("left", "right"):
                pos = coords[:, 0]
            elif w in ("top", "bottom"):
                pos = coords[:, 1]

            subset_wall_activation.append(custom_sigmoid(pos, a=a, b=b))

        wall_activation = np.max(subset_wall_activation, axis=0)

        if only_running_bouts:
            wall_activation = self._split_in_running_bouts(wall_activation)
            index = self._split_in_running_bouts(index)
            return wall_activation, self.time_bouts, index

        return wall_activation, self.time, index

    def get_proximity_from_center(self, bodypart="probe", only_running_bouts=False):
        """Get a sigmoid activation for the label in relation to the center.

        This is the inverse of getting proximity from wall for all walls.

        Parameters
        ----------
        bodypart : str, optional
            Bodypart to use for computations. Default is ``"probe"``.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        tuple
            Tuple of ``center_activation``, time and likelihood indices.
        """
        coords, _, index = self.get_xy_coords(bodypart=bodypart)

        subset_wall_activation = []
        for w in ("left", "right", "top", "bottom"):
            a, b = SIGMOID_PARAMETERS[w]
            if w in ("left", "right"):
                pos = coords[:, 0]
            elif w in ("top", "bottom"):
                pos = coords[:, 1]

            subset_wall_activation.append(custom_sigmoid(pos, a=-a, b=b))

        center_activation = np.min(subset_wall_activation, axis=0)

        if only_running_bouts:
            center_activation = self._split_in_running_bouts(center_activation)
            index = self._split_in_running_bouts(index)
            return center_activation, self.time_bouts, index

        return center_activation, self.time, index

    def get_proximity_from_corner(
        self, corner="top right", bodypart="probe", only_running_bouts=False
    ):
        """Get a 2D sigmoid activation for the label in relation to the `corner`.

        Parameters
        ----------
        corner : str, optional
            Must be one of the four corners of a rectangle ("top right", "top left",
            "bottom right", "bottom left"). Default is ``"top right"``.
        bodypart : str, optional
            Bodypart to use for computations. Default is ``"probe"``.
        only_running_bouts : bool, optional
            Use only running bouts of the experiment. Default is ``False``.

        Returns
        -------
        tuple
            Tuple of ``corner_activation``, time and likelihood indices.

        Raises
        ------
        ValueError
            If corner parameter is not on of the possibilities to choose from.
        """
        if corner == "all":
            corner = ("top right", "top left", "bottom right", "bottom left")

        if not isinstance(corner, list | tuple):
            corner = [corner]

        if not set(corner).issubset(
            ("top right", "top left", "bottom right", "bottom left")
        ):
            raise ValueError(
                "corner parameter must be among the following: top right, top left, "
                f"bottom right, bottom left, not {corner}."
            )

        coords, _, index = self.get_xy_coords(bodypart=bodypart)

        subset_corner_activation = []
        for c in corner:
            ax, bx = SIGMOID_PARAMETERS[c.split()[1]]
            ay, by = SIGMOID_PARAMETERS[c.split()[0]]
            subset_corner_activation.append(
                custom_2d_sigmoid(
                    x=coords[:, 0],
                    ax=ax,
                    bx=bx,
                    y=coords[:, 1],
                    ay=ay,
                    by=by,
                )
            )

        corner_activation = np.max(subset_corner_activation, axis=0)

        if only_running_bouts:
            corner_activation = self._split_in_running_bouts(corner_activation)
            index = self._split_in_running_bouts(index)
            return corner_activation, self.time_bouts, index

        return corner_activation, self.time, index

    def get_place_field_array(
        self,
        coords: tuple | None = None,
        random_coords=False,
        only_running_bouts=False,
        bodypart="body",
    ):
        """Get a place field activation array on specific ``x, y`` coordinate.

        Parameters
        ----------
        coords : tuple, optional
            (x, y) coordinates to create the place field. If `None`, gets coordinates
            from utils.place_fields.get_place_field_coordinates. Default is ``None``.
        random_coords : bool, optional
            If `coords` is `None`, gets coordinates using the `random` parameter or not.
            Default is ``False``.
        only_running_bouts : bool, optional
            Not yet implemented. use only running bouts of the experiment. Default is
            ``False``.
        bodypart : str, optional
            Which bodypart label to use when getting coordinates. Default is ``"body"``.

        Returns
        -------
        tuple (place field array, (time array, place field coordinates))
            Each line in the place field array corresponds to one place field
            coordinate.
        """

        animal_coords, time_array, index = self.get_xy_coords(bodypart=bodypart)
        sigma = 300

        self.place_fields_list = list()
        if coords is None:
            coords = get_place_field_coords(random=random_coords)
        else:
            coords = [coords]

        for coord in coords:
            rv = multivariate_normal(
                mean=[coord[0], coord[1]], cov=[[sigma, 0], [0, sigma]]
            )
            self.place_fields_list.append(rv.pdf(animal_coords))

        self.place_fields_array = np.array(self.place_fields_list)
        return self.place_fields_array, (self.time, coords)

    def get_grid_field_array(
        self,
        params: list[tuple[float, float, float]] | None = None,
        bodypart: str = "body",
    ) -> tuple[npt.NDArray, npt.NDArray, list[tuple[float, float, float]]]:
        """Get a grid activation array using the parameters `params`.

        Parameters
        ----------
        params : list of tuple of (float, float, float) or None, optional
            List of hexagonal parameters to use when getting the grid activations.
            Default is ``None``.
        bodypart : str, optional
            The specified bodypart. Default is ``"body"``.

        Returns
        -------
        grid_fields_array : numpy.ndarray
            Grid field activation array.
        time : numpy.ndarray
            Time array in seconds.
        index : numpy.ndarray
            Index where p-value > pcutout is True, index is False otherwise.
        """

        animal_coords = self.get_xy_coords(bodypart=bodypart)[0]

        if params is None:
            params = set_hexagonal_parameters()

        self.grid_fields_list = list()
        for param in params:
            [xplus, a, angle] = param
            # for pos in animal_coords:
            z = get_value_from_hexagonal_grid(
                animal_coords, xplus=xplus, a=a, angle=angle
            )
            self.grid_fields_list.append(z)

        self.grid_fields_array = np.array(self.grid_fields_list)

        return self.grid_fields_array, self.time, params

    def _split_in_running_bouts(self, array):
        if not hasattr(self, "running_bouts"):
            self.get_running_bouts()

        return np.split(
            array[self.running_bouts],
            np.where(np.diff(np.where(self.running_bouts)[0]) > 1)[0] + 1,
        )

    def _get_distance_between_frames(
        self, bodypart="body", axis="xy", euclidean=False, backup_bps=["probe"]
    ):
        """Get distance from one frame to another for the specific bodypart.

        Parameters
        ----------
        bodypart : str, optional
            The default is 'body'.
        axis : str, optional
            To compute Vx, Vy or V, axis is ``'x'``, ``'y'`` or ``'xy'``, respectively.
            Default is ``'xy'``.
        euclidean : bool, optional
            If ``axis`` is only one dimension, the distance can be the euclidean
            (absolute) or real. Default is ``False``.

        Returns
        -------
        numpy.ndarray
            Distance between frames. First values is set to ``0`` so that the returned
            array has the same size of ``self.nframes``.
        """
        if axis is None:
            axis = "xy"

        coords = []
        if axis in ("x", "xy"):
            coords.append(self.get_position_x(bodypart=bodypart)[0])

        if axis in ("y", "xy"):
            coords.append(self.get_position_y(bodypart=bodypart)[0])

        coords = np.stack(coords)
        if len(axis) == 1 and not euclidean:
            distance = np.diff(coords)
        else:
            distance = np.sum(np.diff(coords) ** 2, axis=0) ** 0.5

        return np.insert(distance, 0, 0)

    def _get_vector_from_two_labels(self, label0, label1):
        """Gets the vector 'label0'->'label1' by simple subtraction label1 - label0.

        Parameters
        ----------
        label0 : str
            Label where the vector will start.
        label1 : str
            Label where the vector will finish.

        Returns
        -------
        vec_x : numpy.ndarray
            Vector distance in the x coordinate. label1_x - label0_x
        vec_y : numpy.ndarray
            Vector distance in the y coordinate. label1_y - label0_y
        """

        if hasattr(self.Dataframe, "pint"):
            vec_x = (
                self.Dataframe.pint.dequantify()[label1 + "_x"]
                - self.Dataframe.pint.dequantify()[label0 + "_x"]
            )
            vec_y = (
                self.Dataframe.pint.dequantify()[label1 + "_y"]
                - self.Dataframe.pint.dequantify()[label0 + "_y"]
            )
        else:
            vec_x = self.Dataframe[label1 + "_x"] - self.Dataframe[label0 + "_x"]
            vec_y = self.Dataframe[label1 + "_y"] - self.Dataframe[label0 + "_y"]

        return vec_x.to_numpy(), vec_y.to_numpy()


def calculate_rectangle_cm_per_pixel(
    coords: npt.ArrayLike, real_width_cm: int | float, real_height_cm: int | float
):
    """Helper function to calculate the centimeter per pixel ratio in a rectangle.

    Parameters
    ----------
    coords : npt.ArrayLike
        ``xy`` coordinates of the rectangle in the order [({top_left_x},
        {top_left_y}), ({top_right_x}, {top_right_y}), ({bottom_left_x},
        {bottom_left_y}), ({bottom_right_x}, {bottom_right_y})].
    real_width_cm : int | float
        Rectangle width in centimeters.
    real_height_cm : int | float
        Rectangle height in centimeters.

    Returns
    -------
    float
        The averaged centimeter per pixel ratio.
    """

    def _distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    width_estimates = [_distance(coords[0], coords[1]), _distance(coords[2], coords[3])]
    height_estimates = [
        _distance(coords[0], coords[2]),
        _distance(coords[1], coords[3]),
    ]
    diag_estimates = [_distance(coords[0], coords[3]), _distance(coords[1], coords[2])]

    real_diag_cm = np.sqrt(real_width_cm**2 + real_height_cm**2)
    cm2px_ratio_width = real_width_cm / np.mean(width_estimates)
    cm2px_ratio_height = real_height_cm / np.mean(height_estimates)
    cm2px_ratio_diag = real_diag_cm / np.mean(diag_estimates)

    return np.mean((cm2px_ratio_diag, cm2px_ratio_height, cm2px_ratio_width))
