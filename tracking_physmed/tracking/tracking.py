import pickle, warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import multivariate_normal

from tracking_physmed.utils import (
    get_cmap,
    get_gaussian_value,
    get_rectangular_value,
    custom_sigmoid,
    custom_2d_sigmoid,
    get_value_from_hexagonal_grid,
    set_hexagonal_parameters,
    get_place_field_coords,
)

SIGMOID_PARAMETERS = {
    "left": (-0.3, 30),
    "right": (0.3, 70),
    "top": (-0.3, 30),
    "bottom": (0.3, 70),
}


def load_tracking(filename, metadata_filename=None, video_filename=None):
    try:
        filename = Path(filename)
    except TypeError:
        raise TypeError(
            "filename argument must be a pathlib.Path (or a type that supports"
            " casting to pathlib.Path, such as string)."
        )

    filename = filename.expanduser().resolve()

    if not filename.is_file():
        raise ValueError(f"File not found: {filename}.")

    assert (
        filename.suffix == ".h5"
    ), f"Accepted filename needs to have extension `.h5` and not `{filename.suffix}`"

    Dataframe = pd.read_hdf(filename)

    if metadata_filename is None:
        metadata_filename = filename.with_name(
            filename.name.replace("filtered", "meta").replace("h5", "pickle")
        )

    if not metadata_filename.is_file():
        raise ValueError(f"Metadata file not found: {metadata_filename}")

    if video_filename is None:
        video_filename = filename.with_name(
            filename.name.replace("tracking-filtered", "recording-labeled").replace(
                "beh.h5", "video.mp4"
            )
        )

    if not video_filename.is_file():
        video_filename = None
        # !TODO raise warning to say there is no video

    track = Tracking(Dataframe, metadata_filename, video_filename)
    track.filename = filename

    return track


def to_tracking_time(arr, time, new_time, offset=0):
    """Returns upsampled array to match `new_time` array. Simply duplicates data to fill new array indices.

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
    assert (
        arr.shape[-1] == time.shape[-1]
    ), f"`arr` and `time` arguments must have the same length, but they are {arr.shape[-1]} and {time.shape[-1]} respectively"
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


class Tracking(object):
    @property
    def tracking_filepath(self):
        return self._tracking_filepath

    @property
    def tracking_directory(self):
        return self._tracking_directory

    @property
    def video_filepath(self):
        return self._video_filepath

    @video_filepath.setter
    def video_filepath(self, str_path):
        accepted_extensions = (".mp4", ".avi", ".mpg")
        if Path(str(str_path)).suffix not in accepted_extensions:
            self._video_filepath = None
            warnings.warn(
                f"Video file should have extensions {accepted_extensions} and not {Path(str(str_path)).suffix}. Returning `NoneType`",
                category=UserWarning,
            )
        else:
            self._video_filepath = Path(str(str_path))

    @property
    def fps(self):
        """Frames per second from metadata of chosen analysis."""
        return self._fps

    @property
    def time(self):
        """Timing index calculated using the `fps` property."""
        return self._time

    @property
    def pcutout(self):
        return self._pcutout

    @pcutout.setter
    def pcutout(self, value):
        self._pcutout = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        self._h = value

    @property
    def ratio_per_pixel(self):
        return self._ratio_per_pixel

    @property
    def spatial_units(self):
        return self._spatial_units

    @spatial_units.setter
    def spatial_units(self, units):

        assert units in (
            "mm",
            "cm",
            "m",
            "px",
        ), f"Units can only be one of these strings: 'mm', 'cm', 'm' or 'px'. It is '{units}'"
        if units == "px":
            self._ratio_per_pixel = 1
        else:
            assert (
                self.ratio_cm_per_pixel is not None
            ), "The cm/px ratio has not yet been set."
            ratio_factor_dict = {"mm": 10, "cm": 1, "m": 0.1}
            self._ratio_per_pixel = self.ratio_cm_per_pixel * ratio_factor_dict[units]

        self._spatial_units = units

    @property
    def speed_smooth_window(self):
        return self._speed_smooth_window

    @speed_smooth_window.setter
    def speed_smooth_window(self, params):
        assert (
            len(params) == 2
        ), "To set the gaussian window, a tuple with (lenght, std) should be passed."
        m, sigma = params
        self._speed_smooth_window = signal.gaussian(M=m, std=sigma)
        self._speed_smooth_window /= sum(self._speed_smooth_window)

    @property
    def scan(self):
        return self._scan

    def attach_scan(self, Scan):
        self._scan = Scan

    def __init__(self, data, metadata_filename, video_filename):

        self.Dataframe = data
        self.metadata = pd.read_pickle(metadata_filename)
        self.metadata_filename = metadata_filename
        self._video_filepath = video_filename

        self.scorer = self.metadata["data"]["Scorer"]
        self.nframes = data.shape[0]
        self.bodyparts = self.metadata["data"]["DLC-model-config file"][
            "all_joints_names"
        ]
        self._fps = self.metadata["data"]["fps"]
        self._time = np.array(data.index / self.fps)
        self.colormap = "plasma"
        colors = get_cmap(len(self.bodyparts), name=self.colormap)
        self.colors = {self.bodyparts[i]: colors(i) for i in range(len(self.bodyparts))}
        self._pcutout = 0.8
        self._speed_smooth_window = signal.gaussian(M=101, std=6)
        self._speed_smooth_window /= sum(self._speed_smooth_window)

        try:
            # checking if coordinates for corners have already been set
            # if not, calls function to do it
            self.metadata["data"]["corner_coords"]
        except KeyError:
            self.set_ratio_coords()

        self._w = 100
        self._h = 100
        self._get_cm2px_ratio()
        self._spatial_units = "cm"
        self._ratio_per_pixel = self.ratio_cm_per_pixel
        # self._downsample

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
            f"Exploration ratio (ratio of visited bins): {infos['exploration_ratio']:.2f}\n"
            f"Exploration std. (std of visits on each bin): {infos['exploration_std']:.2f}\n"
            f"Mean running speed (only running periods): {infos['mean_running_speed']:.2f} cm/s\n"
            f"Mean speed: {infos['mean_speed']:.2f} cm/s\n"
            "-----------------------------------------------------------"
        )

    def manual_relabel(self):
        from tracking_physmed.gui import Manual_relabel

        return Manual_relabel(
            self.tracking_filepath, self.metadata_filepath, self.video_filepath
        )

    def set_ratio_coords(self, coord_list=[], type="rectangle"):
        """If coord_list is not given, it calls Corner_Coords class GUI so the user can label
        the four corners and the Tracking class is able to calculate the ratio px/cm.
        It rights the corner coordinates in the metadata pickle file.

        Parameters
        ----------
        coord_list : list, optional
            Should be a list of [x, y] coordinates for the top left, top right, bottom left and
            bottom right corners such that coord_list = [[tl_x, tl_y], [tr_x, tr_y], ...], by default []
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


    def get_index(self, label, pcutout=None):
        """Gets likelihood indices for `label` and threshold `pcutout`. Returns an array of booleans with True when index >= pcutout and False otherwise.

        Parameters
        ----------
        label : str
        pcutout : float, optional
            Between 0 and 1. If `None`, uses the self.pcutout property. The default is `None`

        Returns
        -------
        array_like
            Array of Trues when index >= pcutout and False otherwise
        """
        if pcutout is None:
            pcutout = self.pcutout

        return self.Dataframe[self.scorer][label]["likelihood"].values >= pcutout

    def get_direction_array(
        self, label0="neck", label1="probe", mode="deg", smooth=False
    ):
        """Gets the direction vector 'label0'->'label1' by simple subtraction label1 - label0.
        Default vector is 'neck'->'probe'. This can be used to get the head direction of the animal, for example.

        Parameters
        ----------
        label0 : str, optional
            Label where the vector will start. The default is 'neck'.
        label1 : str, optional
            Label where the vector will finish. The default is 'probe'.

        Returns
        -------
        direction array : numpy.ndarray
            In degrees, if mode asks for it, otherwise in radians.

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
            smooth_window = signal.gaussian(M=101, std=10)
            smooth_window /= sum(smooth_window)

            resp_in_rad[~index] = 0
            resp_in_rad = np.convolve(smooth_window, resp_in_rad, "same")

            resp_in_rad = np.arctan2(np.sin(resp_in_rad), np.cos(resp_in_rad))

        resp_in_rad[resp_in_rad < 0] += 2 * np.pi
        if mode in ("deg", "degree"):
            return np.degrees(resp_in_rad), index
        return resp_in_rad, index

    def get_direction_angular_velocity(
        self, label0="neck", label1="probe", only_running_bouts=False
    ):

        hd_x, hd_y = self._get_vector_from_two_labels(label0=label0, label1=label1)
        index = self.get_index(label=label1)

        resp_in_rad = np.arctan2(
            hd_y * (-1), hd_x
        )  # multiplication by -1 needed because of video x and y directions

        resp_in_rad = np.unwrap(
            resp_in_rad
        )  # unwrapping so smoothing and derivative can be done

        smooth_window = signal.gaussian(M=101, std=10)
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
        """Gets an array where the direction array (head direction here) is modulated by a guassian function centered in `deg`.

        Parameters
        ----------
        deg : int or float
            Head direction in degrees, between 0 and 360.
        only_running_bouts : bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        hd_deg, index = self.get_direction_array(
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

    def get_direction_histogram(self, degrees=4):

        bins = 360 // degrees

        hd_deg, index = self.get_direction_array()

        return np.histogram(hd_deg[index], bins=bins, range=(0,360))

    def get_xy_coords(self, bodypart="body"):
        """Gets array of x, y coordinates of the `bodypart` label in the shape [nframes, 2].

        Parameters
        ----------
        bodypart : str, optional
            By default 'body'

        Returns
        -------
        tuple
            coords : numpy.ndarray
                Values in x and y multiplied by ``ratio_per_pixel`` for bodypart.
            time_array : numpy.ndarray
                Time array in seconds.
            # index : numpy.ndarray
                Index where p-value > pcutout is True, index is False otherwise.
        """

        x, _, index = self.get_position_x(bodypart=bodypart, pcutout=self.pcutout)
        y, _, _ = self.get_position_y(bodypart=bodypart, pcutout=self.pcutout)

        x *= self.ratio_per_pixel
        y *= self.ratio_per_pixel

        coords = np.array([x, y]).T

        return coords, self.time, index

    def get_position_x(self, bodypart, pcutout=None, absolute=False):
        """Simple function to get x values for bodypart.

        Parameters
        ----------
        bodypart : str
        pcutout : float, optional
            Between 0 and 1. If `None`, uses the self.pcutout property. The default is `None`.

        Returns
        -------
        tuple
            x_bp : numpy.ndarray
                Pixel values in x for bodypart.
            time_array : numpy.ndarray
                Time array in seconds.
            # index : numpy.ndarray
                Index where p-value > pcutout is True, index is False otherwise.

        """
        index = self.get_index(bodypart, pcutout)
        x_bp = self.Dataframe[self.scorer][bodypart]["x"].values
        if absolute is False:
            x_bp = (
                self.Dataframe[self.scorer][bodypart]["x"].values
                - self.metadata["data"]["corner_coords"]["top_left"][0]
            )

        return x_bp, self.time, index

    def get_position_y(self, bodypart, pcutout=None, absolute=False):
        """Simple function to get y values for bodypart.

        Parameters
        ----------
        bodypart : str
        pcutout : float, optional
            Between 0 and 1. If `None`, uses the self.pcutout property. The default is `None`.

        Returns
        -------
        tuple
            y_bp : numpy.ndarray
                Pixel values in y for bodypart.
            time_array : numpy.ndarray
                Time array in seconds.
            index : numpy.ndarray
                Index where p-value > pcutout is True, index is False otherwise.

        """

        index = self.get_index(bodypart, pcutout)
        y_bp = self.Dataframe[self.scorer][bodypart]["y"].values
        if absolute is False:
            y_bp = (
                self.Dataframe[self.scorer][bodypart]["y"].values
                - self.metadata["data"]["corner_coords"]["top_left"][1]
            )

        return y_bp, self.time, index


    def get_speed(
        self, bodypart="body", smooth=True, speed_cutout=0, only_running_bouts=False
    ):
        """Gets speed for given `bodypart`. When getting the distance between frames, the first index is hard set to be 0 so that the output array has the same length as the number of frames.

        Parameters
        ----------
        bodypart : str, optional
            Name of the label to get the speed from, by default 'body'.
        smooth : bool, optional
            If True a Gaussian window will convolve the speed array, by default True.
            The parameters of the Gaussian window can be set via the self.speed_smooth_window variable.
        speed_cutout : int, optional
            If given it will set the speed values under this threshold to 0, by default 0.
        only_running_bouts : bool, optional
            [description], by default False.

        Returns
        -------
        tuple
            speed_array : numpy.ndarray
            time_array : numpy.ndarray
            index : numpy.ndarray
                Index where p-value > pcutout is True, index is False otherwise.
            speed_units : str
                String telling the units of the speed_array
        """
        dist_in_px = self._get_distance_between_frames(bodypart=bodypart)
        speed_in_px_per_second = dist_in_px * self.fps

        index = self.get_index(bodypart, self.pcutout)

        time_array = np.array(self.Dataframe.index) / self.fps

        speed_units = self.spatial_units + "/s"
        speed_array = speed_in_px_per_second * self.ratio_per_pixel

        if smooth:
            speed_array[~index] = 0
            speed_array = np.convolve(self.speed_smooth_window, speed_array, "same")
            speed_array[speed_array < speed_cutout] = 0

        if only_running_bouts == True:
            self.get_running_bouts(speed_array=speed_array, time_array=time_array)
            speed_array = self._split_in_running_bouts(speed_array)
            index = self._split_in_running_bouts(index)
            return speed_array, self.time_bouts, index, speed_units

        return speed_array, self.time, index, speed_units

    def get_running_bouts(self, speed_array=None, time_array=None):
        """Automatic gets running bouts given certain parameters, such as speed_threshold,
        minimal duration of running bout and minimal duration of resting bout.

        Parameters
        ----------
        speed_array : array_like, optional
            Array to get automatic running bouts from, if None then Tracking.get_speed() is called, by default None.
        time_array : array_like, optional
            Time array to be passed as output, by default None.

        Returns
        -------
        tuple
            running_bouts : np.array
                Boolean array with True when the animal is running and False otherwise.
            time_array : np.array
                Array mapping the index of the tracking data to time in seconds.
            final_change_idx : np.array
                Tells at which index the running_bout ends (either True or False).
        """
        if speed_array is None or time_array is None:
            speed_array, time_array, _, _ = self.get_speed(bodypart="body", smooth=True)

        # getting True False array where speed is above 10 cm/s
        self.running_bouts = speed_array > 10
        # getting indices where this True False array changes from True to False or False to True
        change_idx = np.where(np.diff(self.running_bouts))[0]
        # getting length from one change_idx to the next one
        bout_lengths = np.insert(np.diff(change_idx), 0, change_idx[0])

        # This for loop gets every False bout (no running) shorter than 15 seconds,
        # either in the beginning or between running bouts and sets them to True (running)
        for i in range(len(bout_lengths)):
            if bout_lengths[i] < 750 and self.running_bouts[change_idx[i]] == False:
                if i == 0:
                    self.running_bouts[: change_idx[i] + 1] = True
                else:
                    self.running_bouts[change_idx[i - 1] : change_idx[i] + 1] = True

        # again, getting indices of change in the new running_bouts array
        temp_change_idx = np.where(np.diff(self.running_bouts))[0]
        # again, getting lengths from one temp_change_idx to the next one
        temp_bout_lengths = np.insert(np.diff(temp_change_idx), 0, temp_change_idx[0])

        # This for loop gets every True bout (running) shorter then 15 seconds,
        # either in the beginning or between no running bouts and sets them to False (no running)
        for i in range(len(temp_bout_lengths)):
            if (
                temp_bout_lengths[i] < 750
                and self.running_bouts[temp_change_idx[i]] == True
            ):
                if i == 0:
                    self.running_bouts[: temp_change_idx[i] + 1] = False
                else:
                    self.running_bouts[
                        temp_change_idx[i - 1] : temp_change_idx[i] + 1
                    ] = False

        self.final_change_idx = np.where(np.diff(self.running_bouts))[0]
        # final_change_idx tells at which index the running_bout ends (either True or False)
        # using running_bouts[final_change_idx[i]] you have True if the True running bout is
        # ending or False if the False running bout is ending

        self.time_bouts = [
            x
            for x in np.split(
                np.where(self.running_bouts, self.time, 0),
                self.final_change_idx + 1,
            )
            if x[1] != 0
        ]

        return self.running_bouts, self.time, self.final_change_idx

    def get_binned_position(
        self, bodypart="body", bins=[10, 10], only_running_bouts=False
    ):

        x_pos, _, index = self.get_position_x(bodypart=bodypart)
        y_pos = self.get_position_y(bodypart=bodypart)[0]
        x_pos *= self.ratio_per_pixel
        y_pos *= self.ratio_per_pixel

        if only_running_bouts:
            if not hasattr(self, "running_bouts"):
                self.get_running_bouts()

            index = self.running_bouts

        return np.histogram2d(
            x_pos[index], y_pos[index], bins=bins, range=[[0, 100], [0, 100]]
        )

    def get_infos(self, bins=10, bin_only_running_bouts=False):
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

    def print_infos(self, bins=10):
        info_dict = self.get_infos(bins=bins)
        print(
            "--------------------------------------------------------------\n"
            + f"Total tracking time: {info_dict['total_time']} s\n"
            + f"Total running time: {info_dict['total_running_time']:.2f} s\n"
            + f"Total distance run: {info_dict['total_distance']:.2f} cm\n"
            + f"Running time ratio (running time / all time): {info_dict['running_ratio']:.2f}\n"
            + f"Exploration ratio (ratio of visited bins): {info_dict['exploration_ratio']:.3f}\n"
            + f"Exploration std (std of visits on each bin): {info_dict['exploration_std']:.3f}\n"
            + f"Mean running speed (only running periods): {info_dict['mean_running_speed']:.2f} {self.spatial_units}/s\n"
            + f"Mean running speed: {info_dict['mean_speed']:.2f} {self.spatial_units}/s\n"
            + "--------------------------------------------------------------"
        )

    def get_proximity_from_wall(
        self, wall="left", bodypart="probe", only_running_bouts=False
    ):
        """Get a sigmoid response from the label position in relation to the specified wall.

        Parameters
        ----------
        wall : str or list of str or tuple of list, optional
            Wall to use for computations. Can be one of ("left", "right", "top", "bottom"), by default "left"
        bodypart : str, optional
            Bodypart to use for computations, by default "probe".
        only_running_bouts : bool, optional
            Use only running bouts of the experiment, by default False

        Returns
        -------
        tuple
            Tuple of ``wall_activation``, time and likelihood indices

        Raises
        ------
        ValueError
            If wall parameter is not on of the possibilities to choose from.
        """
        if wall == "all":
            wall = ("left", "right", "top", "bottom")
            
        if not isinstance(wall, (list, tuple)):
            wall = [wall]

        if not set(wall).issubset(("left", "right", "top", "bottom")):
            raise ValueError(
                f"wall parameter must be one of the following: left, right, top or bottom, not {wall}."
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

    def get_proximity_from_corner(
        self, corner="top right", bodypart="probe", only_running_bouts=False
    ):
        """Get a 2D sigmoid response from the x and y label position in relation to the specified corner.

        Parameters
        ----------
        corner : str, optional
            Must be one of the four corners of a rectangle ("top right", "top left", "bottom right", "bottom left"),
            by default "top right".
        bodypart : str, optional
            Bodypart to use for computations, by default "probe".
        only_running_bouts : bool, optional
            Use only running bouts of the experiment, by default False.

        Returns
        -------
        tuple
            Tuple of ``corner_activation``, time and likelihood indices

        Raises
        ------
        ValueError
            If corner parameter is not on of the possibilities to choose from.
        """
        if corner == "all":
            corner = ("top right", "top left", "bottom right", "bottom left")

        if not isinstance(corner, (list, tuple)):
            corner = [corner]

        if not set(corner).issubset(("top right", "top left", "bottom right", "bottom left")):
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
        coords: tuple = None,
        random_coords=False,
        only_running_bouts=False,
        bodypart="body",
    ):
        """Gets output array by applying the each frame's x, y coordinates to a specific place field.

        Parameters
        ----------
        coords : tuple, optional
            (x, y) coordinates to create the place field. If `None`, gets coordinates from utils.place_fields.get_place_field_coordinates. By default None
        random_coords : bool, optional
            If `coords` is `None`, gets coordinates using the `random` parameter or not. By default False
        only_running_bouts : bool, optional
            Not yet implemented. By default False
        bodypart : str, optional
            Which bodypart label to use when getting coordinates. By default "body"

        Returns
        -------
        tuple (place field array, (time array, place field coordinates))
            Each line in the place field array corresponds to one place field coordinate.
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

    def get_grid_field_array(self, params=None, bodypart="body"):

        animal_coords, time_array, index = self.get_xy_coords(bodypart=bodypart)

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

        return [
            x
            for x in np.split(
                np.where(self.running_bouts, array, 0), self.final_change_idx + 1
            )
            if x[0] != 0
        ]

    def _get_distance_between_frames(self, bodypart="body", backup_bps=["probe"]):
        """Get distance from one frame to another for the specific bodypart along the whole analysis.

        Parameters
        ----------
        bodypart : str, optional
            The default is 'body'.

        Returns
        -------
        distance between frames : numpy.ndarray
            First values is set to 0 so that the returned array has the same size of self.nframes.

        """
        x_pts = self.get_position_x(bodypart=bodypart)[0]
        y_pts = self.get_position_y(bodypart=bodypart)[0]

        dist_in_px = np.sqrt(np.diff(x_pts) ** 2 + np.diff(y_pts) ** 2)

        return np.insert(dist_in_px, 0, 0)

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

        vec_x = (
            self.Dataframe[self.scorer][label1]["x"]
            - self.Dataframe[self.scorer][label0]["x"]
        )
        vec_y = (
            self.Dataframe[self.scorer][label1]["y"]
            - self.Dataframe[self.scorer][label0]["y"]
        )
        return vec_x.to_numpy(), vec_y.to_numpy()

    def _write_corner_coords(self, coords_list):
        """Writes corner coordinates in the metadata of the analysis so it is there for the next time
        and set_ratio_coords does not need to be called again.
        """
        with open(self.metadata_filename, "wb") as f:
            try:
                self.metadata["data"]["corner_coords"] = {}
                self.metadata["data"]["corner_coords"]["top_left"] = np.array(
                    coords_list[0]
                )
                self.metadata["data"]["corner_coords"]["top_right"] = np.array(
                    coords_list[1]
                )
                self.metadata["data"]["corner_coords"]["bottom_left"] = np.array(
                    coords_list[2]
                )
                self.metadata["data"]["corner_coords"]["bottom_right"] = np.array(
                    coords_list[3]
                )
                print("Corner coordinates saved correctly!")
            except AttributeError:
                print("Corner coordinates were not saved!")
                pass
            pickle.dump(self.metadata, f)

    def _get_cm2px_ratio(self):
        """Helper function that calculates the cm/px ratio from the user inputs of the corners of the arena.

        Parameters
        ----------
        w : width in cm
            Distance between right and left corners. The default is 100.
        h : height in cm
            Distance between top and bottom corners. The default is 100.

        Returns
        -------
        float
            Returns the ratio of cm/px of the images being analysed.
        """
        try:
            estimates = np.empty(4)
            estimates[0] = np.sqrt(
                np.sum(
                    (
                        self.metadata["data"]["corner_coords"]["top_right"]
                        - self.metadata["data"]["corner_coords"]["top_left"]
                    )
                    ** 2
                )
            )
            estimates[1] = np.sqrt(
                np.sum(
                    (
                        self.metadata["data"]["corner_coords"]["bottom_right"]
                        - self.metadata["data"]["corner_coords"]["bottom_left"]
                    )
                    ** 2
                )
            )

            estimates[2] = np.sqrt(
                np.sum(
                    (
                        self.metadata["data"]["corner_coords"]["top_left"]
                        - self.metadata["data"]["corner_coords"]["bottom_left"]
                    )
                    ** 2
                )
            )
            estimates[3] = np.sqrt(
                np.sum(
                    (
                        self.metadata["data"]["corner_coords"]["top_right"]
                        - self.metadata["data"]["corner_coords"]["bottom_right"]
                    )
                    ** 2
                )
            )

            w_estimate = self.w / estimates[:2].mean()
            h_estimate = self.h / estimates[2:].mean()
            self.ratio_cm_per_pixel = (w_estimate + h_estimate) / 2
            return self.ratio_cm_per_pixel

        except KeyError:
            print("Ratio cm/px not yet calculated. See function self.set_ratio_coords.")