"""Animation wrappers for the 2D tracking plots."""

from collections.abc import Callable
from functools import wraps
from typing import cast

import cv2
import numpy as np
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure
from matplotlib.path import Path as mPath

from physbeh.plotting.animate_decorator import TrackingAnimation
from physbeh.tracking import Tracking


def anim2d_decorator(plot_function: Callable) -> Callable:
    """Decorator to animate tracking plots synched with video of corresponding tracking.

    Parameters
    ----------
    plot_function : physbeh plot function
        Usually, plot function with one axes that returns Figure and Axes.

    Returns
    -------
    callable
        The wrapped function, either animated or not.
    """

    @wraps(plot_function)
    def plot_wrapper(
        *args, **kwargs
    ) -> tuple[mpl_Figure, mpl_Axes] | tuple[mpl_Figure, mpl_Axes, TrackingAnimation]:
        """Wrapper function to animate tracking plots.

        Parameters
        ----------
        *args : Any
            Arguments to be passed to the plot function.
        **kwargs : dict[str, Any]
            Keyword arguments to be passed to the plot function.

        Returns
        -------
        tuple[mpl_Figure, mpl_Axes] | tuple[mpl_Figure, mpl_Axes, TrackingAnimation]
            If able to animate, a tuple containing the matplotlib figure, the axes and
            the animation class. Otherwise, a tuple containing just the matpotlib figure
            and the axes.
        """
        do_anim = kwargs.pop("animate", False)
        keys = list(kwargs.keys())
        anim_kwargs = {
            k.replace("animate__", ""): kwargs.pop(k)
            for k in keys
            if k.startswith("animate__")
        }
        fig, ax, lines = plot_function(*args, **kwargs)
        anim_kwargs["lines"] = lines
        anim_kwargs.setdefault("use_video", True)
        if do_anim:
            Trk = [arg for arg in args if isinstance(arg, Tracking)]
            if not Trk:
                Trk = cast(
                    list[Tracking],
                    [value for value in kwargs.values() if isinstance(value, Tracking)],
                )
            anim = Animate_plot2D(
                figure=fig,
                axes=ax,
                video_axes=None,
                time_array=Trk[0].time,
                video_path=Trk[0].video_filepath,
                arena=Trk[0].arena,
                show_timestamp=True,
                **anim_kwargs,
            )
            return fig, ax, anim
        return fig, ax, lines

    return plot_wrapper


class Animate_plot2D(TrackingAnimation):
    """Animation of helper class for 2D tracking plots.

    This obviously needs interactive backend to work.
    This class was not built to direct use. Please use `animate_scan(scan)` to
    animate 2D+t or 3D+t scans.
    If `interactive=True`, then this are the controls for the animation:
        `backspace` -> play/pause
        `up/down` -> adjusts the frame step of the animation (default and minimum
        value is 1 to grab the next frame, if set to 2, will skip 1 frame, and so
        on...
        `+/-` -> adjusts the interval between frames in ms (default is 200), it
        will multiply or divide by 2 if used + or -, respectively

    Parameters
    ----------
    lines : dict
        Dictionary containing the lines to be animated.
    *args : Any
        Arguments to be passed to TrackingAnimation.
    **kwargs : dict[str, Any]
        Keyword arguments to be passed to TrackingAnimation.
    """

    def __init__(self, lines, *args, **kwargs):
        self.lines = lines["lines"]
        self.index = lines["index"]
        if not kwargs.pop("use_video", False):
            kwargs["video_path"] = None
        super().__init__(*args, **kwargs)

        self.collection = self.ax.collections[0]
        self.collection._paths.clear()
        self.collection._paths.append(mPath(self.lines[self.current_frame]))
        self._drawn_artists.append(self.collection)

    def _setup_video_axes(self, arena, video_path, x_crop, y_crop):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise cv2.error("Error opening video stream or file")

        camera_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        camera_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if y_crop[-1] < 0:
            y_crop = (y_crop[0], int(camera_height))

        if x_crop[1] < 0:
            x_crop = (x_crop[0], int(camera_width))

        self.y_slice = slice(*y_crop)
        self.x_slice = slice(*x_crop)

        self.extent = arena.get_extent(camera_width, camera_height)

        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        self.max_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        _, frame = self.cap.read()

        self.vid = self.ax.imshow(
            frame[self.y_slice, self.x_slice, ::-1],
            extent=self.extent,
            zorder=-999,
        )
        self._drawn_artists.append(self.vid)

    def _draw_custom_frame(self):
        ind = np.logical_and(
            self.index >= self.current_frame,
            self.index < self.current_frame + self.frame_step,
        )[:-1]
        self.collection._paths += [mPath(line) for line in self.lines[ind]]
        self.grab_frame()

    def grab_frame(self):
        """Grab frame from video and update the axes."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        _, frame = self.cap.read()

        self.vid.set_array(frame[self.y_slice, self.x_slice, ::-1])
