"""Animation wrappers for the tracking plots."""

import pathlib
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.animation import Animation
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure as mpl_Figure

from physbeh.arena import BaseArena
from physbeh.plotting.figure import BehFigure
from physbeh.tracking import Tracking


def _check_ax_and_fig(ax, fig, **fig_kwargs) -> tuple[mpl_Axes, mpl_Figure]:
    if ax is None:
        if fig is None:
            fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure
        assert fig == ax.figure, "Axes and figure must be from the same object."
    return ax, fig


def anim_decorator(plot_function: Callable) -> Callable:
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
        *args: Any, **kwargs: dict[str, Any]
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
        anim_video = kwargs.pop("animate_video", False)
        do_anim = kwargs.pop("animate", False)
        keys = list(kwargs.keys())
        anim_kwargs = {
            k.replace("animate__", ""): kwargs.pop(k)
            for k in keys
            if k.startswith("animate__")
        }

        Trk = [arg for arg in args if isinstance(arg, Tracking)]
        if not Trk:
            Trk = [value for value in kwargs.values() if isinstance(value, Tracking)]

        axes = kwargs.pop("axes", None)
        figure = kwargs.pop("figure", None)
        figsize = kwargs.pop("figsize", (12, 6))

        figure, (video_axes, axes) = plt.subplots(
            1, 2, width_ratios=[0.4, 0.6], figsize=figsize
        )
        fig, ax = plot_function(*args, figure=figure, axes=axes, **kwargs)
        if Trk[0].video_filepath is not None and (anim_video or do_anim):
            anim = Animate_plot(
                figure=fig,
                axes=ax,
                video_axes=video_axes,
                time_array=Trk[0].time,
                video_path=Trk[0].video_filepath,
                arena=Trk[0].arena,
                show_timestamp=True,
                **anim_kwargs,
            )
            return fig, ax, anim

        return fig, ax

    return plot_wrapper


class TrackingAnimation(Animation):
    """Base animation helper for animating tracking data.

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
    figure : mpl_Figure
        The matplotlib figure object.
    axes : mpl_Axes
        The matplotlib axes object.
    video_axes : mpl_Axes, optional
        The matplotlib axes object to plot the video. Default is ``None``.
    time_array : numpy.ndarray
        The time array of the data being plotted.
    video_path : str | pathlib.Path, optional
        The path to the video file to be plotted. Default is ``None``.
    arena : BaseArena, optional
        The arena object to be used to crop the video. Default is ``None``.
    x_crop : tuple[int, int], optional
        The x-axis crop limits. Default is ``(0, -1)`` to plot the whole x-axis.
    y_crop : tuple[int, int], optional
        The y-axis crop limits. Default is ``(0, -1)`` to plot the whole y-axis.
    interactive : bool, optional
        If ``True``, the animation will be interactive. Default is ``True``.
    show_timestamp : bool, optional
        If ``True``, the timestamp will be shown on the plot. Default is ``True``.
    other_artists : list, optional
        Other artists to be plotted on the axes. Default is ``[]``.
    blit : bool, optional
        If ``True``, the animation will be blitted (see matplotlib's blitting). Default
        is ``True``.

    Attributes
    ----------
    time : numpy.ndarray
        The time array of the data being plotted.
    current_time : float
        The current time given by the current frame.
    """

    @property
    def time(self) -> npt.NDArray:
        """The time array of the data being plotted.

        Returns
        -------
        numpy.ndarray
            The time array in seconds.
        """
        return self._time

    @property
    def current_time(self) -> float:
        """The current time given by the current frame.

        Returns
        -------
        float
            The current time in seconds.
        """
        return self.time[self.current_frame]

    def __init__(
        self,
        figure: BehFigure,
        axes: mpl_Axes,
        video_axes: mpl_Axes | None,
        time_array: npt.NDArray,
        video_path: str | pathlib.Path | None = None,
        arena: BaseArena | None = None,
        x_crop: tuple[int, int] = (0, -1),
        y_crop: tuple[int, int] = (0, -1),
        interactive: bool = True,
        show_timestamp: bool = True,
        other_artists: list = [],
        blit=True,
    ):
        ## Creating custom animation inheriting matplotlib Animation class
        self.is_playing: bool = True

        # frame_step says if animation is going frame by frame (frame_step = 1), or
        # if it is going to skip one frame (frame_step = 2), etc.
        # if sampling frequency of the animation data is high, sometimes skipping
        # some frames is a good trade-off to have a smoother animation
        self.frame_step: int = 1
        self.current_frame: int = 0

        self.fig = figure
        self._interval = 1
        event_source = self.fig.figure.canvas.new_timer(interval=self._interval)
        self._repeat = True

        self._time = time_array
        self.fps = 1 / (time_array[1] - time_array[0])
        self.n_frames = len(time_array)
        self._framedata = range(0, self.n_frames, self.frame_step)
        self._drawn_artists = []

        super().__init__(fig=self.fig.figure, event_source=event_source, blit=blit)
        self._fig = self.fig.figure
        # self._fig.canvas.mpl_disconnect(self._first_draw_id)

        self.video_axes = video_axes
        self.ax = axes
        # self._drawn_artists += images
        if video_path is not None:
            self._setup_video_axes(
                arena=arena, video_path=video_path, x_crop=x_crop, y_crop=y_crop
            )

        self.show_timestamp = show_timestamp
        if show_timestamp:
            # using the axes xlabel as timestamp
            self.time_stamp = axes.set_xlabel(self._get_timestamp())
            self.time_stamp.set_bbox(
                dict(facecolor="white", alpha=1, edgecolor="white")
            )
            other_artists.append(self.time_stamp)

        if other_artists:
            self._drawn_artists += other_artists

        # interval limit is the minimum time delay between frames (in ms)
        # set anim_interval to 1 does not mean it is really going to update frames
        # every 1 ms, but rather it will update as fast as possible
        self.interval_limit = 1
        if not self._blit:
            self._interval = 50
            self.event_source.interval = int(self._interval)
            self.interval_limit = 100
            warnings.warn(
                "Matplotlib figure does not support blit. Animation will not be "
                "optimal. This happens because the backend used does not support "
                "matplotlib blitting."
            )

        if interactive is False:
            # just starts animation and let it playing
            self.play()
        else:
            self.ckeypress = self.fig.figure.canvas.mpl_connect(
                "key_press_event", self.onkeypress
            )
            self.press = False
            self.move = False

            self.cpress = self.fig.figure.canvas.mpl_connect(
                "button_press_event", self._onpress
            )
            self.crelease = self.fig.figure.canvas.mpl_connect(
                "button_release_event", self._onrelease
            )
            self.cmove = self.fig.figure.canvas.mpl_connect(
                "motion_notify_event", self._onmove
            )

    def _setup_video_axes(self, arena, video_path, x_crop, y_crop):
        """TO BE IMPLEMENTED IN CHILD CLASSES."""
        pass

    def _step(self):
        try:
            self._draw_next_frame(next(self.frame_seq), self._blit)
            return True
        except StopIteration:
            # modified from default to restart animation when iterator ends
            self.frame_seq = self.new_frame_seq()
            return True

    def _draw_frame(self, framedata):
        # handles the next frame, if outside of the time dimension, goes back to the
        # beginning
        self.current_frame = framedata

        if self.show_timestamp:
            self.time_stamp.set_text(self._get_timestamp())

        self._draw_custom_frame()

    def _draw_custom_frame(self):
        raise NotImplementedError

    # Slightly reworking matpltolib _blit_draw method so it is a little more flexible
    # For example, built-in blitting in matplotlib assumes blitting occurs only INSIDE
    # axes, so it is not possible to animate labels or titles.
    # This patch will redraw only the artists (self._drawn_artitsts), but it will blit
    # the entire figure. This is possibly a little bit slower, but really the difference
    # is minimal, it is still quite fast
    def _blit_draw(self, artists):
        # Handles blitted drawing, which renders only the artists given instead
        # of the entire figure and the blits the entire figure.
        # Make a separate pass to draw foreground.
        for a in artists:
            self._fig.draw_artist(a)
        # After rendering all the needed artists, blit the entire figure.
        self._fig.figure.canvas.blit()

    def _get_timestamp(self):
        return (
            f"current fr: {self.current_frame:05} | "
            f"time: {self.current_time:06.2f} s"
        )

    def onkeypress(self, event):
        """Define key press events.

        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            Matplotlib key event that is fired when a key is pressed.
        """
        # animation controls
        if event.key == " ":
            self.play()

        elif event.key == "-":
            self._interval *= 2
            self.event_source.interval = int(self._interval)
        elif event.key == "+":
            if self._interval <= self.interval_limit:
                pass
            else:
                self._interval /= 2
        elif event.key == "up":
            self.frame_step += 1
            self._refresh_frame_seq()
        elif event.key == "down":
            if self.frame_step > 1:
                self.frame_step -= 1
                self._refresh_frame_seq()

        print(
            "Frame step:",
            self.frame_step,
            "Timer interval:",
            int(self._interval),
            "msec",
        )
        self.event_source.interval = int(self._interval)

    def _onpress(self, event):
        self.press = True

    def _onmove(self, event):
        if self.press:
            self.move = True

    def _onrelease(self, event):
        if self.press and not self.move:
            self._onclick(event)  # click without moving

        self.press = False
        self.move = False

    def _onclick(self, event):
        pass

    def _refresh_frame_seq(self):
        self._framedata = range(0, self.n_frames, self.frame_step)
        self.frame_seq = self.new_frame_seq()
        _ = [
            next(self.frame_seq) for _ in range(0, self.current_frame, self.frame_step)
        ]

    def play(self):
        """Animation play/pause function."""
        # simply toggle between play/pause
        if self.is_playing:
            self.pause()
            self.is_playing = False

        else:
            self.is_playing = True
            self.resume()

    def _on_resize(self, event):
        # On resize, we need to disable the resize event handling so we don't
        # get too many events. Also stop the animation events, so that
        # we're paused. Reset the cache and re-init. Set up an event handler
        # to catch once the draw has actually taken place.
        self._fig.canvas.mpl_disconnect(self._resize_id)
        # slightly modified from matplotlib default where if animation is paused,
        # resizing it would resume playing. Now it only resumes playing if it
        # was already playing
        if self.is_playing:
            self.pause()
        self._blit_cache.clear()
        self._init_draw()
        self._resize_id = self._fig.canvas.mpl_connect("draw_event", self._end_redraw)

    def _end_redraw(self, event):
        # Now that the redraw has happened, do the post draw flushing and
        # blit handling. Then re-enable all of the original events.
        self._post_draw(None, False)
        # slightly modified from matplotlib default where if animation is paused,
        # resizing it would resume playing. Now it only resumes playing if it
        # was already playing
        if self.is_playing:
            self.resume()
        self._fig.canvas.mpl_disconnect(self._resize_id)
        self._resize_id = self._fig.canvas.mpl_connect("resize_event", self._on_resize)


class Animate_plot(TrackingAnimation):
    """Animation class for 1D plots (feature vs. time).

    Parameters
    ----------
    *args : Any
        Arguments to be passed to the TrackingAnimation.
    **kwargs : Any
        Keyword arguments to be passed to the TrackingAnimation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init playbar that will move along the time axis
        self.play_bar = self.ax.axvline(self.current_time, color="gray")
        self._drawn_artists.append(self.play_bar)

    def onclick(self, event: MouseEvent):
        """Define click events.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Matplotlib mouse event that is fired when a mouse button is pressed.
        """
        if event.inaxes == self.ax:
            self.current_frame = int(event.xdata * self.fps)
            if self.current_frame >= self.max_frames:
                self.current_frame = int(self.max_frames - 1)

            self._draw_next_frame(self.current_frame, self._blit)
            self._refresh_frame_seq()

    def _setup_video_axes(self, arena: BaseArena | None, video_path, x_crop, y_crop):
        self.video_axes.set_aspect("equal")

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

        self.vid = self.video_axes.imshow(
            frame[self.y_slice, self.x_slice, ::-1],
            extent=self.extent,
        )
        self._drawn_artists.append(self.vid)

    def _draw_custom_frame(self):
        self.play_bar.set_xdata([self.current_time, self.current_time])
        self.grab_frame()

    def grab_frame(self):
        """Grab frame from video and update the axes."""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        _, frame = self.cap.read()

        self.vid.set_array(frame[self.y_slice, self.x_slice, ::-1])
