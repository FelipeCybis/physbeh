import warnings
from functools import wraps

import cv2
from matplotlib.animation import Animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tracking_physmed.plotting.animate_plot_fUS import Animate_plot_fUS
from tracking_physmed.tracking import Tracking


def anim_decorator(plot_function):
    """Decorator to animate tracking plots synched with video of corresponding tracking.

    Parameters
    ----------
    plot_function : tracking_physmed plot function
        Usually, plot function with one axes that returns Figure and Axes.
    """

    @wraps(plot_function)
    def plot_wrapper(*args, **kwargs):
        anim_video = kwargs.pop("animate_video", False)
        anim_fus = kwargs.pop("animate_fus", False)
        do_anim = kwargs.pop("animate", False)
        fig, ax = plot_function(*args, **kwargs)
        if anim_video or do_anim:
            Trk = [arg for arg in args if isinstance(arg, Tracking)]
            if not Trk:
                Trk = [
                    value for value in kwargs.values() if isinstance(value, Tracking)
                ]

            if Trk[0].video_filepath is None:
                return fig, ax

            if hasattr(Trk[0], "metadata"):
                cropping = (
                    Trk[0].metadata["data"].get("cropping_parameters", [0, -1, 0, -1])
                )
            else:
                cropping = [0, -1, 0, -1]
            xcrop = cropping[:2]
            ycrop = cropping[2:]
            anim = Animate_plot(
                figure=fig,
                axes=ax,
                time_array=Trk[0].time,
                video_path=Trk[0].video_filepath,
                arena=Trk[0].arena,
                show_timestamp=True,
            )
            return fig, ax, anim
        elif anim_fus:
            Trk = [arg for arg in args if isinstance(arg, Tracking)]
            if not Trk:
                Trk = [
                    value for value in kwargs.values() if isinstance(value, Tracking)
                ]

            if Trk[0].scan is None:
                return fig, ax

            if hasattr(Trk[0], "metadata"):
                cropping = (
                    Trk[0].metadata["data"].get("cropping_parameters", [0, -1, 0, -1])
                )
            else:
                cropping = [0, -1, 0, -1]
            xcrop = cropping[:2]
            ycrop = cropping[2:]

            anim = Animate_plot_fUS(
                fig=fig,
                ax=ax,
                scan=Trk[0].scan,
                video_path=Trk[0].video_filepath,
                x_crop=xcrop,
                y_crop=ycrop,
            )
            return fig, ax, anim

        return fig, ax

    return plot_wrapper


class TrackingAnimation(Animation):
    @property
    def time(self):
        """The time array of the data being plotted.

        Returns
        -------
        numpy.ndarray
            The time array in seconds.
        """
        return self._time

    @property
    def current_time(self):
        """The current time given by the current frame.

        Returns
        -------
        float
            The current time in seconds.
        """
        return self.time[self.current_frame]

    def __init__(
        self,
        figure,
        axes,
        time_array,
        video_path=None,
        arena=None,
        x_crop=[0, -1],
        y_crop=[0, -1],
        interactive=True,
        show_timestamp=True,
        other_artists=[],
    ):
        """Animation of 2/3D+t scans. This obviously needs interactive backend to
        work.
        This class was not built to direct use. Please use `animate_scan(scan)` to
        animate 2D+t or 3D+t scans.
        If `interactive=True`, then this are the controls for the animation:
            `backspace` -> play/pause
            `up/down` -> adjusts the frame step of the animation (default and minimum
            value is 1 to grab the next frame, if set to 2, will skip 1 frame, and so
            on...
            `+/-` -> adjusts the interval between frames in ms (default is 200), it
            will multiply or divide by 2 if used + or -, respectively
        """
        ## Creating custom animation inheriting matplotlib Animation class
        self.is_playing = True

        # frame_step says if animation is going frame by frame (frame_step = 1), or
        # if it is going to skip one frame (frame_step = 2), etc.
        # if sampling frequency of the animation data is high, sometimes skipping
        # some frames is a good trade-off to have a smoother animation
        self.frame_step = 1
        self.current_frame = 0

        self._interval = 50
        event_source = figure.canvas.new_timer(interval=self._interval)
        self._repeat = True

        self._time = time_array
        self.fps = 1 / (time_array[1] - time_array[0])
        self.n_frames = len(time_array)
        self._framedata = range(0, self.n_frames, self.frame_step)
        self._drawn_artists = []

        super().__init__(fig=figure, event_source=event_source, blit=True)
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
            self.ckeypress = figure.canvas.mpl_connect(
                "key_press_event", self.onkeypress
            )
            self.press = False
            self.move = False

            self.cpress = figure.canvas.mpl_connect("button_press_event", self.onpress)
            self.crelease = figure.canvas.mpl_connect(
                "button_release_event", self.onrelease
            )
            self.cmove = figure.canvas.mpl_connect("motion_notify_event", self.onmove)

    def _setup_video_axes(self, arena, video_path, x_crop, y_crop):
        pass

    def _start(self, *args):
        """Starts interactive animation.

        Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        """
        # Do not start the event source if saving() it.
        if self._fig.canvas.is_saving():
            return
        # First disconnect our draw event handler
        self._fig.canvas.mpl_disconnect(self._first_draw_id)

        # Now do any initial draw
        self._init_draw()

        # Add our callback for stepping the animation and
        # actually start the event_source.
        self.event_source.add_callback(self._step)
        self.event_source.start()

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

    def onpress(self, event):
        self.press = True

    def onmove(self, event):
        if self.press:
            self.move = True

    def onrelease(self, event):
        if self.press and not self.move:
            self.onclick(event)  # click without moving

        self.press = False
        self.move = False

    def onclick(self, event):
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
            self._resize_id = self._fig.canvas.mpl_connect(
                "draw_event", self._end_redraw
            )

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
            self._resize_id = self._fig.canvas.mpl_connect(
                "resize_event", self._on_resize
            )


class Animate_plot(TrackingAnimation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init playbar that will move along the time axis
        self.play_bar = self.ax.axvline(self.current_time, color="gray")
        self._drawn_artists.append(self.play_bar)

    def onclick(self, event):
        if event.inaxes == self.ax:
            # self.current_time = event.xdata
            self.current_frame = int(event.xdata * self.fps)
            if self.current_frame >= self.max_frames:
                self.current_frame = int(self.max_frames - 1)

            self._draw_next_frame(self.current_frame, self._blit)
            self._refresh_frame_seq()

    def _setup_video_axes(self, arena, video_path, x_crop, y_crop):
        self.video_axes = make_axes_locatable(self.ax).append_axes(
            "left", size="65%", pad=1
        )
        self.video_axes.set_aspect("equal")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise cv2.error("Error opening video stream or file")

        camera_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        camera_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if y_crop[-1] < 0:
            y_crop[1] = int(camera_height)

        if x_crop[1] < 0:
            x_crop[1] = int(camera_width)

        self.y_slice = slice(*y_crop)
        self.x_slice = slice(*x_crop)

        self.extent = arena.get_extent(camera_width, camera_height)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        _, frame = self.cap.read()

        self.vid.set_array(frame[self.y_slice, self.x_slice, ::-1])
