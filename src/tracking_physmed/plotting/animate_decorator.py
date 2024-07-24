import warnings
from functools import wraps

import cv2

from ..tracking import Tracking
from ..utils import BlitManager
from .animate_plot_fUS import Animate_plot_fUS


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
                fig,
                ax,
                video_path=Trk[0].video_filepath,
                x_crop=xcrop,
                y_crop=ycrop,
                arena=Trk[0].arena,
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


# class _Animate_plot(Animation):
#     def __init__(
#         self,
#         figure,
#         video_path=None,
#         x_crop=[0, -1],
#         y_crop=[0, -1],
#         interactive=True,
#         show_timestamp=True,
#         position_timestamp=[0, 0, 0.5, 0.5],
#         other_artists=[],
#     ):
#         """Animation of 2/3D+t scans. This obviously needs interactive backend to
#         work.
#         This class was not built to direct use. Please use `animate_scan(scan)` to
#         animate 2D+t or 3D+t scans.
#         If `interactive=True`, then this are the controls for the animation:
#             `backspace` -> play/pause
#             `up/down` -> adjusts the frame step of the animation (default and minimum
#             value is 1 to grab the next frame, if set to 2, will skip 1 frame, and so
#             on...
#             `+/-` -> adjusts the interval between frames in ms (default is 200), it
#             will multiply or divide by 2 if used + or -, respectively
#         """
#         ## Creating custom animation inheriting matplotlib Animation class
#         self.is_playing = True

#         # frame_step says if animation is going frame by frame (frame_step = 1), or
#         # if it is going to skip one frame (frame_step = 2), etc.
#         # if sampling frequency of the animation data is high, sometimes skipping
#         # some frames is a good trade-off to have a smoother animation
#         self.frame_step = 1
#         self.current_frame = 0

#         self.anim_interval = 50
#         event_source = figure.canvas.new_timer(interval=self.anim_interval)


#         # self._framedata = range(0, self.n_frames, self.frame_step)
#         self._drawn_artists = []

#         super().__init__(fig=figure, event_source=event_source, blit=True)
#         # self._drawn_artists += images

#         self.show_timestamp = show_timestamp
#         if show_timestamp:
#             # Adding an axes for counting the frames
#             ax_xlabel = figure.add_axes(position_timestamp, frameon=False)
#             ax_xlabel.xaxis.set_visible(False)
#             ax_xlabel.yaxis.set_visible(False)
#             self.time_stamp = ax_xlabel.text(
#                 0.5,
#                 0.5,
#                 "current frame: {fr:04} | time: {sec:07.2f} s".format(
#                     fr=self.current_frame, sec=self.data_time[self.current_frame]
#                 ),
#                 transform=ax_xlabel.transAxes,
#                 ha="center",
#                 va="top",
#                 bbox=dict(facecolor="white", alpha=1, edgecolor="white"),
#             )
#             self._drawn_artists.append(self.time_stamp)
#         if other_artists:
#             self._drawn_artists.append(*other_artists)

#         # interval limit is the minimum time delay between frames (in ms)
#         # set anim_interval to 1 does not mean it is really going to update frames
#         # every 1 ms, but rather it will update as fast as possible
#         self.interval_limit = 1
#         if not self._blit:
#             self.interval_limit = 100
#             self.anim_interval = 200
#             warnings.warn(
#                 "Matplotlib figure does not support blit. Animation will not be
#                 optimal. This happens because the backend used does not support
#                 matplotlib blitting."
#             )

#         if interactive is False:
#             # just starts animation and let it playing
#             self.play()
#         else:
#             self.ckeypress = figure.canvas.mpl_connect(
#                 "key_press_event", self.onkeypress
#             )

#     def _start(self, *args):
#         """
#         Starts interactive animation. Adds the draw frame command to the GUI
#         handler, calls show to start the event loop.
#         """
#         # Do not start the event source if saving() it.
#         if self._fig.canvas.is_saving():
#             return
#         # First disconnect our draw event handler
#         self._fig.canvas.mpl_disconnect(self._first_draw_id)

#         # Now do any initial draw
#         self._init_draw()

#         # Add our callback for stepping the animation and
#         # actually start the event_source.
#         self.event_source.add_callback(self._step)
#         self.event_source.start()

#     def _step(self):
#         try:
#             self.current_frame = next(self.frame_seq)
#             self._draw_next_frame(self.current_frame, self._blit)
#         except StopIteration:
#             # modified from default to restart animation when iterator ends
#             self.frame_seq = self.new_frame_seq()

#     def _draw_frame(self, framedata):
#         # handles the next frame, if outside of the time dimension, goes back to the
#         # beginning
#         for data, im in zip(self.data, self.images):
#             im.set_array(data[..., framedata])
#         if self.show_timestamp:
#             self.time_stamp.set_text(
#                 "current frame: {fr:04} | time: {sec:07.2f} s".format(
#                     fr=framedata, sec=self.data_time[framedata]
#                 )
#             )

#     def _on_resize(self, event):
#         # On resize, we need to disable the resize event handling so we don't
#         # get too many events. Also stop the animation events, so that
#         # we're paused. Reset the cache and re-init. Set up an event handler
#         # to catch once the draw has actually taken place.
#         self._fig.canvas.mpl_disconnect(self._resize_id)
#         # slightly modified from matplotlib default where if animation is paused,
#         # resizing it would resume playing. Now it only resumes playing if it
#         # was already playing
#         if self.is_playing:
#             self.pause()
#         self._blit_cache.clear()
#         self._init_draw()
#         self._resize_id = self._fig.canvas.mpl_connect("draw_event", self._end_redraw)

#     def _end_redraw(self, event):
#         # Now that the redraw has happened, do the post draw flushing and
#         # blit handling. Then re-enable all of the original events.
#         self._post_draw(None, False)
#         # slightly modified from matplotlib default where if animation is paused,
#         # resizing it would resume playing. Now it only resumes playing if it
#         # was already playing
#         if self.is_playing:
#             self.resume()
#         self._fig.canvas.mpl_disconnect(self._resize_id)
#         self._resize_id = self._fig.canvas.mpl_connect("resize_event",
#         self._on_resize)

#     def onkeypress(self, event):
#         # animation controls
#         if event.key == " ":
#             self.play()

#         elif event.key == "-":
#             self.anim_interval *= 2
#             self.event_source.interval = int(self.anim_interval)
#         elif event.key == "+":
#             if self.anim_interval <= self.interval_limit:
#                 pass
#             else:
#                 self.anim_interval /= 2
#         elif event.key == "up":
#             self.frame_step += 1
#             self._framedata = range(self.current_frame, self.n_frames,
#             self.frame_step)
#             self.frame_seq = self.new_frame_seq()
#         elif event.key == "down":
#             if self.frame_step > 1:
#                 self.frame_step -= 1
#                 self._framedata = range(
#                     self.current_frame, self.n_frames, self.frame_step
#                 )
#                 self.frame_seq = self.new_frame_seq()

#         print(
#             "Frame step:",
#             self.frame_step,
#             "Timer interval:",
#             int(self.anim_interval),
#             "msec",
#         )
#         self.event_source.interval = int(self.anim_interval)

#     def play(self):
#         # simply toggle between play/pause
#         if self.is_playing:
#             self.pause()
#             self.is_playing = False

#         else:
#             self.is_playing = True
#             self.resume()


class Animate_plot:
    def __init__(
        self, fig, ax, video_path=None, x_crop=[0, -1], y_crop=[0, -1], arena=None
    ):
        self.fig = fig
        self.ax = ax

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        self.vid = None
        self.useblit = self.fig.canvas.supports_blit
        self.interval_limit = 1
        if not self.useblit:
            self.interval_limit = 100
            warn_msg = (
                "Matplotlib figure does not support blit. Animation will not be"
                "optimal. This happens because the backend used does not support"
                " matplotlib blitting."
            )
            warnings.warn(warn_msg)

        self.bm = BlitManager(self.fig.canvas)

        if video_path:
            self.ax.set_xlabel("")
            self.ax.set_position([0.4, 0.13, 0.55, 0.75])
            self.ax_vid = self.fig.add_axes([0.05, 0.13, 0.3, 0.75])
            # self.fig.set_size_inches(14, 5)

            self.current_frame = 0
            self.current_time = 0

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

            self.grab_first_frame()

            self.ax_vid.set(xlabel="X pixel", ylabel="Y pixel")

            self.ax_xlabel = self.fig.add_axes([0.55, 0.002, 0.3, 0.07], frameon=False)
            self.ax_xlabel.xaxis.set_visible(False)
            self.ax_xlabel.yaxis.set_visible(False)

            self.time_stamp = self.ax_xlabel.annotate(
                self._get_timestamp(),
                (0.5, 0.75),
                xycoords="axes fraction",
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="white"),
                animated=self.useblit,
            )
            self.bm.add_artist(self.time_stamp)

        self.play_bar = self.ax.axvline(
            self.current_time, color="gray", animated=self.useblit
        )
        self.bm.add_artist(self.play_bar)

        self.press = False
        self.move = False

        self.cpress = self.fig.canvas.mpl_connect("button_press_event", self.onpress)
        self.crelease = self.fig.canvas.mpl_connect(
            "button_release_event", self.onrelease
        )
        self.cmove = self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)
        self.ckeypress = self.fig.canvas.mpl_connect("key_press_event", self.onkeypress)
        self.ckeyrelease = self.fig.canvas.mpl_connect(
            "key_release_event", self.onkeyrelease
        )

        self.is_playing = False
        self.is_starting = True

        ## Creating custom animation via matplotlib timer
        self.anim_interval = self.interval_limit
        self.frame_step = 1
        self.custom_ani = self.fig.canvas.new_timer(interval=self.anim_interval)
        self.custom_ani.add_callback(self.update_frame)
        self.custom_ani.add_callback(self.bm.update)

    def onkeypress(self, event):
        if event.key == " ":
            self.play()

        elif event.key == "-":
            self.anim_interval *= 2
            self.custom_ani.interval = int(self.anim_interval)
        elif event.key == "+":
            if self.anim_interval <= self.interval_limit:
                pass
            else:
                self.anim_interval /= 2
        elif event.key == "up":
            self.frame_step += 1
        elif event.key == "down":
            if self.frame_step == 1:
                pass
            else:
                self.frame_step -= 1

        print(
            "Frame step:",
            self.frame_step,
            "Timer interval:",
            int(self.anim_interval),
            "msec",
        )
        self.custom_ani.interval = int(self.anim_interval)

    def onkeyrelease(self, event):
        pass

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
        if event.inaxes == self.ax:
            # self.current_time = event.xdata
            self.current_frame = int(event.xdata * self.fps)
            if self.current_frame >= self.max_frames:
                self.current_frame = int(self.max_frames - 1)
            if not self.is_playing:
                self.grab_frame(self.current_frame)
                self.play_bar.set_xdata([self.current_time, self.current_time])
                self.time_stamp.set_text(self._get_timestamp())

                self.bm.update()

    def grab_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        self.vid = self.ax_vid.imshow(
            frame[self.y_slice, self.x_slice, ::-1],
            animated=self.useblit,
            extent=self.extent,
        )
        self.bm.add_artist(self.vid)
        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.max_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def grab_frame(self, fr):
        if fr < 0:
            fr = 0
        self.current_frame = fr
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        _, frame = self.cap.read()

        self.vid.set_array(frame[self.y_slice, self.x_slice, ::-1])

        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3

    def update_frame(self):
        if (
            self.current_time < self.ax.get_xlim()[1]
            and self.current_time > self.ax.get_xlim()[0]
        ):
            self.current_frame += self.frame_step
            if self.current_frame >= self.max_frames:
                self.current_frame = 0

        elif self.current_frame >= self.max_frames:
            self.current_frame = 0
        else:
            new_time = self.ax.get_xlim()[0] if self.ax.get_xlim()[0] >= 0 else 0
            self.current_frame = new_time // (1 / self.fps) + 1

        self.grab_frame(self.current_frame)

        self.play_bar.set_xdata([self.current_time, self.current_time])
        self.time_stamp.set_text(self._get_timestamp())

    def play(self):
        if self.is_playing:
            self.custom_ani.stop()
            self.is_playing = False

        elif not self.is_playing:
            self.is_playing = True

            if self.is_starting:
                self.custom_ani.start()
                self.is_starting = False

            elif not self.is_starting:
                self.custom_ani.start()

    def _get_timestamp(self):
        return (
            f"current fr: {self.current_frame:05} | "
            f"time: {self.current_time:06.2f} s"
        )
