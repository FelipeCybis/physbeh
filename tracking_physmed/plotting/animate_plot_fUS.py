import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt

from ..utils import BlitManager


class Animate_plot_fUS:
    def __init__(self, fig, ax, scan, video_path, y_crop, x_crop, fus_colorbar=True):
        self.fig = fig
        self.ax = ax

        self.data = np.rot90(scan.get_data()[:, 0, :])

        self.scan_time = scan.time
        self.n_scan_frames = self.data.shape[-1]
        self.dt_scan = scan.voxdim[-1]

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        self.useblit = self.fig.canvas.supports_blit
        self.interval_limit = 1
        self.frame_step = 1
        if not self.useblit:
            self.interval_limit = 100
            warnings.warn(
                "Matplotlib figure does not support blit. Animation will not be optimal. This happens because the backend used does not support matplotlib blitting."
            )

        self.bm = BlitManager(self.fig.canvas)

        self.ax.set_xlabel("")
        self.ax.set_position([0.12, 0.08, 0.8, 0.25])
        self.ax_fus = self.fig.add_axes([0.08, 0.38, 0.42, 0.58])
        self.ax_vid = self.fig.add_axes([0.54, 0.38, 0.42, 0.58])
        self.ax_fus.axes.xaxis.set_visible(False)
        self.ax_fus.axes.yaxis.set_visible(False)
        self.ax_vid.axes.xaxis.set_visible(False)
        self.ax_vid.axes.yaxis.set_visible(False)

        self.current_video_frame = 0
        self.current_fus_frame = 0
        self.current_time = 0

        self.cap = cv2.VideoCapture(str(video_path))
        if self.cap.isOpened() == False:
            raise cv2.error("Error opening video stream or file")

        self.y_crop = y_crop
        self.x_crop = x_crop
        self.grab_first_frame()

        self.data[np.isinf(self.data)] = np.nan
        vmax = np.nanmax(self.data)
        vmin = np.nanmin(self.data)
        if np.abs(vmax) > np.abs(vmin):
            vmin = -vmax
        else:
            vmax = -vmin

        self.fus_im = self.ax_fus.imshow(
            self.data[..., 0], cmap="gray", vmin=vmin, vmax=vmax
        )
        if fus_colorbar:
            plt.colorbar(self.fus_im)
        self.bm.add_artist(self.fus_im)

        self.ax_xlabel = self.fig.add_axes([0.35, 0.001, 0.3, 0.07], frameon=False)
        self.ax_xlabel.xaxis.set_visible(False)
        self.ax_xlabel.yaxis.set_visible(False)

        self.time_stamp = self.ax_xlabel.annotate(
            "current video frame: {vfr:05} | current fUS frame: {fr:04} | time: {sec:06.2f} s".format(
                vfr=self.current_video_frame,
                fr=self.current_fus_frame,
                sec=self.current_time,
            ),
            (0.5, 0.5),
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
        self.custom_ani.add_callback(self.next_frame)
        self.custom_ani.add_callback(self.bm.update)

    def onkeypress(self, event):
        if event.key == " ":
            self.play()

        elif event.key == "-":
            self.anim_interval *= 2
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
            self.current_video_frame = int(event.xdata // (1 / self.video_fps))
            if self.current_video_frame >= self.max_video_frames:
                self.current_video_frame = int(self.max_video_frames - 1)

            if not self.is_playing:
                self.grab_frame()
                self.bm.update()

    def grab_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_video_frame)
        ret, frame = self.cap.read()

        self.vid = self.ax_vid.imshow(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ],
            animated=self.useblit,
        )
        self.bm.add_artist(self.vid)
        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.max_video_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def next_frame(self):
        # handles the next frame, if outside of the time dimension, goes back to the beginning
        self.current_video_frame += self.frame_step
        if self.current_video_frame >= self.max_video_frames:
            self.current_video_frame -= self.max_video_frames

        self.grab_frame()

    def grab_frame(self):
        # simply grabbing requested frame
        # done like this to maybe add a goto frame capability
        pred_time = self.current_video_frame / self.video_fps
        if pred_time > self.ax.get_xlim()[1] or pred_time < self.ax.get_xlim()[0]:
            if (
                self.ax.get_xlim()[0] >= 0
                and self.ax.get_xlim()[0] < self.scan_time[-1]
            ):
                new_time = self.ax.get_xlim()[0]
            else:
                new_time = 0
            self.current_video_frame = int(new_time * self.video_fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_video_frame)
        _, frame = self.cap.read()
        self.vid.set_array(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ]
        )
        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3

        self.current_fus_frame = self._get_fus_frame_from_time(self.current_time)
        self.fus_im.set_array(self.data[..., self.current_fus_frame])

        self.play_bar.set_xdata([self.current_time, self.current_time])
        self.time_stamp.set_text(
            "current video frame: {vfr:05} | current fUS frame: {fr:04} | time: {sec:06.2f} s".format(
                vfr=self.current_video_frame,
                fr=self.current_fus_frame,
                sec=self.current_time,
            )
        )

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

    def _get_fus_frame_from_time(self, time):
        idx = list(np.where(time < self.scan_time)[0])
        if not idx:
            idx = [self.scan_time.shape[-1] - 1]

        return idx[0]


class Animate_video_fUS:
    def __init__(self, tracking, fig=None):
        self.fig = fig
        if self.fig is None:
            self.fig = plt.figure()

        self.data = np.rot90(tracking.scan.get_data()[:, 0, :])
        self.scan_time = tracking.scan.time
        self.n_scan_frames = self.data.shape[-1]
        self.dt_scan = tracking.scan.voxdim[-1]

        self.useblit = self.fig.canvas.supports_blit
        self.interval_limit = 1
        self.frame_step = 1
        if not self.useblit:
            self.interval_limit = 100
            warnings.warn(
                "Matplotlib figure does not support blit. Animation will not be optimal. This happens because the backend used does not support matplotlib blitting."
            )

        self.bm = BlitManager(self.fig.canvas)

        self.ax_fus = self.fig.add_axes([0.08, 0.12, 0.42, 0.8])
        self.ax_vid = self.fig.add_axes([0.54, 0.12, 0.42, 0.8])

        self.current_video_frame = 0
        self.current_fus_frame = 0
        self.current_time = 0

        self.cap = cv2.VideoCapture(str(tracking.video_filepath))
        if self.cap.isOpened() == False:
            raise cv2.error("Error opening video stream or file")

        cropping = tracking.metadata["data"].get("cropping_parameters", [0, -1, 0, -1])
        self.x_crop = cropping[:2]
        self.y_crop = cropping[2:]
        self.grab_first_frame()

        self.fus_im = self.ax_fus.imshow(self.data[..., 0], cmap="gray")
        self.bm.add_artist(self.fus_im)

        self.ax_xlabel = self.fig.add_axes([0.35, 0.003, 0.3, 0.07], frameon=False)
        self.ax_xlabel.xaxis.set_visible(False)
        self.ax_xlabel.yaxis.set_visible(False)

        self.time_stamp = self.ax_xlabel.annotate(
            "current video frame: {vfr:05} | current fUS frame: {fr:04} | time: {sec:06.2f} s".format(
                vfr=self.current_video_frame,
                fr=self.current_fus_frame,
                sec=self.current_time,
            ),
            (0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="white"),
            animated=self.useblit,
        )
        self.bm.add_artist(self.time_stamp)

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
        self.custom_ani.add_callback(self.next_frame)
        self.custom_ani.add_callback(self.bm.update)

    def onkeypress(self, event):
        if event.key == " ":
            self.play()

        elif event.key == "-":
            self.anim_interval *= 2
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
        if event.inaxes in (self.ax_fus, self.ax_vid):
            # self.current_time = event.xdata
            print(event.x)
            # self.current_video_frame = int(event.xdata // (1/self.video_fps))

            if not self.is_playing:
                self.grab_frame()
                self.bm.update()

    def grab_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_video_frame)
        ret, frame = self.cap.read()

        self.vid = self.ax_vid.imshow(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ],
            animated=self.useblit,
        )
        self.bm.add_artist(self.vid)
        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.max_video_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def next_frame(self):
        # handles the next frame, if outside of the time dimension, goes back to the beginning
        self.current_video_frame += self.frame_step
        if self.current_video_frame >= self.max_video_frames:
            self.current_video_frame -= self.max_video_frames

        self.grab_frame()

    def grab_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_video_frame)
        _, frame = self.cap.read()
        self.vid.set_array(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ]
        )
        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3

        self.current_fus_frame = self._get_fus_frame_from_time(self.current_time)
        self.fus_im.set_array(self.data[..., self.current_fus_frame])

        self.time_stamp.set_text(
            "current video frame: {vfr:05} | current fUS fr: {fr:04} | time: {sec:06.2f} s".format(
                vfr=self.current_video_frame,
                fr=self.current_fus_frame,
                sec=self.current_time,
            )
        )

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

    def _get_fus_frame_from_time(self, time):
        idx = list(np.where(time < self.scan_time)[0])
        if not idx:
            idx = [self.scan_time.shape[-1] - 1]

        return idx[0]
