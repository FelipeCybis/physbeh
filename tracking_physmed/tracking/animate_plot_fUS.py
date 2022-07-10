import cv2
import warnings
import numpy as np

from tracking_physmed.utils import BlitManager

class Animate_plot_fUS:
    def __init__(self, fig, ax, scan):

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
        self.ax.set_position([0.4, 0.13, 0.55, 0.75])
        self.ax_fus = self.fig.add_axes([0.05, 0.13, 0.3, 0.75])

        self.current_frame = 0
        self.current_time = self.scan_time[0]

        self.im = self.ax_fus.imshow(self.data[..., 0], cmap="gray")
        self.bm.add_artist(self.im)

        self.ax_fus.set(xlabel="X pixel", ylabel="Y pixel")

        self.ax_xlabel = self.fig.add_axes([0.55, 0.002, 0.3, 0.07], frameon=False)
        self.ax_xlabel.xaxis.set_visible(False)
        self.ax_xlabel.yaxis.set_visible(False)

        self.time_stamp = self.ax_xlabel.annotate(
            "current fUS fr: {fr:05} | time: {sec:06.2f} s".format(
                fr=self.current_frame, sec=self.current_time
            ),
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
            self.current_frame = int(event.xdata // self.dt_scan)
            if self.current_frame >= self.n_scan_frames:
                self.current_frame = int(self.n_scan_frames - 1)

            if not self.is_playing:
                self.grab_frame(self.current_frame)
                self.bm.update()

    def next_frame(self):
        # handles the next frame, if outside of the time dimension, goes back to the beginning
        self.current_frame += self.frame_step
        if self.current_frame >= self.n_scan_frames:
            self.current_frame -= self.n_scan_frames

        self.current_time = self.scan_time[self.current_frame]
        if (
            self.current_time > self.ax.get_xlim()[1]
            or self.current_time < self.ax.get_xlim()[0]
        ):
            if self.ax.get_xlim()[0] >= 0 and self.ax.get_xlim()[0] < self.scan_time[-1]:
                new_time = self.ax.get_xlim()[0]
            else:
                new_time = 0
            self.current_frame = int(new_time // self.dt_scan)
            self.current_time = self.scan_time[self.current_frame]

        self.grab_frame(self.current_frame)

    def grab_frame(self, frame):
        # simply grabbing requested frame
        # done like this to maybe add a goto frame capability
        self.im.set_array(self.data[..., self.current_frame])

        self.play_bar.set_xdata([self.current_time, self.current_time])
        self.time_stamp.set_text(
            "current fUS fr: {fr:05} | time: {sec:06.2f} s".format(
                fr=self.current_frame, sec=self.current_time
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