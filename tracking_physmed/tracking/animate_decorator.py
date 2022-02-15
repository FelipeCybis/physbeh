import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import cv2
from .tracking import Tracking
from tracking_physmed.utils import BlitManager

import warnings


def anim_decorator(plot_function):
    """Decorator to animate tracking plots synched with the video file of corresponding tracking.

    Parameters
    ----------
    plot_function : tracking_physmed plot function
        Usually, plot function with one axes that returns Figure and Axes.
    """
    def plot_wrapper(*args, **kwargs):
        do_anim = kwargs.pop("animate", False)
        fig, ax = plot_function(*args, **kwargs)
        if do_anim:
            Trk = [arg for arg in args if isinstance(arg, Tracking)]
            if not Trk:
                Trk = [
                    value for value in kwargs.values() if isinstance(value, Tracking)
                ]

            if Trk[0].video_filepath is None:
                return fig, ax
                
            cropping = (
                Trk[0].metadata["data"].get("cropping_parameters", [0, -1, 0, -1])
            )
            xcrop = cropping[:2]
            ycrop = cropping[2:]
            anim = Animate_plot(
                fig, ax, video_path=Trk[0].video_filepath, x_crop=xcrop, y_crop=ycrop
            )
        return fig, ax, anim

    return plot_wrapper


class Animate_plot:
    def __init__(self, fig, ax, video_path=None, x_crop=[0, -1], y_crop=[0, -1]):

        self.fig = fig
        self.ax = ax

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        self.vid = None
        self.useblit = self.fig.canvas.supports_blit
        self.interval_limit = 1
        if not self.useblit:
            self.interval_limit = 100
            warnings.warn(
                "Matplotlib figure does not support blit. Animation will not be optimal. This happens because the backend used does not support matplotlib blitting."
            )

        self.bm = BlitManager(self.fig.canvas)

        if video_path:

            self.ax.set_xlabel("")

            self.y_crop = y_crop
            self.x_crop = x_crop

            self.ax.set_position([0.4, 0.13, 0.55, 0.75])
            self.ax_vid = self.fig.add_axes([0.05, 0.13, 0.3, 0.75])

            self.current_frame = 0
            self.current_time = 0

            self.cap = cv2.VideoCapture(str(video_path))
            if self.cap.isOpened() == False:
                raise cv2.error("Error opening video stream or file")
            self.grab_first_frame()

            self.ax_vid.set(xlabel="X pixel", ylabel="Y pixel")

            self.ax_xlabel = self.fig.add_axes([0.55, 0.002, 0.3, 0.07], frameon=False)
            self.ax_xlabel.xaxis.set_visible(False)
            self.ax_xlabel.yaxis.set_visible(False)

            self.time_stamp = self.ax_xlabel.annotate(
                "current fr: {fr:05} | time: {sec:06.2f} s".format(
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
        self.ckeypress = self.fig.canvas.mpl_connect('key_press_event',self.onkeypress)
        self.ckeyrelease = self.fig.canvas.mpl_connect('key_release_event',self.onkeyrelease)

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
            
        print("Frame step:", self.frame_step, "Timer interval:", int(self.anim_interval), "msec")
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
                self.time_stamp.set_text(
                    "current fr: {fr:05} | time: {sec:06.2f} s".format(
                        fr=self.current_frame, sec=self.current_time
                    )
                )

                self.bm.update()

    def grab_first_frame(self):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        self.vid = self.ax_vid.imshow(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ],
            animated=self.useblit,
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

        self.vid.set_array(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ]
        )

        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3


    def update_frame(self):

        if self.current_time < self.ax.get_xlim()[1] and self.current_time > self.ax.get_xlim()[0]:
            self.current_frame += self.frame_step
        elif self.current_frame == self.max_frames:
            self.current_frame = 0
        else:
            new_time = self.ax.get_xlim()[0] if self.ax.get_xlim()[0] >= 0 else 0
            self.current_frame = new_time // (1/self.fps) + 1

        self.grab_frame(self.current_frame)

        self.play_bar.set_xdata([self.current_time, self.current_time])
        self.time_stamp.set_text(
            "current fr: {fr:05} | time: {sec:06.2f} s".format(
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
