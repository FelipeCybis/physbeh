import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import cv2
from .tracking import Tracking

import warnings


def anim_decorator(plot_function):
    def plot_wrapper(*args, **kwargs):
        do_anim = kwargs.pop("animate", False)
        fig, ax = plot_function(*args, **kwargs)
        if do_anim:
            Trk = [arg for arg in args if isinstance(arg, Tracking)]
            if not Trk:
                Trk = [
                    value for value in kwargs.values() if isinstance(value, Tracking)
                ]

            cropping = (
                Trk[0].metadata["data"].get("cropping_parameters", [0, -1, 0, -1])
            )
            xcrop = cropping[:2]
            ycrop = cropping[2:]
            anim = Animate_plot(
                fig, ax, video_path=Trk[0].video_filepath, x_crop=xcrop, y_crop=ycrop
            )
        return anim

    return plot_wrapper


class Animate_plot:
    def __init__(self, fig, ax, video_path=None, x_crop=[0, -1], y_crop=[0, -1]):

        self.fig = fig
        self.ax = ax

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        init_pos = 0.55
        self.vid = None
        self.useblit = self.fig.canvas.supports_blit
        if not self.useblit:
            warnings.warn(
                "Matplotlib figure does not support blit. Animation will not be optimal. This happens because the backend used does not support matplotlib blitting."
            )

        if video_path:

            self.ax.set_xlabel("")

            self.y_crop = y_crop
            self.x_crop = x_crop

            self.ax.set_position([0.4, 0.13, 0.55, 0.7])
            self.ax_vid = self.fig.add_axes([0.05, 0.13, 0.3, 0.8])

            self.current_frame = 0
            self.current_time = 0

            self.cap = cv2.VideoCapture(str(video_path))
            if self.cap.isOpened() == False:
                print("Error opening video stream or file")
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
                # xytext=(10, -10),
                # textcoords="offset points",
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="white"),
                animated=self.useblit,
            )
            self.time_stamp_not_playing = self.ax_xlabel.annotate(
                "current fr: {fr:05} | time: {sec:06.2f} s".format(
                    fr=self.current_frame, sec=self.current_time
                ),
                (0.5, 0.75),
                xycoords="axes fraction",
                # xytext=(10, -10),
                # textcoords="offset points",
                ha="center",
                va="top",
                alpha=1,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="white"),
                animated=False,
            )

        self.anim_mode = 0

        self.bar_height = self.ax.get_ylim()
        self.start_x_pos = self.ax.get_xlim()[0]

        self.play_bar = self.ax.axvline(
            self.current_time, color="gray", animated=self.useblit
        )
        self.stopped_bar = self.ax.axvline(self.current_time, color="gray", alpha=1)

        self.anim_interval = 1
        self.dx = 0.1

        self.press = False
        self.move = False
        self.xlim_changed = False
        self.ylim_changed = False

        self.cpress = self.fig.canvas.mpl_connect("button_press_event", self.onpress)
        self.crelease = self.fig.canvas.mpl_connect(
            "button_release_event", self.onrelease
        )
        self.cmove = self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)

        self.is_playing = False
        self.is_starting = True

        self.ax_btn_play = self.fig.add_axes([0.6, 0.87, 0.045, 0.075])
        self.btn_play = Button(self.ax_btn_play, ">")
        self.btn_play.on_clicked(lambda l: self.play())

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
            print(self.current_frame, event.xdata, self.fps)
            if not self.is_playing:
                self.grab_frame(self.current_frame)
                self.play_bar.set_xdata([self.current_time, self.current_time])
                print(self.current_time)
                self.stopped_bar.set_xdata([self.current_time, self.current_time])
                self.time_stamp_not_playing.set_text(
                    "current fr: {fr:05} | time: {sec:06.2f} s".format(
                        fr=self.current_frame, sec=self.current_time
                    )
                )

            self.fig.canvas.draw()

    def grab_first_frame(self):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        self.vid = self.ax_vid.imshow(
            frame[
                self.y_crop[0] : self.y_crop[1], self.x_crop[0] : self.x_crop[1], ::-1
            ],
            animated=self.useblit,
        )

        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.max_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def grab_frame(self, fr):
        print(fr)
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

    def toggle_animation(self):
        self.time_stamp_not_playing.set_alpha(
            not self.time_stamp_not_playing.get_alpha()
        )
        self.time_stamp_not_playing.set_text(self.time_stamp.get_text())

        self.stopped_bar.set_alpha(not self.stopped_bar.get_alpha())
        self.stopped_bar.set_xdata([self.current_time, self.current_time])
        self.fig.canvas.draw()

    def update_data(self):
        if self.current_frame < self.max_frames:
            self.current_frame += 1
            yield self.current_frame

    def update_frame(self, frame):
        if frame == self.max_frames:
            frame = 0
        self.grab_frame(frame)
        self.play_bar.set_xdata([self.current_time, self.current_time])
        self.time_stamp.set_text(
            "current fr: {fr:05} | time: {sec:06.2f} s".format(
                fr=self.current_frame, sec=self.current_time
            )
        )

        return (
            self.vid,
            self.play_bar,
            self.time_stamp,
        )

    def play(self):

        if self.is_playing:
            self.ani.event_source.stop()
            self.is_playing = False

            self.toggle_animation()

        elif not self.is_playing:
            self.is_playing = True
            self.toggle_animation()

            if self.is_starting:

                self.ani = animation.FuncAnimation(
                    self.fig,
                    func=self.update_frame,
                    frames=self.update_data,
                    interval=self.anim_interval,
                    blit=self.useblit,
                )

                self.is_starting = False
            elif not self.is_starting:
                self.ani.event_source.start()

            self.fig.canvas.draw()
