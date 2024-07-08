import warnings
from functools import wraps

from matplotlib.path import Path as mPath

from ..tracking import Tracking
from ..utils import BlitManager


def anim2d_decorator(plot_function):
    """Decorator to animate tracking plots synched with video of corresponding tracking.

    Parameters
    ----------
    plot_function : tracking_physmed plot function
        Usually, plot function with one axes that returns Figure and Axes.
    """

    @wraps(plot_function)
    def plot_wrapper(*args, **kwargs):
        do_anim = kwargs.pop("animate", False)
        fig, ax, lines = plot_function(*args, **kwargs)
        if do_anim:
            Trk = [arg for arg in args if isinstance(arg, Tracking)]
            if not Trk:
                Trk = [
                    value for value in kwargs.values() if isinstance(value, Tracking)
                ]
            anim = Animate_plot2D(fig, ax, lines, Trk[0].fps)
            return fig, ax, anim
        return fig, ax

    return plot_wrapper


class Animate_plot2D:
    def __init__(self, fig, ax, lines, fps):
        self.fig = fig
        self.ax = ax
        self.lines = lines
        self.ax.collections[0]._paths.clear()
        self.current_frame = 0
        self.ax.collections[0]._paths.append(mPath(lines[self.current_frame]))

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

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

        self.fps = fps
        self.current_time = self.current_frame / self.fps

        self.bm = BlitManager(self.fig.canvas)
        self.bm.add_artist(self.ax.collections[0])
        self.bm.add_artist(self.ax.title)

        self.init_title = self.ax.title.get_text()
        self.ax_timelabel = self.fig.add_axes([0.42, 0.0, 0.2, 0.07], frameon=False)
        self.ax_timelabel.xaxis.set_visible(False)
        self.ax_timelabel.yaxis.set_visible(False)

        self.ax.title.set_text(
            self.init_title
            + f"\ncurrent fr: {self.current_frame:05} |"
            + f" time: {self.current_time:06.2f} s"
        )

        # self.time_stamp = self.ax_timelabel.annotate(
        #         "current fr: {fr:05} | time: {sec:06.2f} s".format(
        #             fr=self.current_frame, sec=self.current_time
        #         ),
        #         (0.5, 0.75),
        #         xycoords="axes fraction",
        #         ha="center",
        #         va="top",
        #         bbox=dict(facecolor="white", alpha=0.6, edgecolor="white"),
        #         animated=self.useblit,
        #     )
        # self.bm.add_artist(self.time_stamp)

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
        pass

    def update_frame(self):
        self.ax.collections[0]._paths += [
            mPath(line)
            for line in self.lines[
                self.current_frame : self.current_frame + self.frame_step
            ]
        ]

        self.current_frame += self.frame_step
        self.current_time = self.current_frame / self.fps

        self.ax.title.set_text(
            self.init_title
            + f"\ncurrent fr: {self.current_frame:05} |"
            + f" time: {self.current_time:06.2f} s"
        )
        # self.time_stamp.set_text(
        #     "current fr: {fr:05} | time: {sec:06.2f} s".format(
        #         fr=self.current_frame, sec=self.current_time
        #     )
        # )

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
