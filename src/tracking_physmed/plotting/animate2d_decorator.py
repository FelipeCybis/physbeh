from functools import wraps

import cv2
import numpy as np
from matplotlib.path import Path as mPath

from tracking_physmed.plotting.animate_decorator import TrackingAnimation
from tracking_physmed.tracking import Tracking


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
                Trk = [
                    value for value in kwargs.values() if isinstance(value, Tracking)
                ]
            anim = Animate_plot2D(
                figure=fig,
                axes=ax,
                time_array=Trk[0].time,
                video_path=Trk[0].video_filepath,
                arena=Trk[0].arena,
                show_timestamp=True,
                **anim_kwargs,
            )
            return fig, ax, anim
        return fig, ax

    return plot_wrapper


class Animate_plot2D(TrackingAnimation):
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
            y_crop[1] = int(camera_height)

        if x_crop[1] < 0:
            x_crop[1] = int(camera_width)

        self.y_slice = slice(*y_crop)
        self.x_slice = slice(*x_crop)

        self.extent = arena.get_extent(camera_width, camera_height)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        _, frame = self.cap.read()

        self.vid.set_array(frame[self.y_slice, self.x_slice, ::-1])
