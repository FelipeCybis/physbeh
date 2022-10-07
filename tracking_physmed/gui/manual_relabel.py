"""
Created on Tue Jan 04 10:01:54 2022

@author: Felipe Cybis Pereira

Script for manually relabel DLC labels
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import cv2


class Manual_relabel:
    def __init__(self, Dataframe_filepath, metadata_filepath, video):
        """GUI to manully relabel tracking frames. This is based on HDF5 files created using DeepLabCut python package.
        This GUI is ment to be used sporadically since the goal is that DeepLabCut gets to label correctly all frames.

        Parameters
        ----------
        Dataframe_filepath : path_like
            Full path to the HDF5 tracking file.
        metadata_filepath : path_like
            Full path to the metadata file of the tracking.
        video : path_like
            Full path to the labeled video file.
        """
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_axes([0.05, 0.125, 0.6, 0.8])

        self.Dataframe_filepath = Dataframe_filepath
        Dataframe = pd.read_hdf(self.Dataframe_filepath)
        self.total_frames = Dataframe.shape[0]
        self.metadata = pd.read_pickle(metadata_filepath)

        self.df_x, self.df_y, self.df_likelihood = Dataframe.values.reshape(
            (Dataframe.shape[0], -1, 3)
        ).T

        self.x1 = 0
        self.x2 = -1
        self.y1 = 0
        self.y2 = -1
        if self.metadata["data"]["cropping"] is True:
            self.y1, self.y2, self.x1, self.x2 = tuple(
                self.metadata["data"]["cropping_parameters"]
            )

        tmp_bpts = Dataframe.columns.get_level_values("bodyparts")
        self.bodyparts = tmp_bpts.values[::3]

        self.colors = plt.cm.get_cmap("plasma", len(self.bodyparts))

        self.bpts2colors = [
            (self.bodyparts[i], i, self.colors(i)) for i in range(len(self.bodyparts))
        ]

        self.im = None
        self.current_frame = 0
        self.cap = cv2.VideoCapture(str(video))
        self.grab_frame(self.current_frame)

        self.cpress = self.fig.canvas.mpl_connect("button_press_event", self.onpress)
        self.crelease = self.fig.canvas.mpl_connect(
            "button_release_event", self.onrelease
        )
        self.cmove = self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)
        self.ckey_press = self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.cpick = self.fig.canvas.mpl_connect("pick_event", self.onpick)

        ax_bounds = self.ax.get_position().bounds
        button_y_position = ax_bounds[1] + ax_bounds[3] - 0.25
        button_x_position = ax_bounds[0] + ax_bounds[2] + 0.05
        self.ax_save_button = self.fig.add_axes(
            [button_x_position, button_y_position, 0.12, 0.06]
        )
        self.save_btn = Button(self.ax_save_button, "Save labels")
        self.save_btn.on_clicked(self.save_Dataframe)

        self.last_picked_artist = None
        self.press = False
        self.move = False

        self.modyfied_frames = dict()

    def save_Dataframe(self, event):

        Dataframe = pd.read_hdf(self.Dataframe_filepath)
        scorer = Dataframe.columns.get_level_values("scorer")[0]
        for frame in self.modyfied_frames.keys():
            for bpt, coords in self.modyfied_frames[frame].items():
                Dataframe[scorer][bpt]["x"][frame] = coords[0]
                Dataframe[scorer][bpt]["y"][frame] = coords[1]
                Dataframe[scorer][bpt]["likelihood"][frame] = 0.98

        Dataframe.to_hdf(self.Dataframe_filepath, key="k", mode="w")
        print("Relabeled data correctly saved!")

    def modifying_bpt(self, x, y, frame):
        """{
        frame :  {bpt : [x_coord, y_coord],
                  bpt : [x_coord, y_coord]},
                  ]
        }"""

        if frame in self.modyfied_frames.keys():
            self.modyfied_frames[frame][self.last_picked_artist[0]] = [x, y]
        else:
            self.modyfied_frames[frame] = {self.last_picked_artist[0]: [x, y]}
        # self.last_picked_artist

    def onpick(self, event):
        for (bpt_line, bpt) in self.bpts_scatter.values():
            if event.artist == bpt_line[0]:
                self.last_picked_artist = (bpt, event.artist)
                print(self.last_picked_artist[0])

    def onpress(self, event):
        self.press = True

    def onmove(self, event):
        if self.press:
            self.move = True
            if self.last_picked_artist is not None:
                self.last_picked_artist[1].set_data(event.xdata, event.ydata)
                self.modifying_bpt(event.xdata, event.ydata, self.current_frame)
                self.fig.canvas.draw()

    def onrelease(self, event):
        if self.press and not self.move:
            self.onclick(event)

        self.last_picked_artist = None
        self.press = False
        self.move = False

    def onclick(self, event):
        if event.inaxes == self.ax and event.button == 3:

            ratio = event.xdata / (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            new_frame = int(ratio * self.total_frames)
            self.grab_frame(new_frame)

    def key_press(self, event):
        change_dict = dict(left=-1, right=1)
        if event.key in ("left", "right"):
            new_frame = self.current_frame + change_dict[event.key]
            if new_frame >= 0 and new_frame < self.total_frames:
                self.grab_frame(frame_no=new_frame)

    def grab_frame(self, frame_no):

        self.current_frame = frame_no
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        _, frame = self.cap.read()

        if self.im is not None:
            self.im.set_array(frame[self.x1 : self.x2, self.y1 : self.y2])
            for ind, (scatter, bpt) in self.bpts_scatter.items():
                alpha_value = 1
                if self.df_likelihood[ind][self.current_frame] < 0.8:
                    alpha_value = 0.4

                scatter[0].set_data(
                    self.df_x[ind][self.current_frame],
                    self.df_y[ind][self.current_frame],
                )

                if self.current_frame in self.modyfied_frames.keys():
                    if bpt in self.modyfied_frames[self.current_frame].keys():
                        coords = self.modyfied_frames[self.current_frame][bpt]
                        scatter[0].set_data(coords[0], coords[1])

                scatter[0].set_alpha(alpha_value)
                scatter[0].set_label(
                    f"{bpt} {self.df_likelihood[ind][self.current_frame]:.5f}"
                )
        else:
            self.im = self.ax.imshow(frame[self.x1 : self.x2, self.y1 : self.y2])
            self.bpts_scatter = {}
            for bpt, ind, color in self.bpts2colors:
                alpha_value = 1
                if self.df_likelihood[ind][self.current_frame] < 0.8:
                    alpha_value = 0.2
                self.bpts_scatter[ind] = (
                    self.ax.plot(
                        self.df_x[ind][self.current_frame],
                        self.df_y[ind][self.current_frame],
                        "o",
                        color=color,
                        alpha=alpha_value,
                        label=f"{bpt} {self.df_likelihood[ind][self.current_frame]:.5f}",
                        picker=True,
                    ),
                    bpt,
                )
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1))

        self.ax.set_title(
            f"Frame {str(self.current_frame)} / {self.current_frame/50:.2f} s"
        )
        self.fig.canvas.draw()
