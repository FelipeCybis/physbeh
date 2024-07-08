"""
Created on Tue Mar 16 18:12:19 2021

@author: Felipe Cybis Pereira

Instead of labelling every corner in every training images in deeplabcut,
I found easier just to create this class that receives a full video path
and asks the user input via a little matplotlib GUI to get coordinates from
top_left, top_right, bottom_left and bottom_right corners.
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class Corner_Coords:
    def __init__(
        self, videopath, function_after_done, x_crop=[None, None], y_crop=[None, None]
    ):
        self.x_crop = [int(value) if value is not None else None for value in x_crop]
        self.y_crop = [int(value) if value is not None else None for value in y_crop]

        self.text_to_write = ["top_left", "top_right", "bottom_left", "bottom_right"]
        self.pts_list = []
        self.txt_list = []
        self.coords_list = []

        self.fig = plt.figure(0, figsize=(11, 8))
        self.ax = self.fig.add_subplot(111)

        self.ax_btn_next_frame = plt.axes([0.79, 0.6, 0.15, 0.075])
        self.btn_next_frame = Button(self.ax_btn_next_frame, "Frame >>")
        self.ax_btn_prev_frame = plt.axes([0.79, 0.5, 0.15, 0.075])
        self.btn_prev_frame = Button(self.ax_btn_prev_frame, "<< Frame")
        self.ax_btn_done = plt.axes([0.79, 0.4, 0.15, 0.075])
        self.btn_done = Button(self.ax_btn_done, "Done")

        self.current_frame = 0
        self.cap = cv2.VideoCapture(str(videopath))

        if self.cap.isOpened() is False:
            print("Error opening video stream or file")
            return

        self.grab_frame(self.current_frame)

        self.press = False
        self.move = False

        self.cpress = self.fig.canvas.mpl_connect("button_press_event", self.onpress)
        self.crelease = self.fig.canvas.mpl_connect(
            "button_release_event", self.onrelease
        )
        self.cmove = self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)

        self.btn_next_frame.on_clicked(lambda: self.next_frame())
        self.btn_prev_frame.on_clicked(lambda: self.prev_frame())
        self.btn_done.on_clicked(lambda: self.done(function_after_done))

        self.fig.subplots_adjust(left=0)
        plt.show()

    def grab_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        fr = self.cap.read()[1]

        try:
            self.ax.set_array(
                fr[
                    self.y_crop[0] : self.y_crop[1],
                    self.x_crop[0] : self.x_crop[1],
                    ::-1,
                ]
            )
        except AttributeError:
            self.ax.imshow(
                fr[
                    self.y_crop[0] : self.y_crop[1],
                    self.x_crop[0] : self.x_crop[1],
                    ::-1,
                ]
            )

        self.ax.set(
            title=(
                f"Current frame: {self.current_frame}\n"
                "Left click to select a corner, right click to remove it"
            )
        )
        self.fig.canvas.draw()

    def next_frame(self):
        self.current_frame += 1
        self.grab_frame(self.current_frame)

    def prev_frame(self):
        if self.current_frame == 0:
            return
        else:
            self.current_frame -= 1
            self.grab_frame(self.current_frame)

    def done(self, function_after_done):
        if len(self.coords_list) == 4:
            plt.close(0)
            print(
                f"Top left corner: {self.coords_list[0]}",
                f"\nTop right corner: {self.coords_list[1]}",
                f"\nBottom left corner: {self.coords_list[2]}",
                f"\nBottom right corner: {self.coords_list[3]}",
            )
            function_after_done(self.coords_list)
        else:
            self.ax.annotate(
                text="Not all corners were labeled!",
                xy=(0.76, 0.35),
                xycoords="figure fraction",
                color="red",
                bbox=dict(facecolor="white", alpha=0.2, edgecolor="red"),
            )
            self.fig.canvas.draw()

    def onpress(self, event):
        self.press = True

    def onmove(self, event):
        if self.press:
            self.move = True

    def onrelease(self, event):
        if self.press and not self.move:
            self.onclick(event)

        self.press = False
        self.move = False

    def onclick(self, event):
        if event.inaxes == self.ax:
            if len(self.pts_list) < 4:
                if event.button == 1:  # LEFT BUTTON
                    self.pts_list.append(self.ax.plot(event.xdata, event.ydata, "+"))
                    self.txt_list.append(
                        self.ax.annotate(
                            text=self.text_to_write[len(self.pts_list) - 1],
                            xy=(event.xdata, event.ydata),
                            xytext=(event.xdata, event.ydata),
                            xycoords="data",
                            bbox=dict(facecolor="white", alpha=0.2),
                        )
                    )

                    self.coords_list.append([int(event.xdata), int(event.ydata)])

            if event.button == 3:  # RIGHT BUTTON
                # deleting last added coordinate
                self.pts_list[-1][0].remove()
                del self.pts_list[-1]
                self.txt_list[-1].remove()
                del self.txt_list[-1]
                self.coords_list.pop()

            self.fig.canvas.draw()
