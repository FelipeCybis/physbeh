"""
Animations of behavior
======================

"""

import physbeh as pb

###############################################################################
# Loading data
# ------------
# We will load a tracking data.

tracking = pb.load_tracking(
    "data/sub-rat75_ses-20220524_task-openfield_tracksys-DLC_acq-slice32_motion.tsv",
    video_filename="data/sub-rat75_ses-20220524_task-openfield_tracksys-DLC_acq-slice32_video.mp4",
)

###############################################################################
# Animations
# ----------
#
# Usually, every plotting function in ``physbeh`` has a decorator that allows them to be
# animated **(if the video loaded with the tracking!!)**. Passing ``animate=True`` to
# the plotting function should be all it takes. This will make the plotting function
# return a third object, the animation object.
#
# .. note::
#    Note we are slicing the tracking data to make the animation shorter. By default it
#    will animate the whole data.
#
# Let's try an animation of the animal instantaneous speed.

fig, ax, anim = pb.plotting.plot_speed(
    tracking[30:120], figsize=(13, 4.5), animate=True, animate__blit=True
)
anim.video_axes.set_xlabel("X (cm)")
anim.video_axes.set_ylabel("Y (cm)")

fig.show()

###############################################################################
# Or the animal's heading.

fig, ax, anim = pb.plotting.plot_head_direction(
    tracking[500:600], figsize=(13, 4.5), animate=True, animate__blit=True
)
anim.video_axes.set_xlabel("X (cm)")
anim.video_axes.set_ylabel("Y (cm)")

fig.show()
