"""
Visualization of behavior
=========================

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
# Behavior plottings
# ------------------
#
# Plotting built-in behavioral parameters can be quite easy. See the available plotting
# functions here :mod:`pb.plotting` or printing `pb.plotting.__all__`.

print(pb.plotting.__all__)


###############################################################################
# Let's try plotting animal speed

fig, ax = pb.plotting.plot_speed(tracking)
fig.show()


###############################################################################
# Or the animal's position with all the bodyparts

fig, ax = pb.plotting.plot_position(tracking, bodyparts="all")
fig.show()
