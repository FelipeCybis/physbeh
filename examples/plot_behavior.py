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
# Animal speed
# ++++++++++++
# Let's try plotting animal speed

fig, ax = pb.plotting.plot_speed(tracking)
fig.show()


###############################################################################
# Animal position
# +++++++++++++++
# Or the animal's position with all the bodyparts

fig, ax = pb.plotting.plot_position(tracking, bodyparts="all")
fig.show()

###############################################################################
# Animal position in 2D
# +++++++++++++++++++++
# Or the animal's position in the 2D space

fig, ax, _ = pb.plotting.plot_position_2d(tracking)
fig.show()

###############################################################################
# Only running bouts
# ++++++++++++++++++
# A very simple algorithm detects (roughly) the animal's running bouts.

fig, ax = pb.plotting.plot_speed(
    tracking, only_running_bouts=True, plot_only_running_bouts=False
)

# This is what plot_only_running_bouts=True does under the hood, basically
pb.plotting.plot_running_bouts(tracking, axes=ax)
fig.show()

###############################################################################
# Occupancy map
# +++++++++++++
# The occupancy map is a 2D histogram of the animal's position over time.

fig, ax = pb.plotting.plot_occupancy(tracking, bins=6)
fig.show()
