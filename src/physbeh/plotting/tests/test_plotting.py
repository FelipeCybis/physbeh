from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes as mpl_Axes

from physbeh.io import load_tracking
from physbeh.plotting import (
    BehFigure,
    plot_acceleration,
    plot_angular_acceleration,
    plot_angular_velocity,
    plot_center_proximity,
    plot_corner_proximity,
    plot_head_direction,
    plot_likelihood,
    plot_position,
    plot_position_x,
    plot_position_y,
    plot_speed,
    plot_wall_proximity,
)
from physbeh.plotting.animate_decorator import TrackingAnimation

DATA_DIR = Path(__file__).parents[4] / "examples" / "data"
MOTION_DATA = (
    DATA_DIR
    / "sub-rat75_ses-20220524_task-openfield_tracksys-DLC_acq-slice32_motion.tsv"
)
VIDEO_DATA = (
    DATA_DIR
    / "sub-rat75_ses-20220524_task-openfield_tracksys-DLC_acq-slice32_video.mp4"
)
TRACKING_WITH_VIDEO = load_tracking(MOTION_DATA, video_filename=VIDEO_DATA)


@pytest.mark.parametrize(
    "plot_func",
    [
        plot_speed,
        plot_wall_proximity,
        plot_corner_proximity,
        plot_angular_acceleration,
        plot_center_proximity,
        plot_likelihood,
        plot_position,
        plot_acceleration,
        plot_position_x,
        plot_position_y,
        plot_head_direction,
        plot_angular_velocity,
    ],
)
def test_default_plottings(tracking, plot_func):
    fig, ax = plot_func(tracking)
    assert isinstance(fig, BehFigure)

    if plot_func in [plot_position]:
        assert isinstance(ax, tuple)
        assert len(ax) == 2
        assert all([isinstance(a, mpl_Axes) for a in ax])
    else:
        assert isinstance(ax, mpl_Axes)

    # plot_position do not accept axes
    if plot_func not in [plot_position]:
        # test animation
        fig, ax, anim = plot_func(TRACKING_WITH_VIDEO, animate=True)
        assert isinstance(anim, TrackingAnimation)

        fig, ax = plt.subplots()
        beh_fig, beh_ax = plot_func(tracking, axes=ax)

        assert fig == beh_fig.figure
        assert beh_ax == ax


@pytest.mark.parametrize(
    "plot_func",
    [
        plot_speed,
        plot_wall_proximity,
        plot_corner_proximity,
        plot_angular_acceleration,
        plot_center_proximity,
        plot_likelihood,
        plot_position,
        plot_acceleration,
        plot_position_x,
        plot_position_y,
        plot_head_direction,
        plot_angular_velocity,
    ],
)
def test_plottings_with_subfigures(tracking, plot_func):
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    beh_fig, beh_ax = plot_func(tracking, figure=subfigs[0])

    if isinstance(beh_ax, tuple):
        beh_ax = beh_ax[0]

    assert beh_ax.figure.figure == beh_fig.figure
    assert subfigs[0].figure == beh_fig.figure

    if plot_func not in [plot_position]:
        fig = plt.figure()
        subfigs = fig.subfigures(1, 2)
        beh_fig, beh_ax, anim = plot_func(
            TRACKING_WITH_VIDEO, figure=subfigs[0], animate=True
        )
        assert isinstance(anim, TrackingAnimation)

        if isinstance(beh_ax, tuple):
            beh_ax = beh_ax[0]

        assert beh_ax.figure.figure == beh_fig.figure
        assert subfigs[0].figure == beh_fig.figure
