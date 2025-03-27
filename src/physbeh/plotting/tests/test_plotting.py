import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes as mpl_Axes

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

    fig, ax = plot_func(tracking, animate=True)

    # plot_position do not accept axes
    if plot_func not in [plot_position]:
        fig, ax = plt.subplots()
        beh_fig, beh_ax = plot_func(tracking, axes=ax)

        assert fig == beh_fig.figure
        assert beh_ax == ax

    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    beh_fig, beh_ax = plot_func(tracking, figure=subfigs[0])

    if isinstance(beh_ax, tuple):
        beh_ax = beh_ax[0]

    assert beh_ax.figure.figure == beh_fig.figure
    assert subfigs[0].figure == beh_fig.figure
