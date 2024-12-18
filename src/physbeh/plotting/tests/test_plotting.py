import pytest
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure

from physbeh.plotting import (
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
    assert isinstance(fig, mpl_Figure)

    if plot_func in [plot_position]:
        assert isinstance(ax, tuple)
        assert len(ax) == 2
        assert all([isinstance(a, mpl_Axes) for a in ax])
    else:
        assert isinstance(ax, mpl_Axes)
