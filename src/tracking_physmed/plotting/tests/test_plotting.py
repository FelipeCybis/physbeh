import matplotlib
import pytest

from tracking_physmed.plotting import (
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
        plot_center_proximity,
        plot_likelihood,
        plot_position,
        plot_position_x,
        plot_position_y,
        plot_head_direction,
    ],
)
def test_default_plottings(tracking, plot_func):
    fig, ax = plot_func(tracking)
    assert isinstance(fig, matplotlib.figure.Figure)

    if plot_func in [plot_position]:
        assert isinstance(ax, tuple)
        assert len(ax) == 2
        assert all([isinstance(a, matplotlib.axes.Axes) for a in ax])
    else:
        assert isinstance(ax, matplotlib.axes.Axes)
