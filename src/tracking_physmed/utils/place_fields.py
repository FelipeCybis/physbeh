"""Functions to generate place-cells and grid-cells patterns."""

import numpy as np


def get_place_field_coords(random=False, size=30):
    """Get random set of place field coordinates.

    Parameters
    ----------
    random : bool, optional
        Whether or not to use numpy random module to generate the grid. Default is
        ``False``.
    size : int, optional
        Grid size. Default is ``30``.

    Returns
    -------
    numpy.ndarray
        The 2D grid.
    """
    if random:
        y_mu_grid, x_mu_grid = (
            np.random.randint(5, 95, size=size),
            np.random.randint(5, 95, size=size),
        )
    else:
        y_mu_grid, x_mu_grid = np.meshgrid(
            np.arange(5, 100, 22.5), np.arange(5, 100, 22.5)
        )

    return np.array([x_mu_grid.flatten(), y_mu_grid.flatten()]).T


def get_value_from_hexagonal_grid(
    x,
    y,
    xplus: float = 0.0,
    a: float = 1 / 10,
    angle: float = 8 * np.pi / 18,
):
    """Get sets of values to generate hexagonal grids.

    Parameters
    ----------
    coord : numpy.ndarray
        The ``x, y`` coordinates numpy array.
    xplus : float, optional
        The hexagonal parameter `xplus`. Default is ``0``.
    a : float, optional
        The hexagonal grid `a` parameter. Default is ``1/10``.
    angle : float, optional
        The hexagonal grid `angle` parameter. Default is ``8*np.pi/18``.

    Returns
    -------
    numpy.ndarray
        The value for the given coordinates in the hexagonal grid defined by `xplus`,
        `a` and `angle`.
    """
    R1 = np.array([a * np.sin(angle + np.pi / 3), a * np.cos(angle + np.pi / 3)])
    R2 = np.array([a * np.sin(angle), a * np.cos(angle)])
    R3 = np.array([a * np.sin(angle - np.pi / 3), a * np.cos(angle - np.pi / 3)])

    x += xplus
    re = (
        np.cos(x * R1[0] + y * R1[1])
        + np.cos(x * R2[0] + y * R2[1])
        + np.cos(x * R3[0] + y * R3[1])
    )
    return re


def set_hexagonal_parameters() -> list[tuple[float, float, float]]:
    """Get hexagonal parameters to build theoretical grid fields.

    Returns
    -------
    list of tuple of 3 float
        List containing different sets of hexagonal parameters.
    """
    x_params = np.array([[0, 9, 18], [0, 12, 24], [0, 18, 36], [0, 23, 46]])
    a_params = np.array([1 / 4, 1 / 6, 1 / 8, 1 / 10])
    angle_params = np.arange(start=0, stop=np.pi / 3, step=np.pi / 9)

    hex_params = []
    for a, x_param in zip(a_params, x_params):
        for x in x_param:
            for angle in angle_params:
                hex_params.append((x, a, angle))
    return hex_params
