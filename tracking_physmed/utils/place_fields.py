import numpy as np


def get_place_field_coords(random=False, size=30):
    if random:
        y_mu_grid, x_mu_grid = np.random.randint(5, 95, size=size), np.random.randint(
            5, 95, size=size
        )
    else:
        y_mu_grid, x_mu_grid = np.meshgrid(
            np.arange(5, 100, 22.5), np.arange(5, 100, 22.5)
        )

    return np.array([x_mu_grid.flatten(), y_mu_grid.flatten()]).T


def get_value_from_hexagonal_grid(coord, xplus=0, a=1 / 10, angle=8 * np.pi / 18):
    (X, Y) = coord.T
    X += xplus

    R1 = np.array([a * np.sin(angle + np.pi / 3), a * np.cos(angle + np.pi / 3)])
    R2 = np.array([a * np.sin(angle), a * np.cos(angle)])
    R3 = np.array([a * np.sin(angle - np.pi / 3), a * np.cos(angle - np.pi / 3)])

    re = (
        np.cos(X * R1[0] + Y * R1[1])
        + np.cos(X * R2[0] + Y * R2[1])
        + np.cos(X * R3[0] + Y * R3[1])
    )
    return re


def set_hexagonal_parameters():
    x_params = np.array([[0, 9, 18], [0, 12, 24], [0, 18, 36], [0, 23, 46]])
    a_params = np.array([1 / 4, 1 / 6, 1 / 8, 1 / 10])
    angle_params = np.arange(start=0, stop=np.pi / 3, step=np.pi / 9)

    hex_params = []
    for a, x_param in zip(a_params, x_params):
        for x in x_param:
            for angle in angle_params:
                hex_params.append([x, a, angle])
    return hex_params
