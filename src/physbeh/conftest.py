"""Fixtures for the physbeh package."""

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from physbeh.tracking import Tracking


def pytest_configure():
    """Use Agg so that no figures pop up."""
    if matplotlib is not None:
        matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def close_all() -> None:
    """Close all matplotlib figures."""
    if matplotlib is not None:
        import matplotlib.pyplot as plt

        plt.close("all")  # takes < 1 us so just always do it


# --------------------------------------- RNG ---------------------------------------  #


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Return a seeded random number generator.

    Returns
    -------
    numpy.random.Generator
        A numpy generator with seed 42.
    """
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def likelihood() -> npt.NDArray[np.float64]:
    """Return a likelihood array.

    Returns
    -------
    numpy.ndarray
        A likelihood array.
    """
    n_frames = 500
    likelihood = np.ones(n_frames)
    likelihood[2:6] = 0
    likelihood[40:44] = 0
    return likelihood


# -------------------------------------- TRACKING -----------------------------------  #


@pytest.fixture()
def tracking(rng: np.random.Generator, likelihood: npt.NDArray[np.float64]) -> Tracking:
    """Return a tracking object with random data.

    Parameters
    ----------
    rng : np.random.Generator
        The random number generator.
    likelihood : np.ndarray
        The likelihood array to be used in the fixture (that is a fixture itself).

    Returns
    -------
    Tracking
        The tracking object.
    """
    data = {
        "body_x": rng.random(likelihood.size),
        "body_y": rng.random(likelihood.size),
        "body_likelihood": likelihood,
        "neck_x": rng.random(likelihood.size),
        "neck_y": rng.random(likelihood.size),
        "neck_likelihood": likelihood,
        "probe_x": rng.random(likelihood.size),
        "probe_y": rng.random(likelihood.size),
        "probe_likelihood": likelihood,
    }
    dataframe = pd.DataFrame(data)

    units = ["px"] * len(dataframe.columns)
    for col, unit in zip(dataframe.columns, units):
        unit = "pint[" + unit + "]" if "_likelihood" not in col else "pint[]"
        dataframe[col] = pd.Series(dataframe[col], dtype=unit)

    return Tracking(data=dataframe, fps=50)
