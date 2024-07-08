import numpy as np
import pandas as pd
import pytest
from tracking_physmed.tracking import Tracking


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def likelihood():
    likelihood = np.ones(100)
    likelihood[2:6] = 0
    likelihood[40:44] = 0
    return likelihood


@pytest.fixture(scope="session")
def tracking(rng, likelihood):
    data = {
        "body_x": rng.random(100),
        "body_y": rng.random(100),
        "body_likelihood": likelihood,
        "head_x": rng.random(100),
        "head_y": rng.random(100),
        "head_likelihood": likelihood,
    }
    dataframe = pd.DataFrame(data)
    return Tracking(data=dataframe, fps=50)


def test_Tracking(tracking):
    assert isinstance(tracking, Tracking)
    # no filename was passed
    assert tracking.filename is None
    # no video_filename was passed
    assert tracking.video_filepath is None


def test_fps(tracking):
    assert tracking.fps == 50
    with pytest.raises(AttributeError, match="has no setter"):
        tracking.fps = 40


def test_time(tracking):
    # time is the indices times sampling frequency
    time = np.arange(len(tracking.Dataframe)) / tracking.fps
    np.testing.assert_equal(tracking.time, time)
    # cannot be set
    with pytest.raises(AttributeError, match="has no setter"):
        tracking.time = 40


def test_labels(tracking):
    assert all([label in ["body", "head"] for label in tracking.labels])


def test_get_index(tracking, likelihood):
    boolean_index = np.array(likelihood >= tracking.pcutout)
    np.testing.assert_equal(boolean_index, tracking.get_index("body"))

    with_parameter = np.array(likelihood >= 0.4)
    np.testing.assert_equal(with_parameter, tracking.get_index("body", 0.4))
