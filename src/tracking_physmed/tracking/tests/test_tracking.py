import numpy as np
import numpy.typing as npt
import pytest

from tracking_physmed.tracking import Tracking


def test_Tracking(tracking: Tracking):
    assert isinstance(tracking, Tracking)
    # no filename was passed
    assert tracking.filename is None
    # no video_filename was passed
    assert tracking.video_filepath is None


def test_fps(tracking: Tracking):
    assert tracking.fps == 50
    with pytest.raises(AttributeError, match="has no setter"):
        tracking.fps = 40


def test_time(tracking: Tracking):
    # time is the indices times sampling frequency
    time = np.arange(len(tracking.Dataframe)) / tracking.fps
    np.testing.assert_equal(tracking.time, time)
    # cannot be set
    with pytest.raises(AttributeError, match="has no setter"):
        tracking.time = 40


def test_labels(tracking: Tracking):
    assert all([label in ["body", "neck", "probe"] for label in tracking.labels])


def test_get_index(tracking: Tracking, likelihood: npt.NDArray[np.float64]):
    boolean_index = np.array(likelihood >= tracking.pcutout)
    np.testing.assert_equal(boolean_index, tracking.get_index("body"))

    with_parameter = np.array(likelihood >= 0.4)
    np.testing.assert_equal(with_parameter, tracking.get_index("body", 0.4))


def test_angular_velocity(tracking: Tracking):
    ang_velocity, time, index = tracking.get_angular_velocity(
        label0="neck", label1="probe", smooth=True
    )
    assert ang_velocity.shape == (len(tracking.Dataframe),)
    assert time.shape == ang_velocity.shape
    np.testing.assert_equal(index, tracking.get_index("probe"))


def test_mock_methods(tracking: Tracking):
    tracking.get_direction_array()
    tracking.get_position_x("body")
    tracking.get_position_y("body")
    tracking.get_proximity_from_corner()
    tracking.get_proximity_from_center()
    tracking.get_speed("body")
    tracking.get_proximity_from_wall()
    tracking.get_acceleration()
    tracking.get_angular_acceleration()
