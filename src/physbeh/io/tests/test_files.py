from pathlib import Path

from physbeh import load_tracking
from physbeh.tracking import Tracking

DATA_PATH = Path(__file__).parent / "data"


def test_load_tracking():
    motion_path = (
        DATA_PATH / "sub-rat2306_ses-20210727_task-openfield_tracksys-DLC_motion.tsv"
    )
    channels_path = (
        DATA_PATH / "sub-rat2306_ses-20210727_task-openfield_tracksys-DLC_channels.tsv"
    )
    tracking = load_tracking(motion_path, channels_path)
    assert isinstance(tracking, Tracking)

    h5_path = DATA_PATH / "sub-rat2306_ses-20210727_type-exp_tracking-filtered_beh.h5"

    tracking = load_tracking(h5_path)
    assert isinstance(tracking, Tracking)
