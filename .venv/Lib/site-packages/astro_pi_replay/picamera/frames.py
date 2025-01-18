import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class PiVideoFrameType:
    frame = 0
    key_frame = 1
    sps_header = 2
    motion_data = 3


class PiVideoFrame(NamedTuple):
    index: int  # type: ignore
    frame_type: int
    frame_size: int
    video_size: int
    split_size: int
    timestamp: int
    complete: bool
