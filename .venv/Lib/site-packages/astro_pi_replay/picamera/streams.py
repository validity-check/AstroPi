import logging
from typing import Optional

import astro_pi_replay.picamera.mmalobj as mo
import astro_pi_replay.picamera.original.streams as orig
from astro_pi_replay.custom_types import IO_TYPE
from astro_pi_replay.picamera.exc import PiCameraValueError
from astro_pi_replay.picamera.frames import PiVideoFrameType

logger = logging.getLogger(__name__)


class PiCameraCircularIO(orig.PiCameraCircularIO):
    def copy_to(
        self,
        output: IO_TYPE,
        size: Optional[int] = None,
        seconds: Optional[int] = None,
        frames: Optional[int] = None,
        first_frame: int = PiVideoFrameType.sps_header,
    ):
        if (size, seconds, frames).count(None) < 2:
            raise PiCameraValueError(
                "You can only specify one of size, seconds, or frames"
            )

        stream, opened = mo.open_stream(output)
        try:
            # Copy everything since we don't have frame information
            # in this picamera implementation
            for buf, _ in self._data.iter_both(False):
                stream.write(buf)
        finally:
            mo.close_stream(stream, opened)


# Re-export these from the original
BufferIO = orig.BufferIO


CircularIO = orig.CircularIO
