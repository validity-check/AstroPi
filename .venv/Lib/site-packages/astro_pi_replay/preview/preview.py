import subprocess
import threading
from threading import Event

import astro_pi_replay.picamera.mmalobj as mo
from astro_pi_replay.picamera.encoders import PiEncoder
from astro_pi_replay.picamera.frames import PiVideoFrameType

# TODO add semaphore to control from the outside


class ProcStdoutConsumer(threading.Thread):
    def __init__(
        self,
        resolution: mo.PiResolution,
        final_format: str,
        encoder: PiEncoder,
        proc: subprocess.Popen[bytes],
    ):
        super().__init__()
        self.resolution = resolution
        self.final_format = final_format
        self.encoder = encoder
        self.proc = proc
        self.stop_event = Event()

    def run(self) -> None:
        """Consumes the ffmpeg subprocess"""

        is_raw: bool = (
            True
            if self.final_format in ["rgb", "rgba", "bgr", "bgra", "yuv"]
            else False
        )

        frame_size: int
        if is_raw:
            # stream frame by frame when raw
            frame_size = (
                self.resolution.width * self.resolution.height * len(self.final_format)
            )

            if self.final_format == "yuv":
                # YUV has a byte ratio of 4:6 hence multiply by 2/3
                frame_size = round(frame_size * 2 / 3)
        else:
            # When not raw, just copy 100kb at a time
            # FIXME recalculate to get desired bitrate
            frame_size = 100 * 1000

        contents: bytes
        while not self.stop_event.is_set() and self.proc.poll() is None:
            if self.proc.stdout is None:
                break
            contents = self.proc.stdout.read(frame_size)
            self.encoder.outputs[PiVideoFrameType.frame][0].write(contents)
        if self.proc.stdout is not None:
            contents = self.proc.stdout.read()
            self.encoder.outputs[PiVideoFrameType.frame][0].write(contents)

    def stop(self):
        self.stop_event.set()
