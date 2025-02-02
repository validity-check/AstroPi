import datetime
import json
import logging
import multiprocessing
import re
import subprocess

import numpy as np
from PIL import Image

from astro_pi_replay import PROGRAM_NAME
from astro_pi_replay.exception import AstroPiReplayException
from astro_pi_replay.resources import get_resource

logger = logging.getLogger(__name__)

try:
    import tkinter as tk

    from PIL import ImageTk
except ImportError:
    logger.info("Tkinter not found - calling preview will not work")


class CameraPreview(multiprocessing.Process):
    def __init__(self, file: str) -> None:
        self.file = file
        self.file_width, self.file_height, self.framerate = self.fetch_metadata(file)
        multiprocessing.Process.__init__(self)

    def fetch_metadata(self, file: str) -> tuple[int, int, int]:
        """Fetches the width, height, and framerate from the given
        video file."""
        out = subprocess.run(  # nosec B603, B607
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", file],
            check=True,
            capture_output=True,
        )
        metadata = json.loads(out.stdout)["streams"][0]
        quotient = re.findall("\\d+", metadata["avg_frame_rate"])

        if len(quotient) != 2:
            raise Exception("Cannot read framerate " + f"from {self.file}")
        framerate: int = round(int(quotient[1]) / int(quotient[0]))
        return int(metadata["width"]), int(metadata["height"]), framerate

    def run(self) -> None:
        """Start the output stream to the pipe.
        Frames are outputted as fast as possioble to the pipe as
        the -re option introduced a significant delay at the beginning
        of the stream. The receiver must therefore read the input frames
        with the correct framerate - this is achieved approximately using
        the after() Tkinter method with the appropriate interval.
        """
        self.proc = subprocess.Popen(  # nosec B603, B607
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-i",
                self.file,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-an",
                "-tune",
                "zerolatency",
                "pipe:",
            ],
            stdout=subprocess.PIPE,
        )

        self.start_time: datetime.datetime = datetime.datetime.now()
        self.interval = self.framerate * 1000

        self.window = tk.Tk()
        self.window.title(" ".join(PROGRAM_NAME.split("_")).title())
        self.window.call(
            "wm",
            "iconphoto",
            self.window._w,  # type: ignore
            ImageTk.PhotoImage(file=str(get_resource("AstroPi_2021_colour.png"))),
        )
        self.label = tk.Label(self.window)
        self.label.grid()
        self.label.pack()
        self.label.after(self.interval, self._refresh)
        logger.debug("Starting Tkinter main loop...")
        self.window.mainloop()

    def start(self) -> None:
        self._target = self.run
        super().start()

    def _refresh(self) -> None:
        """
        Recursive function that consumes a frame from
        the streamed pipe. It is called approxamitely
        every self.interval milliseconds, which is
        calculated from the framerate of the file
        being streamed.
        """
        if self.proc.stdout is not None:
            logger.debug(
                f"Reading {self.file_width*self.file_height*3} bytes from pipe..."
            )
            frame_bytes = self.proc.stdout.read(self.file_width * self.file_height * 3)
            if frame_bytes:
                array = np.frombuffer(frame_bytes, np.uint8).reshape(
                    [self.file_height, self.file_width, 3]
                )
                img = Image.fromarray(array)
                imgtk = ImageTk.PhotoImage(image=img)
                logger.debug("Adding new image to window")
                self.label.configure(image=imgtk)
                self.label.image = imgtk  # type: ignore
                self.label.after(self.interval, self._refresh)
            else:
                logger.debug("No frame bytes read")
        else:
            raise AstroPiReplayException(f"Cannot preview {self.file}")
