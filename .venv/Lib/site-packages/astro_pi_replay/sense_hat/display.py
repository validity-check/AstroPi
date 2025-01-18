import logging
import multiprocessing
import queue

import numpy as np
from PIL import Image

from astro_pi_replay import PROGRAM_NAME
from astro_pi_replay.resources import get_resource

logger = logging.getLogger(__name__)

try:
    import tkinter as tk

    from PIL import ImageTk
except ImportError:
    logger.debug("Tkinter not found - sense hat animated display will not work")


class SenseHatDisplay(multiprocessing.Process):
    def __init__(self, image: np.ndarray) -> None:
        self._image: np.ndarray = image
        self.queue: multiprocessing.Queue[np.ndarray] = multiprocessing.Queue()
        multiprocessing.Process.__init__(self)

    def run(self) -> None:
        self.window = tk.Tk()
        self.window.title(" ".join(PROGRAM_NAME.split("_")).title())
        self.window.call(
            "wm",
            "iconphoto",
            self.window._w,  # type: ignore
            ImageTk.PhotoImage(file=str(get_resource("AstroPi_2021_colour.png"))),
        )

        # scale to 400x400
        imgtk = self._to_tk_image(self._image)
        self.label = tk.Label(self.window, image=imgtk)
        self.label.grid()
        self.label.pack()

        self.framerate: int = 25
        self.interval: int = round(1000 / self.framerate)
        self.label.after(self.interval, self._refresh)
        logger.debug("Starting Tkinter main loop...")
        self.window.mainloop()

    def _refresh(self):
        try:
            imgtk = self._to_tk_image(self.queue.get())
            logging.debug(f"Adding new image to window: {hash(str(self._image))}")
            self.label.image = imgtk  # type: ignore
            self.label.configure(image=imgtk)
        except queue.Empty:
            pass
        self.label.after(self.interval, self._refresh)

    def _to_tk_image(
        self, im: np.ndarray, scaling_factor: int = 50
    ) -> "ImageTk.PhotoImage":
        im = np.repeat(np.repeat(im, scaling_factor, axis=0), scaling_factor, axis=1)
        img: Image.Image = Image.fromarray(im)
        return ImageTk.PhotoImage(image=img)
