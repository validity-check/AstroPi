import logging
import math
import os
import queue
import time
import typing
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from astro_pi_replay.custom_types import (
    DEFAULT_X_Y_Z_DICT,
    RGB,
    RGBC,
    RollPitchYawDict,
    XYZDict,
)
from astro_pi_replay.executor import AstroPiExecutor
from astro_pi_replay.preview.teardown_protocol import SupportsBackgroundTaskTeardown
from astro_pi_replay.sense_hat.abstract_sense_hat import (
    SenseHatAPI,
    SenseHatColourSensorAPI,
    SenseHatStickAPI,
)
from astro_pi_replay.sense_hat.display import SenseHatDisplay

logger = logging.getLogger(__name__)


def xyzdict_reducer(df: pd.DataFrame) -> XYZDict:
    return df.rename(lambda col: col.split("_")[-1]).to_dict()


def to_dict_reducer(df: pd.DataFrame) -> dict:
    return df.to_dict()


def SenseHatColourSensorAdapter(executor: AstroPiExecutor) -> SenseHatColourSensorAPI:
    class _SenseHatColourSensorAdapter(SenseHatColourSensorAPI):
        def __init__(self):
            super().__init__(int(), int(), object())
            self._integration_cycles = 1

        # Private
        def _scale(self, value) -> int:
            """Scales from a normalised value to an
            approximate raw value by reversing the
            steps in the original SenseHat module"""
            return value * (self.max_raw // 256)

        # Public

        @property
        @executor.sense_hat_replay(col_names=["blue"])
        def blue(self) -> int:
            return super().blue

        @property
        def blue_raw(self) -> int:
            return self._scale(self.blue)

        @property
        @executor.sense_hat_replay(col_names=["clear"])
        def clear(self) -> int:
            return super().clear

        @property
        def clear_raw(self) -> int:
            return self._scale(self.clear)

        @property
        @executor.sense_hat_replay(
            col_names=["red", "green", "blue", "clear"], reducer=lambda s: tuple(s[:4])
        )
        def colour(self) -> RGBC:
            return super().colour

        @property
        def colour_raw(self) -> RGBC:
            colour: RGBC = self.colour
            return typing.cast(RGBC, tuple(map(lambda x: self._scale(x), colour)))

        @property
        def enabled(self) -> bool:
            return True

        @enabled.setter
        def enabled(self, _: bool) -> None:
            pass

        @property
        def gain(self) -> int:
            return 1

        @gain.setter
        def gain(self, _: int) -> None:
            pass

        @property
        @executor.sense_hat_replay(col_names=["green"])
        def green(self) -> int:
            return super().green

        @property
        def green_raw(self) -> int:
            return self._scale(self.green)

        @property
        def integration_cycles(self) -> int:
            return self._integration_cycles

        @integration_cycles.setter
        def integration_cycles(self, value: int) -> None:
            self._integration_cycles = value

        @property
        def integration_time(self) -> float:
            return 0.0024

        @property
        def max_raw(self) -> int:
            return 1024

        @property
        @executor.sense_hat_replay(col_names=["red"])
        def red(self) -> int:
            return super().red

        @property
        def red_raw(self) -> int:
            return self._scale(self.red)

        @property
        def rgb(self) -> RGB:
            return self.colour[:3]

    return _SenseHatColourSensorAdapter()


def SenseHatAdapter(maybe_executor: Optional[AstroPiExecutor] = None) -> SenseHatAPI:
    executor: AstroPiExecutor
    if maybe_executor is None:
        executor = AstroPiExecutor()
    else:
        executor = maybe_executor

    class _SenseHatAdapter(SenseHatAPI, SupportsBackgroundTaskTeardown):
        """
        This is an object that conforms to the SenseHat interface
        that returns default values for every function call.
        For most types the default value is obvious, but
        check check abstract_sense_hat.py if in doubt.
        """

        # TODO pass from env or estimate
        # using g = \frac{GM}{r^{2}} where r is dependent on height of ISS
        _ACCELERATION_OF_GRAVITY: float = 9.81
        # This array should be displayed with a clockwise 90 degree rotation
        # this is done in self._display
        _image: np.ndarray = np.zeros((8, 8, 3), dtype=np.uint8)
        _GAMMA_DEFAULT: list[int] = [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            15,
            17,
            18,
            20,
            21,
            23,
            25,
            27,
            29,
            31,
        ]
        _GAMMA_LOW_LIGHT: list[int] = [
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            7,
            7,
            8,
            8,
            9,
            10,
            10,
        ]

        _gamma: list[int] = _GAMMA_DEFAULT
        _text_dict: dict[str, list[list[int]]] = {}

        def __init__(self) -> None:
            text_assets = "sense_hat_text"
            super().__init__(str(), text_assets)

            # Load text assets
            dir_path = os.path.dirname(__file__)
            self._load_text_assets(
                os.path.join(dir_path, "%s.png" % text_assets),
                os.path.join(dir_path, "%s.txt" % text_assets),
            )
            self._colour = SenseHatColourSensorAdapter(executor)
            self._stick = SenseHatStickAPI()

            self._display_proc: Optional[SenseHatDisplay] = None
            self._teardown_registered = False

        def _close_window(self) -> None:
            if self._display_proc is not None:
                self._display_proc.terminate()
                self._display_proc.join(10)
                if self._display_proc.exitcode is None:
                    self._display_proc.kill()
                self._display_proc = None

        def _display(self) -> None:
            # Internal method
            try:
                import matplotlib.pyplot as plt

                plt.imshow(np.rot90(self._image, k=3))
                plt.show()
            except ModuleNotFoundError:
                pass

        def _open_window(self) -> None:
            def _open_window_internal():
                self._display_proc = SenseHatDisplay(self._image)
                self._display_proc.start()

            self._register_background_proc(_open_window_internal, self._close_window)

        def _get_char_pixels(self, s: str) -> list[list[int]]:
            """
            Internal. Safeguards the character indexed dictionary for the
            show_message function below
            """

            if len(s) == 1 and s in self._text_dict.keys():
                return list(self._text_dict[s])
            else:
                return list(self._text_dict["?"])

        def _load_text_assets(self, text_image_file: str, text_file: str) -> None:
            text_pixels: list[list[int]] = self.load_image(text_image_file, False)

            with open(text_file, "r") as f:
                loaded_text: str = f.read()
            for index, s in enumerate(loaded_text):
                start: int = index * 40
                end: int = start + 40
                char: list[list[int]] = text_pixels[start:end]
                self._text_dict[s] = char

        def _save_matrix(self):
            arr: np.ndarray = np.rot90(self._image, k=3)
            img: Image.Image = Image.fromarray(arr)
            resized_img: Image.Image = img.resize((256, 256), resample=0)
            resized_img.save(
                executor.configuration.sense_hat_snapshot_dir
                / f"{executor._state._sense_hat_snapshot_index}.png"
            )
            executor._state._sense_hat_snapshot_index += 1

        def _trim_whitespace(self, char):  # For loading text assets only
            """
            Internal. Trims white space pixels from the front and back of loaded
            text characters
            """

            psum = lambda x: sum(sum(x, []))  # noqa: E731
            if psum(char) > 0:
                is_empty = True
                while is_empty:  # From front
                    row = char[0:8]
                    is_empty = psum(row) == 0
                    if is_empty:
                        del char[0:8]
                is_empty = True
                while is_empty:  # From back
                    row = char[-8:]
                    is_empty = psum(row) == 0
                    if is_empty:
                        del char[-8:]
            return char

        @property
        def accelerometer(self) -> RollPitchYawDict:
            return self.orientation

        @property
        @executor.sense_hat_replay(
            col_names=["acc_x", "acc_y", "acc_z"], reducer=xyzdict_reducer
        )
        def accelerometer_raw(self) -> XYZDict:
            return DEFAULT_X_Y_Z_DICT

        @property
        def compass(self) -> float:
            return self.orientation["yaw"]

        @property
        @executor.sense_hat_replay(
            col_names=["mag_x", "mag_y", "mag_z"], reducer=xyzdict_reducer
        )
        def compass_raw(self) -> XYZDict:
            return DEFAULT_X_Y_Z_DICT

        @property
        def colour(self) -> SenseHatColourSensorAPI:
            return self._colour

        @property
        def color(self) -> SenseHatColourSensorAPI:
            return self.colour

        def flip_h(self, redraw: bool = True) -> list[list[int]]:
            self._image = np.fliplr(self._image)
            return self.get_pixels()

        def flip_v(self, redraw: bool = True) -> list[list[int]]:
            self._image = np.flipud(self._image)
            return self.get_pixels()

        @property
        def gamma(self) -> list[int]:
            return _SenseHatAdapter._gamma

        @gamma.setter
        def gamma(self, buffer: list[int]) -> None:
            if len(buffer) != 32:
                raise ValueError("Gamma array must be of length 32")

            if not all(b <= 31 for b in buffer):
                raise ValueError("Gamma values must be bewteen 0 and 31")

            _SenseHatAdapter._gamma = buffer

        def gamma_reset(self) -> None:
            _SenseHatAdapter._gamma = _SenseHatAdapter._GAMMA_DEFAULT

        def get_pixel(self, x: int, y: int) -> list[int]:
            return self._image[y, x].tolist()

        def get_pixels(self) -> list[list[int]]:
            return self._image.reshape((64, 3)).tolist()

        def get_temperature_from_humidity(self) -> float:
            return self.temperature

        def get_temperature_from_pressure(self) -> float:
            return self.temperature

        @property
        def gyroscope(self) -> RollPitchYawDict:
            return self.orientation

        @property
        @executor.sense_hat_replay(
            col_names=["gyro_x", "gyro_y", "gyro_z"], reducer=xyzdict_reducer
        )
        def gyroscope_raw(self) -> XYZDict:
            return DEFAULT_X_Y_Z_DICT

        @property
        @executor.sense_hat_replay(col_names=["hum"])
        def humidity(self) -> float:
            return float()

        def has_colour_sensor(self) -> bool:
            return True

        def load_image(self, file_path: str, redraw: bool = True) -> list[list[int]]:
            if not os.path.exists(file_path):
                raise IOError("%s not found" % file_path)

            img = Image.open(file_path).convert("RGB")

            pixel_list: list[list[int]] = list(map(list, img.getdata()))

            if redraw:
                self.set_pixels(pixel_list)

            return pixel_list

        @property
        def low_light(self) -> bool:
            return self.gamma == _SenseHatAdapter._GAMMA_LOW_LIGHT

        @low_light.setter
        def low_light(self, value: bool) -> None:
            if value:
                self.gamma = _SenseHatAdapter._GAMMA_LOW_LIGHT
            else:
                self.gamma_reset()

        @property
        def orientation(self) -> RollPitchYawDict:
            return dict(
                (
                    (k, math.degrees(v) % 360)
                    for k, v in self.orientation_radians.items()
                )
            )

        @property
        def orientation_radians(self) -> RollPitchYawDict:
            # The SenseHat has a LSM9DS1 IMU. Datasheet URL:
            # http://web.archive.org/web/20221210143027/https://www.st.com/resource/en/datasheet/lsm9ds1.pdf

            # Formulas taken from:
            # http://web.archive.org/web/20230605010710/https://atadiat.com/en/e-towards-understanding-imu-basics-of-accelerometer-and-gyroscope-sensors/
            # Could investigate using ahrs in the future

            # https://uk.mathworks.com/help/aeroblks/about-aerospace-coordinate-systems.html

            # 1. Approximate roll and pitch using the accel
            accel_raw = self.accelerometer_raw
            pitch_in_radians = math.asin(
                -1 * accel_raw["x"] / _SenseHatAdapter._ACCELERATION_OF_GRAVITY
            )
            roll_in_radians = math.atan2(accel_raw["y"], accel_raw["z"])

            # 2. Approximate the yaw using the magnetometer
            # with tilt compensation / projection
            compass_raw = self.compass_raw
            x_h = compass_raw["x"] * math.cos(pitch_in_radians) + compass_raw[
                "z"
            ] * math.sin(pitch_in_radians)
            y_h = (
                compass_raw["x"]
                * math.sin(roll_in_radians)
                * math.sin(pitch_in_radians)
                + compass_raw["y"] * math.cos(roll_in_radians)
                - compass_raw["z"]
                * math.sin(roll_in_radians)
                * math.cos(pitch_in_radians)
            )
            yaw_in_radians = math.atan2(-y_h, x_h)

            return {
                "roll": roll_in_radians,
                "pitch": pitch_in_radians,
                "yaw": yaw_in_radians,
            }

        @property
        @executor.sense_hat_replay(col_names=["pres"])
        def pressure(self) -> float:
            return float()

        @property
        def rotation(self) -> int:
            return self._rotation

        @rotation.setter
        def rotation(self, r: int) -> None:
            self.set_rotation(r, True)

        def set_imu_config(
            self, compass_enabled: bool, gyro_enabled: bool, accel_enabled: bool
        ) -> None:
            self._compass_enabled = compass_enabled
            self._gyro_enabled = gyro_enabled
            self._accel_enabled = accel_enabled

        def set_pixels(self, pixel_list: list[list[int]]) -> None:
            img = np.array(pixel_list, dtype=np.uint8).reshape((8, 8, 3))
            k = (-self._rotation % 360) // 90
            self._image = np.rot90(img, k=k)

            if executor.configuration.snapshot_sense_hat_display:
                self._save_matrix()

            # Emit event to subscriber # TODO refactor this out
            if self._display_proc is not None:
                try:
                    self._display_proc.queue.put_nowait(self._image)
                except queue.Full:
                    pass

        def set_pixel(self, x: int, y: int, *args) -> None:
            pixel_error = "Pixel arguments must be given as (r, g, b) or r, g, b"

            if len(args) == 1:
                pixel = args[0]
                if len(pixel) != 3:
                    raise ValueError(pixel_error)
            elif len(args) == 3:
                pixel = args
            else:
                raise ValueError(pixel_error)

            self._image[y, x] = np.array(pixel, dtype=np.uint8)

            if executor.configuration.snapshot_sense_hat_display:
                self._save_matrix()

        # TODO move to abstract class
        def set_rotation(self, r: int, redraw: bool = True) -> None:
            # rotation is defined clockwise!
            allowed_values: list[int] = [0, 90, 180, 270]
            if r not in allowed_values:
                raise ValueError("Rotation must be 0, 90, 180 or 270 degrees")
            old: int = self._rotation
            self._rotation = r
            if redraw:
                # the original library rotates clockwise, so convert
                # to anticlockwise
                num_anticlockwise_turns: int = ((old - r) % 360) // 90
                self._image = np.rot90(self._image, k=num_anticlockwise_turns)

                if executor.configuration.snapshot_sense_hat_display:
                    self._save_matrix()

        def show_letter(
            self,
            s: str,
            text_colour: list[int] = [255, 255, 255],
            back_colour: list[int] = [0, 0, 0],
        ) -> None:
            if len(s) > 1:
                raise ValueError("Only one character may be passed into this method")
            # We must rotate the pixel map left through 90 degrees when drawing
            # text, see _load_text_assets
            previous_rotation = self._rotation
            self._rotation -= 90
            if self._rotation < 0:
                self._rotation = 270
            dummy_colour = [0, 0, 0]
            pixel_list = [dummy_colour] * 8
            pixel_list.extend(self._get_char_pixels(s))
            pixel_list.extend([dummy_colour] * 16)
            coloured_pixels = [
                text_colour if pixel == [255, 255, 255] else back_colour
                for pixel in pixel_list
            ]
            self.set_pixels(coloured_pixels)
            self._rotation = previous_rotation

        def show_message(
            self,
            text_string: str,
            scroll_speed: float = 0.1,
            text_colour: list[int] = [255, 255, 255],
            back_colour: list[int] = [0, 0, 0],
        ) -> None:
            # We must rotate the pixel map left through 90 degrees when drawing
            # text, see _load_text_assets
            previous_rotation = self._rotation
            self._rotation -= 90
            if self._rotation < 0:
                self._rotation = 270
            dummy_colour = [None, None, None]
            string_padding = [dummy_colour] * 64
            letter_padding = [dummy_colour] * 8
            # Build pixels from dictionary
            scroll_pixels = []
            scroll_pixels.extend(string_padding)
            for s in text_string:
                scroll_pixels.extend(self._trim_whitespace(self._get_char_pixels(s)))
                scroll_pixels.extend(letter_padding)
            scroll_pixels.extend(string_padding)
            # Recolour pixels as necessary
            coloured_pixels = [
                text_colour if pixel == [255, 255, 255] else back_colour
                for pixel in scroll_pixels
            ]
            # Shift right by 8 pixels per frame to scroll
            scroll_length = len(coloured_pixels) // 8
            for i in range(scroll_length - 8):
                start = i * 8
                end = start + 64
                self.set_pixels(coloured_pixels[start:end])
                time.sleep(scroll_speed)
            self._rotation = previous_rotation

        @property
        def stick(self) -> SenseHatStickAPI:
            return self._stick

        @property
        def temp(self) -> float:
            return self.temperature

        @property
        @executor.sense_hat_replay(col_names=["temp"])
        def temperature(self) -> float:
            return float()

    return _SenseHatAdapter()
