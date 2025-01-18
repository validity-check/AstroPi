#!/usr/bin/python
from typing import Callable, Optional

from astro_pi_replay.custom_types import (
    DEFAULT_CALLABLE,
    DEFAULT_RGB_TUPLE,
    DEFAULT_RGBC_TUPLE,
    DEFAULT_ROLL_PITCH_YAW_DICT,
    DEFAULT_X_Y_Z_DICT,
    RGB,
    RGBC,
    InputEvent,
    RollPitchYawDict,
    XYZDict,
)


class SenseHatColourSensorAPI:
    def __init__(self, gain: int, integration_cycles: int, interface: object):
        self.interface: object = interface

    @property
    def blue(self) -> int:
        return int()

    @property
    def blue_raw(self) -> int:
        return int()

    @property
    def clear(self) -> int:
        return int()

    @property
    def clear_raw(self) -> int:
        return int()

    @property
    def brightness(self) -> int:
        return self.clear_raw

    @property
    def colour(self) -> RGBC:
        return DEFAULT_RGBC_TUPLE

    @property
    def color(self) -> RGBC:
        return self.colour

    @property
    def colour_raw(self) -> RGBC:
        return DEFAULT_RGBC_TUPLE

    @property
    def color_raw(self) -> RGBC:
        return self.colour_raw

    @property
    def enabled(self) -> bool:
        return bool()

    @enabled.setter
    def enabled(self, status: bool) -> None:
        pass

    @property
    def gain(self) -> int:
        return int()

    @gain.setter
    def gain(self, gain: int) -> None:
        pass

    @property
    def green(self) -> int:
        return int()

    @property
    def green_raw(self) -> int:
        return int()

    @property
    def integration_cycles(self) -> int:
        return int()

    @integration_cycles.setter
    def integration_cycles(self, integration_cycles: int) -> None:
        pass

    @property
    def integration_time(self) -> float:
        return float()

    @property
    def max_raw(self) -> int:
        return int()

    @property
    def red(self) -> int:
        return int()

    @property
    def red_raw(self) -> int:
        return int()

    @property
    def rgb(self) -> RGB:
        return DEFAULT_RGB_TUPLE


class SenseHatStickAPI:
    SENSE_HAT_EVDEV_NAME: str = str()
    EVENT_FORMAT: str = str()
    EVENT_SIZE: int = int()
    EV_KEY: int = int()

    STATE_RELEASE: int = int()
    STATE_PRESS: int = int()
    STATE_HOLD: int = int()

    KEY_UP: int = int()
    KEY_LEFT: int = int()
    KEY_RIGHT: int = int()
    KEY_DOWN: int = int()
    KEY_ENTER: int = int()

    def close(self) -> None:
        pass

    @property
    def direction_any(self) -> Callable:
        return DEFAULT_CALLABLE

    @direction_any.setter
    def direction_any(self, value: Callable) -> None:
        pass

    @property
    def direction_down(self) -> Callable:
        return DEFAULT_CALLABLE

    @direction_down.setter
    def direction_down(self, value: Callable) -> None:
        pass

    @property
    def direction_left(self) -> Callable:
        return DEFAULT_CALLABLE

    @direction_left.setter
    def direction_left(self, value: Callable) -> None:
        pass

    @property
    def direction_middle(self) -> Callable:
        return DEFAULT_CALLABLE

    @direction_middle.setter
    def direction_middle(self, value: Callable) -> None:
        pass

    @property
    def direction_right(self) -> Callable:
        return DEFAULT_CALLABLE

    @direction_right.setter
    def direction_right(self, value: Callable) -> None:
        pass

    @property
    def direction_up(self) -> Callable:
        return DEFAULT_CALLABLE

    @direction_up.setter
    def direction_up(self, value: Callable) -> None:
        pass

    def get_events(self) -> list[InputEvent]:
        return list()

    def wait_for_event(self, emptybuffer: bool) -> Optional[InputEvent]:
        pass


class SenseHatAPI:
    # Not sure if these belong here really...
    SENSE_HAT_FB_NAME: str = str()
    SENSE_HAT_FB_FBIOGET_GAMMA: int = int()
    SENSE_HAT_FB_FBIOSET_GAMMA: int = int()
    SENSE_HAT_FB_FBIORESET_GAMMA: int = int()
    SENSE_HAT_FB_GAMMA_DEFAULT: int = int()
    SENSE_HAT_FB_GAMMA_LOW: int = int()
    SENSE_HAT_FB_GAMMA_USER: int = int()
    SETTINGS_HOME_PATH: str = str()

    def __init__(self, imu_settings_file: str, text_assets: str):
        self._rotation = 0
        pass

    @property
    def accel(self) -> RollPitchYawDict:
        return self.accelerometer

    @property
    def accel_raw(self) -> XYZDict:
        return self.accelerometer_raw

    @property
    def accelerometer(self) -> RollPitchYawDict:
        return DEFAULT_ROLL_PITCH_YAW_DICT

    @property
    def accelerometer_raw(self) -> XYZDict:
        return DEFAULT_X_Y_Z_DICT

    def clear(self, *args) -> None:
        pass

    @property
    def compass(self) -> float:
        return float()

    @property
    def compass_raw(self) -> XYZDict:
        return DEFAULT_X_Y_Z_DICT

    @property
    def colour(self) -> SenseHatColourSensorAPI:
        return SenseHatColourSensorAPI(int(), int(), object())

    @property
    def color(self) -> SenseHatColourSensorAPI:
        return self.colour

    def flip_h(self, redraw: bool = True) -> list[list[int]]:
        return list()

    def flip_v(self, redraw: bool = True) -> list[list[int]]:
        return list()

    @property
    def gamma(self) -> list[int]:
        return list()

    @gamma.setter
    def gamma(self, buffer: list[int]) -> None:
        pass

    def gamma_reset(self) -> None:
        pass

    def get_accelerometer(self) -> RollPitchYawDict:
        return self.accelerometer

    def get_accelerometer_raw(self) -> XYZDict:
        return self.accelerometer_raw

    def get_compass(self) -> float:
        return self.compass

    def get_compass_raw(self) -> XYZDict:
        return self.compass_raw

    def get_gyroscope(self) -> RollPitchYawDict:
        return self.gyroscope

    def get_gyroscope_raw(self) -> XYZDict:
        return self.gyroscope_raw

    def get_humidity(self) -> float:
        return self.humidity

    def get_orientation(self) -> RollPitchYawDict:
        return self.get_orientation_degrees()

    def get_orientation_degrees(self) -> RollPitchYawDict:
        return self.orientation

    def get_orientation_radians(self) -> RollPitchYawDict:
        return self.orientation_radians

    def get_pixel(self, x: int, y: int) -> list[int]:
        return list()

    def get_pixels(self) -> list[list[int]]:
        return list()

    def get_pressure(self) -> float:
        return self.pressure

    def get_temperature(self) -> float:
        return self.get_temperature_from_humidity()

    def get_temperature_from_humidity(self) -> float:
        return float()

    def get_temperature_from_pressure(self) -> float:
        return float()

    @property
    def gyro(self) -> RollPitchYawDict:
        return self.gyroscope

    @property
    def gyro_raw(self) -> XYZDict:
        return self.gyroscope_raw

    @property
    def gyroscope(self) -> RollPitchYawDict:
        return DEFAULT_ROLL_PITCH_YAW_DICT

    @property
    def gyroscope_raw(self) -> XYZDict:
        return DEFAULT_X_Y_Z_DICT

    @property
    def humidity(self) -> float:
        return float()

    def has_colour_sensor(self) -> bool:
        return bool()

    def load_image(self, file_path: str, redraw: bool = True) -> list[list[int]]:
        return list()

    @property
    def low_light(self) -> bool:
        return bool()

    @low_light.setter
    def low_light(self, value: int) -> None:
        pass

    @property
    def orientation(self) -> RollPitchYawDict:
        return DEFAULT_ROLL_PITCH_YAW_DICT

    @property
    def orientation_radians(self) -> RollPitchYawDict:
        return DEFAULT_ROLL_PITCH_YAW_DICT

    @property
    def pressure(self) -> float:
        return float()

    @property
    def rotation(self) -> int:
        return int()

    @rotation.setter
    def rotation(self, r: int) -> None:
        pass

    def set_imu_config(
        self, compass_enabled: bool, gyro_enabled: bool, accel_enabled: bool
    ) -> None:
        pass

    def set_pixels(self, pixel_list: list[list[int]]) -> None:
        pass

    def set_pixel(self, x: int, y: int, *args) -> None:
        pass

    def set_rotation(self, r: int, redraw: bool = True) -> None:
        pass

    def show_letter(
        self,
        s: str,
        text_colour: list[int] = [255, 255, 255],
        back_colour: list[int] = [0, 0, 0],
    ) -> None:
        pass

    def show_message(
        self,
        text_string: str,
        scroll_speed: float = 0.1,
        text_colour: list[int] = [255, 255, 255],
        back_colour: list[int] = [0, 0, 0],
    ) -> None:
        pass

    @property
    def stick(self) -> SenseHatStickAPI:
        return SenseHatStickAPI()

    @property
    def temp(self) -> float:
        return self.temperature

    @property
    def temperature(self) -> float:
        return float()
