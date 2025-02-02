import abc
import ctypes
import logging
import platform
import subprocess
import threading
import time
from abc import abstractmethod, abstractproperty
from fractions import Fraction
from typing import BinaryIO, Iterable, Optional, Union

from colorzero import Color

from astro_pi_replay.custom_types import IO_TYPE, UV, XYWH
from astro_pi_replay.exception import AstroPiReplayException
from astro_pi_replay.picamera.exc import PiCameraRuntimeError, PiCameraValueError
from astro_pi_replay.picamera.frames import PiVideoFrame
from astro_pi_replay.picamera.mmalobj import PiFramerateRange, PiResolution
from astro_pi_replay.picamera.renderers import PiOverlayRenderer, PiRenderer

logger = logging.getLogger(__name__)


class PiCamera(abc.ABC):
    """
    This is for a V2 camera i.e. a Sony IMX219
    """

    CAMERA_PREVIEW_PORT: int = 0
    CAMERA_VIDEO_PORT: int = 1
    CAMERA_CAPTURE_PORT: int = 2
    MAX_RESOLUTION: PiResolution = PiResolution(width=4056, height=3040)
    MAX_FRAMERATE: int = 120
    DEFAULT_ANNOTATE_SIZE: int = 32
    CAPTURE_TIMEOUT: int = 60

    # TODO
    # SENSOR_MODES = {
    #     'ov5647': {
    #         1: mo.PiSensorMode('1080p', (1, 30), full_fov=False),
    #         2: mo.PiSensorMode('2592x1944', (1, 15), still=True),
    #         3: mo.PiSensorMode('2592x1944', (1/6, 1), still=True),
    #         4: mo.PiSensorMode('1296x972', (1, 42)),
    #         5: mo.PiSensorMode('1296x730', (1, 49)),
    #         6: mo.PiSensorMode('VGA', (42, 60)),
    #         7: mo.PiSensorMode('VGA', (60, 90)),
    #         },
    #     'imx219': {
    #         1: mo.PiSensorMode('1080p', (1/10, 30), full_fov=False),
    #         2: mo.PiSensorMode('3280x2464', (1/10, 15), still=True),
    #         3: mo.PiSensorMode('3280x2464', (1/10, 15), still=True),
    #         4: mo.PiSensorMode('1640x1232', (1/10, 40)),
    #         5: mo.PiSensorMode('1640x922', (1/10, 40)),
    #         6: mo.PiSensorMode('720p', (40, 90), full_fov=False),
    #         7: mo.PiSensorMode('VGA', (40, 90), full_fov=False),
    #         },
    #     }

    METER_MODES: dict[str, int] = {"average": 0, "spot": 1, "backlit": 2, "matrix": 3}

    EXPOSURE_MODES: dict[str, int] = {
        "off": 0,
        "auto": 1,
        "night": 2,
        "nightpreview": 3,
        "backlight": 4,
        "spotlight": 5,
        "sports": 6,
        "snow": 7,
        "beach": 8,
        "verylong": 9,
        "fixedfps": 10,
        "antishake": 11,
        "fireworks": 12,
    }

    FLASH_MODES: dict[str, int] = {
        "off": 0,
        "auto": 1,
        "on": 2,
        "redeye": 3,
        "fillin": 4,
        "torch": 5,
    }

    AWB_MODES: dict[str, int] = {
        "off": 0,
        "auto": 1,
        "sunlight": 2,
        "cloudy": 3,
        "shade": 4,
        "tungsten": 5,
        "fluorescent": 6,
        "incandescent": 7,
        "flash": 8,
        "horizon": 9,
    }

    IMAGE_EFFECTS: dict[str, int] = {
        "none": 0,
        "negative": 1,
        "solarize": 2,
        "sketch": 6,
        "denoise": 7,
        "emboss": 8,
        "oilpaint": 9,
        "hatch": 10,
        "gpen": 11,
        "pastel": 12,
        "watercolor": 13,
        "film": 14,
        "blur": 15,
        "saturation": 16,
        "colorswap": 17,
        "washedout": 18,
        "posterise": 19,
        "colorpoint": 20,
        "colorbalance": 21,
        "cartoon": 22,
        "deinterlace1": 23,
        "deinterlace2": 24,
    }

    DRC_STRENGTHS: dict[str, int] = {"off": 0, "low": 1, "medium": 2, "high": 3}

    RAW_FORMATS: set[str] = {"yuv", "rgb", "rgba", "bgr", "bgra"}
    STEREO_MODES: dict[str, int] = {"none": 0, "side-by-side": 1, "top-bottom": 2}
    CLOCK_MODES: dict[str, int] = {"reset": 2, "raw": 1}

    # ISP_BLOCKS = {
    #     'black-level':   1 << 2,
    #     'lens-shading':  1 << 3,
    #     'white-balance': 1 << 5,
    #     'bad-pixel':     1 << 7,
    #     'crosstalk':     1 << 9,
    #     'demosaic':      1 << 11,
    #     'gamma':         1 << 18,
    #     'sharpening':    1 << 22,
    #     }

    # COLORSPACES = {
    #     'auto':   mmal.MMAL_COLOR_SPACE_UNKNOWN,
    #     'jfif':   mmal.MMAL_COLOR_SPACE_JPEG_JFIF,
    #     'bt601':  mmal.MMAL_COLOR_SPACE_ITUR_BT601,
    #     'bt709':  mmal.MMAL_COLOR_SPACE_ITUR_BT709,
    #     }

    _DEFAULT_EXIF_TAGS: dict[str, str] = {
        "IFD0.Model": "RP_imx477",
        "IFD0.Make": "RaspberryPi",
    }

    def __init__(
        self,
        camera_num: int = 0,
        stereo_mode: str = "none",
        stereo_decimate: bool = False,
        resolution: Optional[Union[tuple[int, int], str]] = None,
        framerate: Optional[Fraction] = None,
        sensor_mode: int = 0,
        led_pin=None,
        clock_mode: str = "reset",
        framerate_range: Optional[PiFramerateRange] = None,
    ) -> None:
        self.analog_gain: Fraction = Fraction(8, 1)  # TODO sample
        self.annotate_background: Optional[Color] = None
        self.annotate_foreground: Color = Color("#ffffff")
        self.annotate_frame_num: bool = False
        self.annotate_text: str = ""
        self.annotate_text_size: int = 32

        self.awb_gains: tuple[Fraction, Fraction] = (
            Fraction(29, 16),
            Fraction(773, 256),
        )  # TODO sample
        self.awb_mode: str = "auto"
        self.brightness: int = 50
        self._camera_num: int = camera_num
        self.clock_mode: str = clock_mode
        self._closed: bool = False
        self.color_effects: Optional[UV] = None
        self.contrast: int = 0
        self.crop: XYWH = (0.0, 0.0, 1.0, 1.0)
        self.digital_gain: Fraction = Fraction(187, 128)  # TODO sample
        self.drc_strength: str = "off"
        self._encoders_lock: threading.Lock = threading.Lock()
        self.exif_tags: dict[str, str] = PiCamera._DEFAULT_EXIF_TAGS
        self.exposure_compensation: int = 0
        self.exposure_mode: str = "auto"
        self.exposure_speed: int = int()  # TODO sample
        self.flash_mode: str = "off"
        self.framerate: Fraction = (
            framerate if framerate is not None else Fraction(30, 1)
        )  # TODO sample
        self.framerate_delta: Fraction = Fraction(0, 1)  # TODO sample
        self.framerate_range: PiFramerateRange = (
            framerate_range
            if framerate_range is not None
            else PiFramerateRange(low=Fraction(30, 1), high=Fraction(30, 1))
        )  # TODO sample
        self.hflip: bool = False
        self.image_denoise: bool = True
        self.image_effect: str = "none"
        self.image_effect_params: Optional[tuple] = None
        self.iso: int = 0
        self.ISO: int = self.iso
        self._led: Optional[bool] = None
        self._led_pin: int = led_pin if led_pin is not None else 32
        self.meter_mode: str = "average"
        self.overlays: list[PiRenderer] = []
        self.preview_alpha: int = 255
        self.preview_fullscreen: bool = True
        self.preview_layer: int = 2
        self.preview_window: Optional[XYWH] = None
        self._previewing: bool = False
        self.raw_format: str = "yuv"
        self._recording: bool = False
        self._resolution: PiResolution = self._convert_to_PiResolution(resolution)
        self.revision: str = "imx477"
        self.rotation: int = 0
        self.saturation: int = 0
        self.sensor_mode: int = sensor_mode
        self.sharpness: int = 0
        self.shutter_speed: int = 0
        self._stereo_mode: str = stereo_mode
        self._stereo_decimate: bool = stereo_decimate
        self.still_stats: bool = False
        self.vflip: bool = False
        self.video_denoise: bool = True
        self.video_stabilization: bool = False
        self.zoom: XYWH = self.crop

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self):
        pass

    @abstractmethod
    def add_overlay(
        self,
        source: BinaryIO,
        size: Optional[tuple[int, int]] = None,
        format: Optional[str] = None,
        **options,
    ) -> PiOverlayRenderer:
        pass

    @abstractmethod
    def capture(
        self,
        output: IO_TYPE,
        format: Optional[str] = None,
        use_video_port: bool = False,
        resize: Optional[tuple[int, int]] = None,
        splitter_port: int = 0,
        bayer: bool = False,
        **options,
    ) -> None:
        pass

    @abstractmethod
    def capture_continuous(
        self,
        output: IO_TYPE,
        format: Optional[str] = None,
        use_video_port: bool = False,
        resize: Optional[tuple[int, int]] = None,
        splitter_port: int = 0,
        burst: bool = False,
        bayer: bool = False,
        **options,
    ) -> Iterable:
        pass

    @abstractmethod
    def capture_sequence(
        self,
        outputs: Iterable[IO_TYPE],
        format: str = "jpeg",
        use_video_port: bool = False,
        resize: Optional[tuple[int, int]] = None,
        splitter_port: int = 0,
        burst: bool = False,
        bayer: bool = False,
        **options,
    ) -> None:
        pass

    def close(self):
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @abstractproperty
    def frame(self) -> Optional[PiVideoFrame]:
        if not self.recording:
            raise PiCameraRuntimeError(
                "Cannot query frame information " + "when camera is not recording"
            )
        return None  # TODO

    @property
    def led(self) -> None:
        raise AttributeError("")  # TODO

    @led.setter
    def led(self, value: bool) -> None:
        self._led = value

    @property
    def previewing(self) -> bool:
        return self._previewing

    @abstractproperty
    def preview(self) -> Optional[PiRenderer]:
        pass

    @abstractmethod
    def record_sequence(
        self,
        outputs: Iterable[IO_TYPE],
        format: str = "h264",
        resize: Optional[tuple[int, int]] = None,
        splitter_port: int = 1,
        **option,
    ) -> Iterable[IO_TYPE]:
        pass

    @property
    def recording(self) -> bool:
        return self._recording

    @abstractmethod
    def remove_overlay(self, overlay: PiOverlayRenderer) -> None:
        pass

    def request_key_frame(self, splitter_port: int = 1) -> None:
        pass

    @property
    def resolution(self) -> PiResolution:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Optional[Union[tuple[int, int], str]]) -> None:
        self._resolution = self._convert_to_PiResolution(resolution)

    def _convert_to_PiResolution(
        self, resolution: Optional[Union[tuple[int, int], str]]
    ) -> PiResolution:
        if resolution is None:
            return PiResolution(width=1280, height=720)
        else:
            # TODO parse the strings properly...
            width = resolution[0]
            height = resolution[1]
            if not (
                (0 < int(width) <= PiCamera.MAX_RESOLUTION.width)
                and (0 < int(height) <= PiCamera.MAX_RESOLUTION.height)
            ):
                raise PiCameraValueError(f"Invalid resolution requested: {resolution}")

            return PiResolution(width=width, height=height)

    @abstractmethod
    def split_recording(
        self,
        output: IO_TYPE,
        splitter_port: int = 1,
        **options,
    ) -> Optional[PiVideoFrame]:
        pass

    @abstractmethod
    def start_preview(self, **options) -> PiRenderer:
        self._previewing = True
        return PiRenderer(self)  # TODO - should be preview one

    @abstractmethod
    def start_recording(
        self,
        # TODO create bufferable and writeable protocol types and a synonym
        # for this union
        output: IO_TYPE,
        format: Optional[str] = None,
        resize: Optional[tuple[int, int]] = None,
        splitter_port: int = 1,
        **options,
    ) -> None:
        self._recording = True

    @abstractmethod
    def stop_preview(self) -> None:
        self._previewing = False

    @abstractmethod
    def stop_recording(self, splitter_port: int = 1) -> None:
        self._recording = False

    @property
    def timestamp(self) -> int:
        system: str = platform.system()
        command_args: list[str]
        val: int
        if system in ["Linux", "Darwin"]:
            if system == "Linux":
                command_args = ["cat", "/proc/uptime"]
            else:
                command_args = ["sysctl", "kern.boottime"]

            command: str = " ".join(command_args)
            logger.debug(command)
            out = subprocess.run(
                command_args, check=True, capture_output=True, text=True
            )  # nosec B603

            if system == "Linux":
                val = round(float(out.stdout.strip().split()[0]))
            else:
                val = int(out.stdout.strip().split()[4].replace(",", ""))
        elif system == "Windows":
            val = int(ctypes.windll.kernel32.GetTickCount64())  # type: ignore
        else:
            raise AstroPiReplayException(f"Unsupported system {system}")
        return val

    def wait_recording(self, timeout: int = 0, splitter_port: int = 1) -> None:
        if timeout > 0:
            time.sleep(timeout)

    # private methods expected by picamera's internal implementation
    def _stop_capture(self, port):
        """
        Stops the camera capturing frames.

        This method stops the camera feeding frames to any attached encoders,
        but only disables capture if the port is the camera's still port, or if
        there's a single active encoder on the video splitter.
        """
        pass
