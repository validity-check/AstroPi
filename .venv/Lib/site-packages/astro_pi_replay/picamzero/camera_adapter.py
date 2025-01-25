import dataclasses
import logging
import math
import os
import subprocess
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from time import sleep
from typing import Any, Optional, Union, cast

import cv2
import numpy as np
from PIL import Image

from astro_pi_replay.exception import (
    AstroPiReplayException,
    AstroPiReplayRuntimeError,
    DependencyNotInstalledException,
    FfmpegNotInstalledException,
)
from astro_pi_replay.executor import AstroPiExecutor
from astro_pi_replay.libcamera import controls
from astro_pi_replay.picamzero.ImageWrapper import ImageWrapper
from astro_pi_replay.picamzero.PicameraZeroException import PicameraZeroException
from astro_pi_replay.resources.downloader import get_replay_sequence_dir, get_video

from . import utilities as utils

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARN)

GPS_IFD_CODE: int = 0x8825


@dataclasses.dataclass
class Overlay:
    image: ImageWrapper
    position: tuple[int, int]
    transparency: float


def decorate_all_non_magic_methods(decorator):
    """
    Class decorator that decorates every non-magic method
    in the class with the given decorator.
    """

    def class_decorator(clz):
        for name, method in clz.__dict__.items():
            if callable(method) and not (name.startswith("__") and name.endswith("__")):
                setattr(clz, name, decorator(method))
        return clz

    return class_decorator


def run(cmd: list[str], **kwargs):
    cmd_string: str = " ".join(cmd)

    logger.debug(f"Executing {cmd_string}")
    proc = subprocess.run(cmd, text=True, capture_output=True, **kwargs)  # nosec B603
    if proc.returncode != 0:
        raise AstroPiReplayRuntimeError(
            os.linesep.join(["Encountered error:", proc.stderr])
        )


def CameraAdapter(
    maybe_executor: Optional[AstroPiExecutor] = None,
    # Different camera and processor combinations
    # support a different range of resolutions.
    # This is the minimum 'maximum' for all combinations
    MAX_VIDEO_SIZE: tuple[int, int] = (1920, 1080),
    HQC_SENSOR_RESOLUTION: tuple[int, int] = (4056, 3040),
    *args,
    **kwargs,
):
    executor: AstroPiExecutor
    if maybe_executor is None:
        executor = AstroPiExecutor()
    else:
        executor = maybe_executor

    executor._state._picamera_instances_count += 1
    if executor._state._picamera_instances_count > 1:
        raise PicameraZeroException(
            "Only one Camera instance is allowed.",
            "Ensure you are not trying to create multiple Camera objects.",
        )

    def swallow_dependency_not_installed_exceptions(func):
        """
        Decorator to swallow DependencyNotInstalledExceptions
        if the executor has been told it is running in a browser.

        This is indicated by setting executor.is_running_in_browser = True
        """

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DependencyNotInstalledException as e:
                if executor.is_running_in_browser:
                    e.log_warning()
                else:
                    raise e

        return wrapper

    @decorate_all_non_magic_methods(swallow_dependency_not_installed_exceptions)
    class _CameraAdapter:
        _SUPPORTED_VIDEO_FORMATS: list[str] = ["mp4"]
        _SUPPORTED_PHOTO_FORMATS: list[str] = ["jpg", "jpeg", "png"]

        # Taken from camera_controls from a Raspberry Pi 4
        # with HQC: print(pc2.camera_controls)
        CONTROLS = {
            "Sharpness": (0.0, 16.0, 1.0),
            "ExposureValue": (-8.0, 8.0, 0.0),
            "AeConstraintMode": (0, 3, 0),
            "ScalerCrop": (
                (0, 0, 128, 128),
                (0, 0) + HQC_SENSOR_RESOLUTION,
                (2, 0, 4052, 3040),
            ),
            "AnalogueGain": (1.0, 22.2608699798584, None),
            "NoiseReductionMode": (0, 4, 0),
            "AeMeteringMode": (0, 3, 0),
            "ExposureTime": (60, 674181621, None),
            "HdrMode": (0, 4, 0),
            "AwbEnable": (False, True, None),
            "Saturation": (0.0, 32.0, 1.0),
            "Contrast": (0.0, 32.0, 1.0),
            "ColourGains": (0.0, 32.0, None),
            "Brightness": (-1.0, 1.0, 0.0),
            "FrameDurationLimits": (24994, 674193371, None),
            "AeFlickerPeriod": (100, 1000000, None),
            "AwbMode": (0, 7, 0),
            "AeFlickerMode": (0, 1, 0),
            "AeExposureMode": (0, 3, 0),
            "StatsOutputEnable": (False, True, False),
            "AeEnable": (False, True, None),
        }

        def __init__(self) -> None:
            """
            Creates a Camera object based on a Picamera2 object

            :param Picamera2 pc2:
                An internal Picamera2 object. This can be accessed by
                advanced users who want to use methods we have not
                wrapped from the Picamera2 library.
            """
            self._recording: Optional[str] = None
            self._recording_start: Optional[datetime] = None

            # Camera
            self.hflip = False
            self.vflip = False

            # Annotation
            self._text: Optional[str] = None
            self._text_properties = {
                "font": utils.check_font_in_dict("plain1"),
                "color": (255, 255, 255, 255),
                "position": (0, 0),
                "scale": 3,
                "thickness": 3,
                "bgcolor": None,
                "position": (0, 0),
            }

            # self.pc2.start()
            self._preview_size: tuple[int, int] = HQC_SENSOR_RESOLUTION
            self._still_size: tuple[int, int] = HQC_SENSOR_RESOLUTION
            self._video_size: tuple[int, int] = MAX_VIDEO_SIZE
            self._brightness: float = 0.0
            self._contrast: float = 1.0
            self._exposure: Optional[int] = None
            self._gain: Optional[float] = None
            self._white_balance: Optional[controls.AwbModeEnum] = None
            self._greyscale: bool = False
            self._overlay: Optional[Overlay] = None

        def __del__(self):
            """
            Cleanup the Camera instance when it is deleted
            """
            executor._state._picamera_instances_count -= 1

        # PRIVATE METHODS
        # ----------------------------------

        def _create_video(
            self, final_filename: str, start: datetime, duration: float
        ) -> None:
            # calculate the time since the replay started
            delta: timedelta = start - executor._state.get_start_time()
            video: Path = get_video()
            if not video.exists():
                executor._get_downloader().fetch_sequence_file(video)

            if not executor._has_ffmpeg:
                raise FfmpegNotInstalledException(
                    action="Creating videos", desired_result="create a video"
                )

            cmd: list[str] = [
                "ffmpeg",
                "-ss",
                str(delta.total_seconds()),
                "-to",
                str(delta.total_seconds() + duration),
                # input
                "-i",
                str(video),
                "-c",
                "copy",
                str(final_filename),
            ]
            logger.debug(" ".join(cmd))
            run(cmd)

        def _detect_format(
            self,
            filename: Optional[str],
            allowed_formats: list[str],
            default_format: str,
        ) -> str:
            if filename is None:
                raise RuntimeError("Please specify a filename")

            split_filename = filename.split(".")
            last = split_filename[-1]
            if len(split_filename) == 1:
                final_filename = f"{last}.{default_format}"
            elif last == default_format:
                final_filename = filename
            else:
                raise RuntimeError(
                    f"Unsupported format: {last}. "
                    + "Supported formats are: "
                    + f"'{', '.join(allowed_formats)}'"
                )

            return final_filename

        def log_warning(self, action: str):
            if not executor.configuration.is_transparent_to_user:
                logger.warning(
                    f"{action.capitalize()} has no effect when running "
                    + "using the replay tool, since the data has been collected "
                    + "already and is just being replayed. It will have the desired "
                    + "effect when run on the ISS"
                )

        # ----------------------------------
        # PROPERTIES
        # ----------------------------------

        # Check that the value given for a control is allowed
        def _check_control_in_range(self, name: str, value: Any) -> bool:
            try:
                minvalue, maxvalue, _ = self.CONTROLS[name]
            except Exception as e:
                raise PicameraZeroException(
                    f"The control {e} doesn't exist", "Check for spelling errors?"
                )

            if value > maxvalue or value < minvalue:
                raise PicameraZeroException(
                    f"Invalid {name.lower()} value",
                    f"{name} must be between {minvalue} and {maxvalue}",
                )
            return True

        @property
        def pc2(self):
            raise AstroPiReplayException(
                "Direct access of pc2 is not supported in Astro-Pi-Replay"
            )

        @property
        def preview_size(self) -> tuple[int, int]:
            return self._preview_size

        @preview_size.setter
        def preview_size(self, size: tuple[int, int]):
            size = utils.check_camera_size(
                HQC_SENSOR_RESOLUTION,
                size,
                error_msg_type="preview",
            )
            self.log_warning("Setting preview_size")
            self._preview_size = size

        @property
        def still_size(self) -> tuple[int, int]:
            return self._still_size

        @still_size.setter
        def still_size(self, size: tuple[int, int]):
            size = utils.check_camera_size(
                HQC_SENSOR_RESOLUTION,
                size,
                error_msg_type="image",
            )
            self.log_warning("Setting still_size")
            self._still_size = size

        @property
        def video_size(self) -> tuple[int, int]:
            return self._video_size

        @video_size.setter
        def video_size(self, size: tuple[int, int]):
            size = utils.check_camera_size(
                HQC_SENSOR_RESOLUTION,
                size,
                error_msg_type="video",
            )
            self.log_warning("Setting video_size")
            self._video_size = size

        # Brightness
        @property
        def brightness(self) -> float:
            """
            Get the brightness

            :return float:
            Brightness value between -1.0 and 1.0
            """
            return self._brightness

        @brightness.setter
        def brightness(self, bvalue: float):
            """
            Set the brightness

            :param float bvalue:
                Floating point number between -1.0 and 1.0
            """
            if self._check_control_in_range("Brightness", bvalue):
                self.log_warning("Setting brightness")
                self._brightness = bvalue

        # Contrast
        @property
        def contrast(self) -> float:
            """
            Get the contrast

            :return float:
                Contrast value between 0.0 and 32.0
            """
            return self._contrast

        @contrast.setter
        def contrast(self, cvalue: float):
            """
            Set the contrast

            :param float cvalue:
                Floating point number between 0.0 and 32.0
                Normal value is 1.0
            """
            if self._check_control_in_range("Contrast", cvalue):
                self.log_warning("Setting contrast")
                self._contrast = cvalue

        @property
        def exposure(self) -> Optional[int]:
            """
            Get the exposure

            :returns int:
                Exposure value (max and min depend on mode)
            """
            return self._exposure

        @exposure.setter
        def exposure(self, etime: int):
            """
            Set the exposure

            :param int etime:
                The exposure time (max and min depend on mode)
            """
            if self._check_control_in_range("ExposureTime", etime):
                self.log_warning("Setting exposure")
                self._exposure = etime

        @property
        def gain(self) -> Optional[float]:
            """
            Get the gain

            :returns float:
                Gain value (max and min depend on mode)
            """
            return self._gain

        @gain.setter
        def gain(self, gvalue: float):
            """
            Set the analogue gain

            :param float gvalue:
                The analogue gain (max and min depend on mode)
            """
            if self._check_control_in_range("AnalogueGain", gvalue):
                self.log_warning("Setting gain")
                self._gain = gvalue

        @property
        def white_balance(self) -> Optional[str]:
            """
            Get the white balance mode

            :return str:
                The selected white balance mode as a string
            """

            if self._white_balance is None:
                return None
            rev_possible_controls = cast(
                dict[controls.AwbModeEnum, str],
                utils.possible_controls(reverse_kv=True),
            )
            return rev_possible_controls[self._white_balance]

        @white_balance.setter
        def white_balance(self, wbmode: str):
            """
            Set the white balance mode

            :param str wbmode:
                A white balance mode from the allowed list
                (at present, Custom is not allowed)
            """
            possible_controls = cast(
                dict[str, controls.AwbModeEnum], utils.possible_controls()
            )
            if wbmode.lower() not in utils.possible_controls():
                if wbmode.lower() == "custom":
                    raise PicameraZeroException(
                        "Custom white balance is not supported yet",
                        "White balance can be " + ", ".join(possible_controls.keys()),
                    )
                else:
                    raise PicameraZeroException(
                        "Invalid white balance mode",
                        "White balance can be " + ", ".join(possible_controls.keys()),
                    )
            self.log_warning("Setting white_balance")
            self._white_balance = possible_controls[wbmode.lower()]

        @property
        def greyscale(self) -> bool:
            return self._greyscale

        @greyscale.setter
        def greyscale(self, on: bool) -> None:
            """
            Apply greyscale to the preview and image
            You have to call this _after_ the preview has started or it wont apply
            Does NOT apply to video

            :param bool on:
                Whether greyscale should be on
            """
            self.log_warning("Setting greyscale")
            self._greyscale = on

        # ----------------------------------
        # METHODS
        # ----------------------------------

        def flip_camera(self, vflip=False, hflip=False):
            """
            Flip the image horizontally or vertically
            """
            self.log_warning("Setting flip_camera")
            self.vflip = vflip
            self.hflip = hflip

        def start_preview(self):
            """
            Show a preview of the camera
            """
            self.log_warning("Starting preview")

        def stop_preview(self):
            """
            Stop the preview
            """
            pass

        def annotate(
            self,
            text="Default Text",
            font="plain1",
            color=(255, 255, 255, 255),
            scale=3,
            thickness=3,
            position=(0, 0),
            bgcolor=None,
        ):
            """
            Set a text overlay on the preview and on images
            """
            self._text = text

            font = utils.check_font_in_dict(font)
            color = utils.convert_color(color)

            self._text_properties = {
                "font": font,
                "color": color,
                "scale": scale,
                "thickness": thickness,
                "bgcolor": bgcolor,
                "position": position,
            }

        def add_image_overlay(
            self,
            image_path: Union[str, Path],
            position: tuple[int, int] = (0, 0),
            transparency: float = 0.5,
        ):
            overlay_img, position, transparency = utils.check_image_overlay(
                image_path, position, transparency
            )

            if not overlay_img.mode == "RGBA":
                overlay_img.convert("RGBA")

                # Modify the alpha channel to match the transparency
                overlay_array: np.ndarray = overlay_img.toarray()
                overlay_array[:, :, 3] = np.full(
                    overlay_array[:, :, 3].shape, round(transparency * 255)
                )
                overlay_img = overlay_img.fromarray(overlay_array)

            self._overlay = Overlay(
                image=overlay_img, position=position, transparency=transparency
            )

        def take_video_and_still(
            self,
            filename: Optional[str] = None,
            duration: int = 20,
            still_interval: int = 4,
        ):
            """
            Take video for <duration> and take a still every <interval> seconds?
            """
            if filename is None:
                raise RuntimeError("Must provide filename")
            split_filename = filename.split(".")
            if len(split_filename) > 1:
                raise RuntimeError("Can only specify basename")

            self.start_recording(filename)
            max_i: int = round(duration / still_interval)
            i: int = 1
            while (
                self._recording_start is not None
                and i <= max_i
                and (datetime.now() - self._recording_start).total_seconds() < duration
            ):
                before_photo: datetime = datetime.now()
                self.take_photo(f"{filename}-{i}.jpg")
                i += 1
                after_photo: datetime = datetime.now()
                still_duration = (after_photo - before_photo).total_seconds()
                if still_duration > still_interval:
                    logger.warning(
                        "Image capture took longer than "
                        + "still interval specified - this happens "
                        + "the replay tool replays old data "
                        + "which cannot be changed."
                    )
                video_duration = (after_photo - self._recording_start).total_seconds()
                seconds_until_video_finished = duration - video_duration
                if seconds_until_video_finished < still_interval:
                    sleep(still_interval - seconds_until_video_finished)
                elif still_interval > still_duration:
                    sleep(still_interval - still_duration)

            self.stop_recording()

        def _get_next_photo(self) -> ImageWrapper:
            name: str = str(
                executor._replay_next(
                    str(get_replay_sequence_dir() / "photos" / "photo_index.csv"),
                    "datetime",
                    ["name"],
                    allow_interpolation=False,
                )
            )

            image_path: Path = get_replay_sequence_dir() / "photos" / name
            if not image_path.exists():
                executor._get_downloader().fetch_sequence_file(image_path)
            im: ImageWrapper = ImageWrapper(image_path)

            if self._overlay:
                w, h = im.size
                overlay_img = self._overlay.image
                pos_w, pos_h = self._overlay.position
                if pos_w > w or pos_h > h:
                    raise PicameraZeroException(
                        "Overlay position" + "is bigger than the image size",
                        hint="Reduce the overlay position to less "
                        + f"than ({w}, {h})",
                    )
                remaining_w, remaining_h = (w - pos_w, h - pos_h)
                overlay_w, overlay_h = overlay_img.size
                if overlay_w > remaining_w or overlay_h > remaining_h:
                    overlay_img.resize((remaining_w, remaining_h))
                im.paste(overlay_img, (pos_w, pos_h), mask=overlay_img)

            if self._text:
                text_prop: dict = self._text_properties
                # Create the background
                x, y = text_prop["position"]
                text_size, _ = cv2.getTextSize(
                    self._text,
                    text_prop["font"],
                    text_prop["scale"],
                    text_prop["thickness"],
                )
                text_w, text_h = text_size

                im_array: np.ndarray = im.toarray()
                if text_prop["bgcolor"] is not None:
                    cv2.rectangle(
                        im_array,
                        text_prop["position"],
                        (x + text_w, y + text_h),
                        text_prop["bgcolor"],
                        -1,
                    )
                cv2.putText(
                    im_array,
                    self._text,
                    (x, y + text_h + text_prop["scale"] - 4),
                    text_prop["font"],
                    text_prop["scale"],
                    text_prop["color"],
                    text_prop["thickness"],
                )
                im = im.fromarray(im_array)
            return im

        def capture_array(self) -> np.ndarray:
            """
            Takes a photo at full resolution and saves it as an
            (RGB) numpy array.

            This can be used in further processing using libraries
            like opencv.

            :return np.ndarray:
                A full resolution image as a raw RGB numpy array
            """
            im = self._get_next_photo()
            return im.toarray()

        def take_photo(self, filename=None, gps_coordinates=None) -> str:
            """
            Takes a jpeg image using the camera
            :param str filename: The name of the file to save the photo.
            If it doesn't end with '.jpg', the ending '.jpg' is added.
            :param tuple[tuple[float, float, float, float],
                         tuple[float, float, float, float]] gps_coordinate:
            The gps coordinates to be associated
            with the image, specified as a (latitude, longitude) tuple where
            both latitude and longitude are themselves tuples of the
            form (sign, degrees, minutes, seconds). This format
            can be generated from the skyfield library's signed_dms
            function.
            """
            final_filename: str = utils.format_filename(filename, ".jpg")

            im: ImageWrapper = self._get_next_photo()
            exif: Image.Exif = im.getexif()
            if gps_coordinates:
                gps = utils.signed_dms_coordinates_to_exif_dict(gps_coordinates)
                # Convert to Fractional, like Pillow needs
                for k, dms in gps["GPS"].items():
                    if isinstance(dms, tuple):
                        gps["GPS"][k] = tuple([Fraction(v[0], v[1]) for v in dms])
                exif[GPS_IFD_CODE] = gps["GPS"]
            im.save(final_filename, exif=exif)

            return final_filename

        capture_image = take_photo

        def capture_sequence(
            self,
            filename: Optional[Union[str, Path]] = None,
            num_images: int = 10,
            interval: float = 1,
            make_video: bool = False,
        ):
            """
            Take a series of <num_images> and save them as
            <filename> with auto-number, also set the interval between
            """
            # Format the filename using appropriate zero-padded sequence
            padding_amount: str = str(math.ceil(math.log10(num_images + 1)))
            ext: str = "-{:0" + padding_amount + "d}.jpg"
            final_filename: str = utils.format_filename(filename, ext=ext)

            start_time: datetime = datetime.now()
            for i in range(num_images):
                name = final_filename.format(i + 1)
                # the take_photo method sleeps until the next
                # frame is available
                self.take_photo(name)
                time_after = datetime.now()
                delta = (time_after - start_time).total_seconds()
                if delta > interval:
                    logger.warning(
                        "Slept longer than interval because "
                        + "the interval between photos from the "
                        + "replayed dataset is longer. On the ISS, "
                        + "the actual interval may be closer to "
                        + "the value you have specified."
                    )
                else:
                    remainder = interval - delta
                    logger.debug("Sleeping an additional " + f"{remainder} seconds")
                    sleep(remainder)

            if make_video:
                if not executor._has_ffmpeg:
                    raise FfmpegNotInstalledException(
                        action="Making videos", desired_result="make a video"
                    )
                video_name = utils.format_filename(filename, ext="-timelapse.mp4")

                cmd: list[str] = [
                    "ffmpeg",
                    "-framerate",
                    "1",
                    "-i",
                    utils.format_filename(filename, ext=f"-%{padding_amount}d.jpg"),
                    "-r",
                    "30",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    video_name,
                ]
                cmd_as_string: str = " ".join(cmd)
                logger.debug(f"Running {cmd_as_string}")
                run(cmd)

        # Synonym method for capture_sequence
        take_sequence = capture_sequence

        def record_video(
            self, filename: Optional[Union[str, Path]] = None, duration: int = 5
        ):
            """
            Record a video
            """

            start_time: datetime = datetime.now()
            final_filename: str = utils.format_filename(filename, ".mp4")

            if not executor._has_ffmpeg:
                raise FfmpegNotInstalledException(
                    action="Recording a video", desired_result="record a video"
                )

            video: Path = get_video()
            if not video.exists():
                executor._get_downloader().fetch_sequence_file(video)

            # calculate the time since the replay started
            delta: timedelta = datetime.now() - executor._state.get_start_time()
            cmd: list[str] = [
                "ffmpeg",
                "-ss",
                str(delta.total_seconds()),
                "-to",
                str(delta.total_seconds() + duration),
                # input
                "-i",
                str(video),
                "-c",
                "copy",
                str(final_filename),
            ]
            logger.debug(" ".join(cmd))
            run(cmd)
            elapsed: float = (datetime.now() - start_time).total_seconds()
            remainder: float = duration - elapsed
            if remainder > 0:
                sleep(remainder)

        # Synonym method for record_video
        take_video = record_video

        def start_recording(
            self, filename: Optional[Union[str, Path]] = None, preview: bool = False
        ) -> None:
            """
            Record a video of undefined length
            """
            if self._recording or self._recording_start:
                logger.warning("You have already started a recording!")
                logger.warning("Skipping this request")
                return
            self._recording = utils.format_filename(filename, ".mp4")
            # Log the time the recording was started
            self._recording_start = datetime.now()

        def stop_recording(self) -> None:
            """
            Stop recording video
            """
            if not self._recording or not self._recording_start:
                raise RuntimeError("Recording not started")

            duration: float = (datetime.now() - self._recording_start).total_seconds()
            self._create_video(
                final_filename=self._recording,
                start=self._recording_start,
                duration=duration,
            )
            self._recording = None
            self._recording_start = None

    return _CameraAdapter(*args, **kwargs)
