import logging
import os

from astro_pi_replay import ONLINE_NAME

logger = logging.getLogger(__name__)


class AstroPiReplayRuntimeError(RuntimeError):
    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)


class AstroPiReplayException(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)

    def __repr__(self) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message


class DependencyNotInstalledException(AstroPiReplayException):
    def __init__(
        self,
        dependency_name: str,
        rp_os_package_name: str,
        download_help_url: str,
        action: str,
        desired_result: str,
    ) -> None:
        self.dependency_name: str = dependency_name
        self.rp_os_package_name: str = rp_os_package_name
        self.download_help_url: str = download_help_url
        self.action: str = action
        self.desired_result: str = desired_result
        self.message: str = self.format_message()
        super().__init__(self.message)

    def log_warning(self) -> None:
        logger.warning(
            f"{self.action.capitalize()} is not currently "
            + f"supported by {ONLINE_NAME}, sorry."
        )
        logger.warning(
            f"On an Astro Pi this would {self.desired_result} as " + "you requested."
        )

    def format_message(self) -> str:
        return os.linesep.join(
            [
                f"Please install {self.dependency_name} to {self.desired_result}",
                "On Raspberry Pi OS, this can be done using " + "the command below:",
                "",
                f"  sudo apt-get install {self.rp_os_package_name}",
                "",
                "For download instructions on other operating systems, "
                + f"check the {self.dependency_name} website:",
                "",
                f"  {self.download_help_url}",
            ]
        )


class FfmpegNotInstalledException(DependencyNotInstalledException):
    def __init__(self, action: str, desired_result: str) -> None:
        self.action: str = action
        self.desired_result: str = desired_result
        super().__init__(
            dependency_name="ffmpeg",
            rp_os_package_name="ffmpeg",
            download_help_url="https://www.ffmpeg.org/download.html",
            action=action,
            desired_result=desired_result,
        )
