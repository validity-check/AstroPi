import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import requests
from requests.exceptions import RequestException

from astro_pi_replay import PROGRAM_CMD_NAME, PROGRAM_NAME, __version__
from astro_pi_replay.venv_resolver import VenvInfo, VenvResolver
from astro_pi_replay.version_utils import compare_semver

logger = logging.getLogger(__name__)
PYPI_URL: str = f"https://pypi.org/simple/{PROGRAM_NAME.replace('_','-')}"


class SelfUpdater:
    def _check_for_updates(self) -> list[str]:
        """
        Check PyPi for a new version
        """

        to_return: list[str] = []
        try:
            logger.debug(f"Checking {PYPI_URL} for a new version")
            response: requests.Response = requests.get(
                PYPI_URL,
                headers={"Accept": "application/vnd.pypi.simple.v1+json"},
                timeout=3,
            )
            if response.status_code != 200:
                logger.debug(f"Received status code {response.status_code}")
                return to_return
            json_data: dict = json.loads(response.content.decode("utf-8"))

            latest_available: str = json_data["versions"][-1]
            if compare_semver(latest_available, __version__) == 1:
                to_return.append(f"An update to {PROGRAM_CMD_NAME} is available")
                to_return.append(f"To update, run {PROGRAM_CMD_NAME} update")
            else:
                logger.debug(f"An update to {PROGRAM_NAME} is not required")
        except (
            RequestException,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
        ) as e:
            logger.debug(e)
        return to_return

    def check_for_updates(self) -> None:
        for line in self._check_for_updates():
            logger.info(line)

    def _update(self, venv_info: VenvInfo) -> None:
        subprocess.run(  # nosec B603
            [str(venv_info.pip), "install", "--upgrade", "astro_pi_replay"],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        logger.info("Update complete")

    def update(self, venv_dirname: Optional[Path] = None) -> None:
        """
        Update requested (via Astro-Pi-Replay update)
        """
        venv_path: Path = venv_dirname if venv_dirname is not None else Path(sys.prefix)
        venv_info: VenvInfo = VenvResolver.resolve_venv_dirs(venv_path, sys.platform)

        # move the replay dirs to a temporary location to avoid redownloading them
        temp_dir: Path = Path(mkdtemp())
        # replay_dir: Path = get_replay_dir()
        replay_dir: Path = (
            venv_info.site_packages_dir / PROGRAM_NAME / "resources" / "replay"
        )
        # TODO shouldn't replay dir depend on venv anyway?
        shutil.move(replay_dir, temp_dir)
        try:
            # now the replay dir is currently temp_dir / replay_dir.name
            self._update(venv_info)
        finally:
            source: Path = temp_dir / replay_dir.name
            target: Path = replay_dir
            if target.exists():
                # the replay directory is included in the manifest
                shutil.rmtree(target)
            logger.debug(f"Moving {source} into {target.parent}")
            shutil.move(
                source,
                target.parent,
            )
