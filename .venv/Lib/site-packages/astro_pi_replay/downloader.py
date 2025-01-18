import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional, TypeVar

import pandas as pd
import requests
from tqdm import tqdm  # todo only if not in emscripten

from astro_pi_replay import PROGRAM_NAME, __version__
from astro_pi_replay.exception import AstroPiReplayException
from astro_pi_replay.resources import (
    REPLAY_SEQUENCE_ENV_VAR,
    RESOURCE_DIR,
    get_replay_dir,
)

logger = logging.getLogger(__name__)

GPG_EMAIL = "enquiries@astro-pi.org"
BUCKET_NAME: str = "static.raspberrypi.org"
BUCKET_URL: str = os.environ.get(
    f"__{PROGRAM_NAME.upper()}_BUCKET_URL",
    f"https://{BUCKET_NAME}")
URL_BASE: str = f"{BUCKET_URL}/files/astro-pi"
GPG_KEY_URL = f"{URL_BASE}/astro-pi.gpg"  # TODO add key-rotation
url_prefix: str = f"{URL_BASE}/{PROGRAM_NAME}"
asset_url: str = f"{url_prefix}/assets"
asset_prefix: str = str(Path(asset_url).relative_to(Path(BUCKET_URL)))
version_url_prefix: str = f"{url_prefix}/{__version__}"
SEQUENCES_FILENAME: str = "sequences.csv"
SEQUENCES_FILE: Path = RESOURCE_DIR / SEQUENCES_FILENAME

T = TypeVar("T")

ONE_HOUR: int = 60 * 60


class Downloader:
    TEST_ASSETS = "test_data"

    def __init__(self) -> None:
        tempdir: Path = Path(tempfile.gettempdir())
        tempdir /= str(uuid.uuid4())
        tempdir.mkdir()
        self.tempdir = tempdir
        self.checked_for_sequences_override: bool = False

    def _check_sha256(self, sha256_file: Path) -> Optional[bool]:
        with sha256_file.open("r") as f:
            lines: list[str] = f.read().strip().split("\n")
        for line in lines:
            split_line: list[str] = re.split(r"\s+", line)
            if len(split_line) != 2:
                raise ValueError(f"File {sha256_file.name} has an invalid format")
            expected_sha256, filename = split_line[0], Path(split_line[1])
            with (sha256_file.parent / filename).open("rb") as f:
                actual_sha256 = hashlib.sha256(f.read()).hexdigest()
            if expected_sha256 != actual_sha256:
                return False
        return True

    def _check_gpg_signature(self, gpg_sig_file: Path) -> Optional[bool]:
        logger.debug("Checking if gpg is installed...")
        gpg_path = shutil.which("gpg")

        if gpg_path is not None and Path(gpg_path).exists():
            logger.debug("Checking if the astro pi GPG key has been imported...")
            command_args = ["gpg", "--list-public-keys", f"<{GPG_EMAIL}>"]
            logger.debug(" ".join(command_args))
            proc = subprocess.run(  # nosec B603
                command_args,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                logger.info(
                    f"GPG public key for {GPG_EMAIL} not found. "
                    + "Skipping integrity check"
                )
                commands: str = os.linesep.join(
                    [
                        f"wget {GPG_KEY_URL}",
                        f"gpg --import {GPG_KEY_URL.split('/')[-1]}",
                    ]
                )
                "\n"
                logger.info(f"You may import a key with: {commands}")
                return None

            logger.debug("Verifying the integrity of the download")
            command_args = [
                "gpg",
                "--verify",
                str(gpg_sig_file),
                str(gpg_sig_file).replace(gpg_sig_file.suffix, ""),
            ]
            logger.debug(" ".join(command_args))
            proc = subprocess.run(  # nosec B603
                command_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return True if proc.returncode == 0 else False
        return None

    def _unzip(self, zip_file: Path) -> Path:
        with zipfile.ZipFile(str(zip_file)) as z:
            for member in tqdm(z.infolist(), unit="iB"):
                logger.debug(member)
                try:
                    z.extract(member, self.tempdir)
                except zipfile.error as e:
                    logger.error(e)

        os.remove(zip_file)
        return self.tempdir

    async def download_file(self, url: str, destination_dir: Path) -> Path:
        local_filename: str = url.split("/")[-1]
        destination: Path = destination_dir / local_filename
        if sys.platform == "emscripten":
            res = await self.get_(url)
            view = await res.memoryview()
            with (destination_dir / local_filename).open("wb") as f:
                f.write(view)
        else:
            with requests.get(url, stream=True, timeout=ONE_HOUR) as r:
                r.raise_for_status()
                total_length = int(r.headers.get("content-length", 0))
                chunk_size = 5 * 1024
                prog_bar = tqdm(total=total_length, unit="iB", unit_scale=True)
                with (self.tempdir / local_filename).open("wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        prog_bar.update(len(chunk))
                        f.write(chunk)
                if self.tempdir != destination_dir:
                    shutil.copy2(self.tempdir / local_filename, destination)

        logger.debug(f"Download to {destination_dir / local_filename}")
        return destination

    async def download(self, asset_name: str) -> None:
        downloaded: list[Path] = []
        asset_name += ".zip"
        for file in [f"{asset_name}.sha256", f"{asset_name}.sig", f"{asset_name}"]:
            logger.info(f"Downloading {file}...")
            url = f"{asset_url}/{file}"
            downloaded.append(await self.download_file(url, self.tempdir))

        logger.debug(f"Tempdir {self.tempdir} contains: {os.listdir(self.tempdir)}")
        logger.info("Checking the integrity of the downloaded data...")
        if not self._check_sha256(downloaded[0]):
            raise AstroPiReplayException(
                "Downloaded file failed integrity check. Try again."
            )
        result: Optional[bool] = self._check_gpg_signature(downloaded[1])

        if result is not None and result is False:
            raise AstroPiReplayException("Downloaded file failed security check.")
        else:
            os.remove(downloaded[0])
            os.remove(downloaded[1])

    def has_downloaded(self, asset_name: str) -> bool:
        return f"{asset_name}" in os.listdir(self.tempdir)

    def has_installed(
        self,
        resolution: tuple[int, int],
        photography_type: str,
        sequence_name: Optional[str] = None,
    ) -> bool:
        sequence: str
        if sequence_name is None:
            sequence = os.environ.get(
                REPLAY_SEQUENCE_ENV_VAR,
                self.search_for_sequence(resolution, photography_type),
            )
        else:
            sequence = sequence_name

        try:
            return (get_replay_dir() / f"{photography_type}/{sequence}").exists()
        except FileNotFoundError:
            return False

    async def get_(self, url: str, timeout: int = 0):
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch

            # this will throw an `OSError: Failed to fetch`
            # when not exists.
            # TODO support timeout...

            res = await pyfetch(url)

            # TODO create Proxy object that support the same requests API...
            """
            ['__class__', '__delattr__', '__dict__', '__dir__',
             '__doc__', '__eq__', '__format__', '__ge__',
             '__getattribute__', '__getstate__', '__gt__', '__hash__',
             '__init__', '__init_subclass__', '__le__', '__lt__',
             '__module__', '__ne__', '__new__', '__reduce__',
             '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',
             '__str__', '__subclasshook__', '__weakref__', '_create_file',
             '_into_file', '_raise_if_failed', '_url', 'body_used',
             'buffer', 'bytes', 'clone', 'headers', 'js_response',
             'json', 'memoryview', 'ok', 'raise_for_status',
             'redirected', 'status', 'status_text', 'string',
             'text', 'type', 'unpack_archive', 'url']
            """
            res.status_code = res.status
            # res.content
            return res
        else:
            res = requests.get(url, timeout=timeout)

        return res

    async def check_for_sequences_override(self):
        """
        Consults the S3 bucket to see if there
        have been any dynamic overrides to sequences.csv
        file, updating the sequences.csv file if so.
        """

        try:
            res = await self.get_(
                f"{version_url_prefix}/{SEQUENCES_FILENAME}", timeout=5
            )
            if res.status_code == 200:
                # override the file in resources
                with open(SEQUENCES_FILE, "w") as f:
                    f.write(res.content.decode("utf-8"))
        finally:
            self.checked_for_sequences_override = True

    def search_for_sequence(
        self, resolution: tuple[int, int], photography_type: str
    ) -> str:
        """
        Returns the id of the most appropriate photo sequence given the requested
        resolution and photography type.
        """
        sequence: str

        # save it if not already open
        df = pd.read_csv(SEQUENCES_FILE)

        filtered = df[
            (df["photography_type"] == photography_type)
            & (df["resolution"] == "x".join((str(res) for res in resolution)))
        ]

        if len(filtered) > 0:
            sequence = filtered["sequence_id"][0]
        else:
            raise AstroPiReplayException(
                f"No photos with resolution {resolution} "
                + f"and photography type {photography_type} are available"
            )
        return sequence

    async def install(
        self,
        resolution: tuple[int, int],
        photography_type: str,
        sequence: Optional[str],
        test_assets_only: bool = False,
        with_video: bool = False,
    ) -> None:
        if not self.checked_for_sequences_override:
            await self.check_for_sequences_override()

        sequence_id: str
        if test_assets_only:
            sequence_id = "test_data"
        elif sequence is not None:
            sequence_id = sequence
        else:
            sequence_id = self.search_for_sequence(resolution, photography_type)

        sequences_to_install: list[str] = [sequence_id]
        if with_video:
            sequences_to_install.append(sequence_id + "_videos")

        for seq in sequences_to_install:
            logger.debug(f"Request to install {seq}")

            if self.has_installed(resolution, photography_type, seq):
                logger.debug(f"{seq} already installed")
                continue
            if not self.has_downloaded(seq):
                await self.download(seq)

            downloaded_file: Path = self.tempdir / (seq + ".zip")
            unzipped_dir: Path = self._unzip(downloaded_file)
            destination_dir: Path = get_replay_dir() / photography_type

            if unzipped_dir != destination_dir:
                shutil.copytree(unzipped_dir, destination_dir, dirs_exist_ok=True)
