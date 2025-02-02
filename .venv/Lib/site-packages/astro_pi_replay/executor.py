import collections
import functools
import importlib.util
import logging
import os
import platform
import shutil
import site
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from functools import partial, wraps
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
import scipy as sp
from scipy.interpolate._interpolate import interp1d as Interpolator

from astro_pi_replay import LOGGING_FORMAT, PACKAGE_ROOT, PROGRAM_CMD_NAME, PROGRAM_NAME
from astro_pi_replay.configuration import Configuration, get_default_venv_dir
from astro_pi_replay.custom_types import ExecutionMode
from astro_pi_replay.exception import AstroPiReplayException
from astro_pi_replay.resources import (
    SENSE_HAT_CSV_FILE,
    get_replay_sequence_dir,
    get_start_time,
)
from astro_pi_replay.resources.downloader import Downloader
from astro_pi_replay.resources.timed_downloader import TimedDownloader
from astro_pi_replay.venv_resolver import VenvResolver

logger = logging.getLogger(__name__)


class Lifecycle(Enum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"


class AstroPiExecutorState:
    """
    Wrapper class for the executor instance's shared, mutable state.
    """

    def __init__(self) -> None:
        self._last_sense_hat_row_index: int = 0
        self._last_picamera_photo_index: int = 0
        self._start_time: datetime = datetime.now()
        self._sense_hat_snapshot_index: int = 1
        self._picamera_instances_count: int = 0
        # the time spent waiting for the network
        self._network_time: float = 0

    def get_start_time(self) -> datetime:
        return timedelta(seconds=self._network_time) + self._start_time


class AstroPiExecutor:
    """
    Class containing the replaying logic (as instance methods)
    + the CLI main methods (as static methods).

    This class is instantiated (by the API adapter classes) only
    when ExecutionMode is REPLAY, in order control the replaying
    of data. Otherwise, its static methods are used to setup a
    venv and run main.py files.

    The class is a singleton
    """

    MODULES_TO_STUB: list[str] = [
        "sense_hat",
        "picamera",
        "orbit",
        "picamzero",
        "astro_pi_orbit",
    ]
    NOT_FOUND = f"{PROGRAM_CMD_NAME} not found"

    """
    Checks whether the current interpreter is running in a venv,
    as defined here in https://docs.python.org/3/library/venv.html#how-venvs-work
    """
    _instance: Optional["AstroPiExecutor"] = None  # singleton instance
    _callbacks: dict[Lifecycle, list[Callable]] = collections.defaultdict(list)

    def __new__(
        cls,
        datetime_col: str = "datetime",
        datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
        replay_mode: bool = True,
        state: Optional[AstroPiExecutorState] = None,
        configuration: Optional[Configuration] = None,
        downloader: Optional[Downloader] = None,
    ) -> "AstroPiExecutor":
        """
        datetime_format example: 2022-01-31 12:21:15.123456
        """
        if cls._instance is None:
            logger.debug("Creating new instance")
            cls._instance = super(AstroPiExecutor, cls).__new__(cls)

            cls.datetime_col: str = datetime_col
            cls.datetime_format: str = datetime_format
            cls.replay_mode: bool = replay_mode
            if state is None:
                state = AstroPiExecutorState()
            cls.interpolators: dict[str, Interpolator] = {}
            cls._state: AstroPiExecutorState = state

            if configuration is not None:
                logger.debug(f"Received configuration: {str(configuration)}")
            else:
                logger.debug("Configuration not passed directly - loading")
            cls.configuration = (
                configuration if configuration is not None else Configuration.load()
            )
            # Set in (astro-pi-replay-online) to alter some error messages
            cls.is_running_in_browser: bool = False
            cls.downloader = downloader
        else:
            logger.debug("Executor already instantiated")

        return cls._instance

    def sense_hat_replay(self, *args, **kwargs) -> Callable:
        """
        Decorator used to conditionally replay data from file for the SenseHat.
        """
        filename: str = str(get_replay_sequence_dir() / SENSE_HAT_CSV_FILE)

        if "filename" not in kwargs:
            kwargs["filename"] = filename
        return self.replay(*args, **kwargs)

    def replay(
        self,
        reducer: Callable[[pd.DataFrame], object] = lambda df: df.iloc[0],
        filename: Optional[str] = None,
        col_names: Optional[list[str]] = None,
        *args,
        **kwargs,
    ) -> Callable:
        """
        Decorator used to replay data from files, conditionally.
        """

        # TODO check args and kwargs for unexpected inputs (it should
        # only be the func to be decorated)

        # This is the actual decorator
        def decorator(func: Callable):
            # Activity here is processed at load-time
            logger.debug(f"Decorating function '{func.__name__}'")

            # This defines the functionality the decorator should do
            @wraps(func)
            def _replay(*_args, **_kwargs):
                if self.replay_mode:
                    nonlocal filename, col_names, reducer
                    if filename is None:
                        raise AstroPiReplayException("Cannot have empty filename")
                    if col_names is None:
                        col_names = [func.__name__]

                    return self._replay_next(
                        filename, self.datetime_col, col_names, reducer
                    )
                else:
                    return func(*_args, **_kwargs)

            return _replay

        # This deals with the standard decorator case (no brackets)
        # whereby: no kwargs + standard func positional arg
        # is given.
        if len(kwargs) == 0 and len(args) > 0 and callable(args[0]):
            logger.debug("Standard decorator")
            return partial(decorator, args[0])
        # Otherwise, return the actual decorator function
        else:
            logger.debug("Returning actual decorator")
        return decorator

    def _get_elapsed_time_relative_to(self, first_time: pd.Timestamp) -> pd.Timestamp:
        """
        Gets the elapsed time since the executor started and adds it to
        the first_time given. The elapsed time is therefore relative to
        the input.
        """
        start_time: datetime = self._state.get_start_time()
        logger.debug(f"Start_time: {start_time}")
        now: datetime = datetime.now()
        # TODO manually code the first call to return index 0 to not
        # skip the first index...not rounding
        # as that would make it get stuck
        delta_in_seconds: pd.Timedelta = pd.Timedelta(
            (now - start_time).total_seconds(), "seconds"
        )
        # first_time: pd.Timestamp = df.iloc[0].name
        proposed_time: pd.Timestamp = first_time + delta_in_seconds

        logger.debug(f"now: {now}")
        logger.debug(f"delta_tmp: {(now - start_time).total_seconds()}")
        logger.debug(f"delta_in_seconds: {delta_in_seconds}")
        logger.debug(f"first_time: {first_time}")
        logger.debug(f"proposed_time: {proposed_time}")

        return proposed_time

    def _find_next_datum(self, df: pd.DataFrame) -> int:
        """
        Finds the next row in the given dataframe indexed by
        datetime, based on the elapsed time.
        """
        first_time: pd.Timestamp = self._get_first_time(df)
        proposed_time: pd.Timestamp = self._get_elapsed_time_relative_to(first_time)

        # Find the nearest time using the proposed time
        nearest_i = df.index.get_indexer(pd.Index([proposed_time]), method="backfill")[
            0
        ]

        logger.debug(f"Nearest i: {nearest_i}")
        logger.debug(f"first_time: {first_time}")
        logger.debug(f"proposed_time: {proposed_time}")
        self._state._last_sense_hat_row_index = nearest_i

        if not self.configuration.no_wait_images:
            actual_time = df.iloc[nearest_i].name
            logger.debug(f"Actual time: {actual_time}")
            actual_delta: int = (actual_time - first_time).total_seconds()
            logger.debug(f"Actual delta: {actual_delta}")
            cutoff: datetime = self._state.get_start_time() + timedelta(
                seconds=actual_delta
            )
            logger.debug(f"Cutoff: {cutoff}")
            delta = (cutoff - datetime.now()).total_seconds()
            logger.debug(f"Replay delta: {delta}")
            if delta > 0:
                logger.debug("Sleeping until delta has passed")
                time.sleep(delta)

        return nearest_i

    @functools.cache
    def _df_from_replay_file(self, filename: str, datetime_col: str) -> pd.DataFrame:
        # Detect file type
        suffix = filename.split(".")[-1]
        if suffix == "csv":
            df = pd.read_csv(filename, parse_dates=[datetime_col])
        elif suffix == "tsv":
            df = pd.read_csv(filename, sep="\t", parse_dates=[datetime_col])
        elif suffix == "parquet":
            df = pd.read_parquet(
                filename,
            )
        else:
            raise AstroPiReplayException(f"Unsupported filetype '{suffix}'.")
        df = df.set_index(datetime_col)
        return df

    def _get_first_time(self, df: pd.DataFrame):
        return df.iloc[0].name

    def _add_network_time(self, network_time: float) -> None:
        self._state._network_time += network_time

    def _get_downloader(self) -> Downloader:
        if self.downloader is None:
            self.downloader = TimedDownloader(self._add_network_time)
        return self.downloader

    def _interpolate(
        self, datetime_col: str, col_names: list[str], df: pd.DataFrame
    ) -> pd.DataFrame:
        first_time: pd.Timestamp = self._get_first_time(df)
        d: pd.Timestamp = self._get_elapsed_time_relative_to(first_time)
        sub_df_dict: dict[str, list[Union[float, int, datetime]]] = {
            datetime_col: [d.timestamp()]
        }
        for col_name in col_names:
            if col_name not in self.interpolators:
                self.interpolators[col_name] = sp.interpolate.interp1d(
                    df.index.map(datetime.timestamp), df[col_name].to_numpy()
                )
            interpolator = self.interpolators[col_name]
            try:
                # convert to pydatetime to ensure in same timezone
                # as the interpolated x values
                value = interpolator(d.to_pydatetime(warn=False).timestamp())
            except ValueError:
                logger.debug(traceback.format_exc())
                if d.timestamp() < interpolator.x[0]:
                    value = interpolator.y[0]
                else:
                    value = interpolator.y[-1]
            sub_df_dict[col_name] = value
        sub_df: pd.DataFrame = pd.DataFrame.from_dict(sub_df_dict)
        sub_df = sub_df.set_index(datetime_col)
        return sub_df.iloc[0]

    @staticmethod
    def _reset():
        AstroPiExecutor._instance = None
        AstroPiExecutor._df_from_replay_file.cache_clear()
        AstroPiExecutor._state = AstroPiExecutorState()

    def _replay_next(
        self,
        filename: str,
        datetime_col: str,
        col_names: list[str],
        reducer: Callable[[pd.DataFrame], object] = lambda s: s.iloc[0],
        allow_interpolation: bool = True,
    ) -> object:
        """Internal method that opens the given filename,
        downloading it if required, and returns the given col names,
        using the reducer. In effect, this replays the data.

        allow_interpolation: Whether to respect the interpolate_sense_hat
        variable.
        """

        file_path: Path = Path(filename)
        if not file_path.exists() and self.configuration.streaming_mode:
            # download the file
            self._get_downloader().fetch_sequence_file(file_path)

        df = self._df_from_replay_file(filename, datetime_col)

        for col_name in col_names:
            if col_name not in df.columns:
                raise AstroPiReplayException(
                    f"Column '{col_name}' not found "
                    + f"in file '{filename}'.\n\n"
                    + "Detected columns: \n\t"
                    + ", ".join(df.columns)
                )

        if allow_interpolation and self.configuration.interpolate_sense_hat:
            return reducer(self._interpolate(datetime_col, col_names, df))

        nearest_i = self._find_next_datum(df)

        return reducer(df[col_names].iloc[nearest_i])

    @staticmethod
    def _check_package_installed(venv_python3) -> subprocess.CompletedProcess[str]:
        """
        Runs a program using the venv Python to check if
        the current package is installed.
        """
        dynamic_program: str = "; ".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "module = importlib.util.find_spec(" + f"'{PROGRAM_NAME}')",
                f"to_print = '{AstroPiExecutor.NOT_FOUND}' if module is None "
                + "else Path(module.origin).parent",
                "print(to_print)",
            ]
        )

        out = subprocess.run(  # nosec B603: no user input
            [venv_python3, "-c", dynamic_program],
            check=True,
            capture_output=True,
            text=True,
        )
        return out

    def time_since_start(self) -> datetime:
        """Time relative to the original start time, as specified
        in the metadata.json file"""
        execution_start_time: datetime = self._state.get_start_time()
        now: datetime = datetime.now()
        delta: timedelta = now - execution_start_time

        original_start_time: datetime = get_start_time()
        return original_start_time + delta

    @property
    def _has_ffmpeg(self) -> bool:
        try:
            subprocess.run(  # nosec B603, B607
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except FileNotFoundError:
            return False

    @property
    def _has_ffprobe(self) -> bool:
        try:
            subprocess.run(  # nosec B603, B607
                ["ffprobe", "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except FileNotFoundError:
            return False

    @property
    def _has_tkinter(self) -> bool:
        try:
            import tkinter  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _detect_execution_mode() -> ExecutionMode:
        return (
            ExecutionMode.LIVE
            if all(
                importlib.util.find_spec(module) is not None
                for module in AstroPiExecutor.MODULES_TO_STUB
            )
            else ExecutionMode.REPLAY
        )

    @staticmethod
    def add_name_is_main_guard(main: Path) -> Path:
        """
        Checks if the file at the given path includes an if name == "__main__"
        expression. If it does, returns the same file.
        Otherwise, returns a modified copy of the file with the original contents
        inside the if expression.
        """
        substrings: list[str] = [
            'if __name__ == "__main__":',
            "if __name__ == '__main__':",
        ]
        with main.open("r") as f:
            contents = f.read().strip()

        includes_guard = False
        for substring in substrings:
            if substring in contents:
                includes_guard = True
        logger.debug(f"Main file includes guard: {includes_guard}")

        if not includes_guard:
            # 1. Copy the original file to a temp folder
            tempdir = Path(tempfile.gettempdir())
            executor_temp_dir: Path = tempdir / PROGRAM_NAME
            executor_temp_dir.mkdir(exist_ok=True)
            original_main: Path = executor_temp_dir / "original_main.py"
            logger.debug(f"Copying original main to {original_main}")
            shutil.copy2(main, original_main)

            # 2. Write modifications to a temp file
            tabbed = os.linesep.join([f"    {line}" for line in contents.splitlines()])
            if len(tabbed) == 0:
                tabbed = "    pass"
            main_copy: Path = tempdir / "main.py"
            with main_copy.open("w") as f:
                f.write(substrings[0] + os.linesep)
                f.write(tabbed)

            # 3. Temporarily overwrite the main.py in the original location
            # with the modified version
            logger.debug(f"Overwriting {main} with {main_copy}")
            shutil.copy2(main_copy, main)

            # 4. Add teardown to replace the modified main.py with the
            # original after execution
            AstroPiExecutor._register_callback(
                Lifecycle.AFTER, lambda: shutil.copy2(original_main, main)
            )

        return main

    @staticmethod
    def add_debug_logging_config(main: Path) -> Path:
        logger.debug("Setting log level inside main")
        tempdir: Path = Path(tempfile.gettempdir()) / PROGRAM_NAME
        main_copy: Path = tempdir / "main_debug_copy.py"
        with main.open() as f:
            contents = f.read()
        with main_copy.open("w") as f:
            f.write(
                # TODO only write import datetime
                # if not already overloaded by from datetime import datetime
                os.linesep.join(
                    [
                        "import logging as replay_tool_logging",
                        "replay_tool_logging.basicConfig("
                        + "level=replay_tool_logging.DEBUG,"
                        + f"format='{LOGGING_FORMAT}')",
                        "replay_tool_logging.getLogger('PIL').setLevel("
                        + "replay_tool_logging.INFO)",
                        "import datetime as replay_tool_datetime",
                        "replay_tool_logging.Formatter.formatTime = ("
                        + "lambda self, record, "
                        + "datefmt=None: replay_tool_datetime.datetime.fromtimestamp("
                        + "record.created, "
                        + "replay_tool_datetime.timezone.utc).astimezone().isoformat("
                        + "sep='T',timespec='milliseconds'))"
                        + os.linesep,
                    ]
                )
            )
            f.write(contents)

        logger.debug(f"Copying {main_copy} into {main}")
        shutil.copy2(main_copy, main)
        # Teardown is taken care of in add_name_is_main_guard
        return main

    @staticmethod
    def _register_callback(lifecycle: Lifecycle, callback: Callable) -> None:
        """
        Registers a callback to be executed at a specific point in
        the executor lifecycle
        """
        AstroPiExecutor._callbacks[lifecycle].append(callback)

    @staticmethod
    def _run_callbacks(lifecycle: Lifecycle) -> None:
        """
        Runs the registered callbacks for the specified point in the executor
        lifecycle
        """
        for callback in AstroPiExecutor._callbacks[lifecycle]:
            callback()

    @staticmethod
    def _setup_venv(venv_dirname: Path, name: str = "venv") -> VenvResolver:
        venv_dir: Path = venv_dirname / name

        already_exists: bool = venv_dir.exists()

        # Copy or creates the venv to the venv_dir, depending on if we're
        # already in one
        venv_resolver: VenvResolver = VenvResolver(venv_dir)

        if already_exists and not venv_resolver.rebuilt:
            logger.debug("Venv already exists and is compatible - ")
            logger.debug(
                "therefore assuming that the executor and its stubs "
                + "do not need to be installed"
            )
            return venv_resolver

        # Install the executor package
        # transitive dependencies are covered due to the --system-site-packages
        logger.debug("Installing stubbed modules in the venv...")

        executor_install_path: Optional[str] = venv_resolver.is_package_installed(
            PROGRAM_NAME
        )
        if executor_install_path is None:
            logger.debug(f"Installing {PROGRAM_CMD_NAME} into venv...")

            # Copies the resources already downloaded as well
            shutil.copytree(
                PACKAGE_ROOT,
                venv_resolver.venv_info.site_packages_dir / PROGRAM_NAME,
                dirs_exist_ok=True,
            )

            executor_install_path = venv_resolver.is_package_installed(PROGRAM_NAME)
            if executor_install_path is None:
                raise AstroPiReplayException(
                    f"Could not set up {PROGRAM_CMD_NAME} environment"
                )

        # Install stubs into the venv
        logger.debug(f"Installing stubs into venv at {executor_install_path}")
        venv_resolver.copy_stubs(AstroPiExecutor.MODULES_TO_STUB, executor_install_path)

        return venv_resolver

    @staticmethod
    def install_global() -> None:
        logger.debug("Installing stubbed modules in the venv...")
        destination: str = site.getsitepackages()[0]

        for module in AstroPiExecutor.MODULES_TO_STUB:
            logger.debug(f"Installing {module}")
            shutil.copytree(
                Path(__file__).parent / module,
                Path(destination) / module,
            )
        logger.debug("Done")

    @staticmethod
    def run(
        execution_mode: Optional[ExecutionMode],
        venv_dirname: Optional[Path],
        main: Path,
        debug: bool = os.environ.get(f"{PROGRAM_NAME.upper()}_DEBUG", None) is not None,
    ) -> None:
        """
        This method runs the given main file using the given execution mode.
        If the passed in execution mode is Replay mode, then creates a venv
        in the venv_dirname if it does not already exist.

        execution_mode: Whether to replay data or capture live data.
        venv_dirname: The directory to create the venv in replay mode, if it does not
        exist.
        main: The filename to execute - generally called main.py.
        """

        if execution_mode is None:
            execution_mode = AstroPiExecutor._detect_execution_mode()
            logger.debug(f"Detected execution mode: {execution_mode}")
        if not main.exists() or not main.is_file():
            raise AstroPiReplayException(f"File {main} is not a regular file")

        env: Optional[dict[str, str]]
        python: str

        # Conditionally create the venv
        # TODO create spinner/progress bar for this
        if execution_mode == ExecutionMode.REPLAY:
            if venv_dirname is None:
                logging.debug("venv_dirname is None - fetching value from env")
                venv_dirname = get_default_venv_dir()
                logging.debug(f"Found {venv_dirname}")

            venv: VenvResolver = AstroPiExecutor._setup_venv(venv_dirname)

            # Prepare the environment to be used in the subprocess.
            env = os.environ.copy()
            env["PATH"] = os.path.pathsep.join(
                [str(venv.venv_info.script_dir), env["PATH"]]
            )
            env["VIRTUAL_ENV"] = str(venv.venv_dir)

            python = str(venv.venv_info.python)
        else:
            logging.debug("Running in live mode")
            env = None
            python = "python.exe" if sys.platform == "win32" else "python"
            resolved_python: Optional[str] = shutil.which(python)
            if resolved_python is not None:
                python = resolved_python
            else:
                raise Exception(f"Could not find {python}. Is it installed?")

        # Add if __name__ == "__main__" guard as needed
        # (required by multiprocessing in CameraPreview currently FIXME)
        main = AstroPiExecutor.add_name_is_main_guard(main)

        if debug:
            main = AstroPiExecutor.add_debug_logging_config(main)

        try:
            # Run the program that was passed in
            # TODO this is checked already
            if platform.system() in ["Linux", "Darwin", "Windows"]:
                # -u is for unbuffered Python, which is what is used on the
                # Astro Pis on the ISS.
                args: list[str] = [rf"{python}", "-u", str(main.resolve())]
                logger.debug(f"Executing '{' '.join(args)}' in subprocess")

                def custom_excepthook(type, value, tb):
                    """Hides the internals of the lib
                    from the stack trace"""
                    size = len(list(traceback.walk_tb(tb)))
                    traceback.print_tb(tb, size - 5)

                sys.excepthook = custom_excepthook
                subprocess.run(
                    args, env=env if env is not None else env, check=True
                )  # nosec B603: runs main as intended

            else:
                raise OSError(f"Unsupported system {os}")
        finally:
            AstroPiExecutor._run_callbacks(Lifecycle.AFTER)  # teardown


# Integration tests:
# - using qemu?
# - On Windows, Linux, Darwin ensure that the adapter can be installed
# - On RP4, ensure it calls the real lib (integration test) - I should test this now...!
# perhaps an i2c bus can be emulated...
