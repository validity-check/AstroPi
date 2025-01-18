import dataclasses
import fileinput
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from astro_pi_replay import PROGRAM_NAME, __version__
from astro_pi_replay.exception import AstroPiReplayException, AstroPiReplayRuntimeError
from astro_pi_replay.version_utils import compare_semver

logger = logging.getLogger(__name__)

VENV_REPLAY_VERSION_FILE_NAME: str = "astro_pi_replay_version.txt"
VENV_CONFIG_FILE_NAME: str = "pyvenv.cfg"


@dataclasses.dataclass
class VenvInfo:
    activate: Path
    deactivate: Optional[Path]
    executor: Optional[Path]
    pip: Path
    python: Path
    script_dir: Path
    site_packages_dir: Path


class VenvResolver:
    """
    Wrapper class used to interact with a Python venv
    """

    # TODO: move these to __init__ since this is package wide
    SUPPORTED_PLATFORMS: list[str] = ["linux", "win32", "darwin"]
    ALLOW_UNSUPPORTED_PLATFORM: bool = False

    NOT_FOUND = f"{PROGRAM_NAME} not found"

    def _setup(
        self, _venv_dir: Optional[Union[str, Path]], modify_venv_dir: bool
    ) -> tuple[Path, VenvInfo, bool]:
        # Separated function for type-checking
        venv_dir: Path
        venv_info: VenvInfo

        should_rebuild: bool = False
        if _venv_dir is not None and Path(_venv_dir).exists():
            logger.debug(f"Venv {_venv_dir} exists...")
            logger.debug("Checking if it should be rebuilt...")
            should_rebuild = VenvResolver._should_rebuild_venv(Path(_venv_dir))
            if should_rebuild and modify_venv_dir:
                logger.debug(
                    f"Removing venv_dir {_venv_dir} as it is "
                    + "from an old installation."
                )
                shutil.rmtree(_venv_dir)
                logger.debug(f"Reinitalising venv at {_venv_dir}")
                venv_dir, venv_info = self._init_venv(Path(_venv_dir))
            elif should_rebuild:
                raise AstroPiReplayException(
                    "Rebuild required but modify_venv_dir is False. Aborting"
                )
            else:
                logger.debug("Venv exists and does not need rebuilding!")
                venv_dir = Path(_venv_dir)
                venv_info = VenvResolver.resolve_venv_dirs(venv_dir, self.platform)
        elif _venv_dir is not None:
            logger.debug(f"Venv given, {_venv_dir}, does not exist")
            if modify_venv_dir:
                logger.debug(f"Initialising {_venv_dir}...")
                venv_dir, venv_info = self._init_venv(Path(_venv_dir))
            else:
                raise AstroPiReplayRuntimeError(
                    "venv_dir does not exist, but modify_venv_dir is False. Aborting"
                )
        else:
            logger.debug("venv_dir is None (does not exist)")
            if modify_venv_dir:
                logger.debug("Initialising...")
                venv_dir, venv_info = self._init_venv()
            else:
                raise AstroPiReplayRuntimeError(
                    "venv_dir is None " + "but modify_venv_dir is False. Aborting"
                )

        return (venv_dir, venv_info, should_rebuild)

    def __init__(
        self, venv_dir: Optional[Union[str, Path]] = None, modify_venv_dir: bool = True
    ) -> None:
        """
        modify_venv_dir: Whether to modify the venv dir by initialising
        or reinitialising the venv where necessary.
        """
        self.platform: str = self._verify_platform()
        self.venv_dir: Path
        self.venv_info: VenvInfo
        self.rebuilt: bool
        self.venv_dir, self.venv_info, self.rebuilt = self._setup(
            venv_dir, modify_venv_dir
        )

    def copy_stubs(self, stubs: list[str], install_path: str):
        logger.debug("Installing stubbed modules in the venv...")

        for module in stubs:
            logger.debug(f"Installing {module}")
            shutil.copytree(
                Path(install_path) / module,
                self.venv_info.site_packages_dir / module,
            )

    def install(
        self,
        name: str,
        workdir: Optional[Path] = None,
        flags: Optional[list[str]] = None,
        editable: bool = False,
    ) -> None:
        """Executes pip install with the given args using the resolved pip"""
        before_directory: str = os.getcwd()
        chdir: bool = False
        try:
            logger.debug(f"Initial working directory: {str(os.getcwd())}")
            if workdir is not None:
                os.chdir(workdir)
                logger.debug(f"Workir changed to {workdir}")
                chdir = True
            print_name: str = name
            if name == os.curdir and workdir is not None:
                print_name = workdir.name
            logger.debug(f"Installing {print_name} into venv...")
            args: list[str] = [
                str(self.venv_info.pip),
                "install",
                name,
                "--disable-pip-version-check",
            ]
            if editable:
                args.insert(2, "--editable")
            if flags is not None:
                args = [str(self.venv_info.pip), "install"] + flags + [name]
            logger.debug(" ".join(args))

            subprocess.run(
                args, check=True, stdout=subprocess.DEVNULL
            )  # nosec B603: no user input
        finally:
            if chdir:
                os.chdir(before_directory)

    def is_package_installed(self, name: str) -> Optional[str]:
        """
        Runs a program using the venv Python to check if
        the given package is installed.
        """
        dynamic_program: str = "; ".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "module = importlib.util.find_spec(" + f"'{name}')",
                f"to_print = '{VenvResolver.NOT_FOUND}' if module is None "
                + "else Path(module.origin).parent",
                "print(to_print)",
            ]
        )

        args: list[str] = [str(self.venv_info.python), "-c", dynamic_program]
        logger.debug(" ".join(args))
        out = subprocess.run(  # nosec B603: no user input
            args,
            check=True,
            capture_output=True,
            text=True,
        )

        stripped: str = out.stdout.strip()
        found: bool = stripped != VenvResolver.NOT_FOUND
        if found:
            logger.debug(f"Found {PROGRAM_NAME} installed at {stripped}")
            return stripped
        return None

    def is_in_venv(self) -> bool:
        return sys.prefix != sys.base_prefix

    def list_dependencies(self) -> str:
        """Executes pip freeze using the resolved pip"""
        args: list[str] = [str(self.venv_info.python), "-m", "pip", "freeze"]
        logger.debug(" ".join(args))
        out = subprocess.run(
            args, text=True, check=True, capture_output=True, shell=False
        )  # nosec B603
        logger.debug(out)
        return out.stdout

    @staticmethod
    def resolve_venv_dirs(venv_dir: Path, platform: str) -> VenvInfo:
        activate: Path
        deactivate: Optional[Path]
        executor: Optional[Path]
        pip: Path
        python: Path
        script_dir: Path
        site_packages_dir: Path

        script_dir = venv_dir / "bin"
        site_packages_dir = venv_dir / "lib"
        site_packages_dir /= f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages_dir /= "site-packages"
        activate = script_dir / "activate"
        deactivate = script_dir / "deactivate"
        executor = script_dir / PROGRAM_NAME
        pip = script_dir / "pip"
        python = script_dir / "python"

        if platform in ["win32"]:
            script_dir = venv_dir / "Scripts"
            site_packages_dir = venv_dir / "Lib" / "site-packages"
            activate = script_dir / (activate.name + ".bat")
            deactivate = script_dir / (deactivate.name + ".bat")
            executor = script_dir / (executor.name + ".exe")
            pip = script_dir / (pip.name + ".exe")
            python = script_dir / (python.name + ".exe")

        mandatory_paths: list[Path] = [
            script_dir,
            site_packages_dir,
            activate,
            pip,
            python,
        ]
        for path in mandatory_paths:
            if not path.exists():
                error_message: str = (
                    f"Couldn't resolve {path} in {venv_dir}. "
                    + "Try deleting the venv and recreating. If this issue persists, "
                    + f"please log an issue on github.com/{PROGRAM_NAME}."
                )
                logger.error(error_message)
                raise AstroPiReplayRuntimeError(error_message)

        return VenvInfo(
            activate, deactivate, executor, pip, python, script_dir, site_packages_dir
        )

    def _init_venv(self, venv_dir: Path = Path("venv")) -> tuple[Path, VenvInfo]:
        if self.platform == "win32":
            # Windows venvs may/do not support symlinks
            # and will emit a warning if unsupported (and then default
            # to copying). For a more user-friendly experience, these
            # warnings are suppressed.
            logging.getLogger("venv").setLevel("ERROR")

        logger.info("Preparing environment (this may take a few moments)...")
        try:
            import venv

            venv.create(
                venv_dir, symlinks=True, system_site_packages=True, with_pip=True
            )
        except subprocess.CalledProcessError as e:
            import traceback

            logger.error(e)
            logger.error(e.stdout)
            logger.error(e.stderr)
            traceback.print_exc()
            raise e
        (venv_dir / VENV_REPLAY_VERSION_FILE_NAME).write_text(__version__)

        venv_info: VenvInfo = VenvResolver.resolve_venv_dirs(venv_dir, self.platform)

        # upgrade pip (must be >= 22.3 for a specific bugfix)
        # see: https://github.com/pypa/pip/issues/6264#issuecomment-1088660972
        update_pip_args: list[str] = [
            str(venv_info.python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
        logger.debug(f"Upgrading pip: {' '.join(update_pip_args)}")
        subprocess.run(
            update_pip_args, check=True, stdout=subprocess.DEVNULL  # nosec B603
        )

        if self.is_in_venv():
            logger.debug(
                "Detected that you running in a venv:"
                + f"\n\t{sys.prefix}.\n"
                + "However, running in replay mode will use a "
                + "separate copied (modified) venv."
            )

            # Incorporate dependencies from the current venv into the
            # Astro-Pi-Replay venv
            with (venv_info.site_packages_dir / "extra.pth").open("w") as f:
                current_venv_info: VenvInfo = VenvResolver.resolve_venv_dirs(
                    Path(sys.prefix).resolve(), self.platform
                )

                f.write(str(current_venv_info.site_packages_dir))

        if self.platform == "win32":
            logging.getLogger("venv").setLevel("WARNING")

        return venv_dir, venv_info

    @staticmethod
    def _should_rebuild_venv(venv_dir: Path) -> bool:
        config: Path = venv_dir / VENV_CONFIG_FILE_NAME
        try:
            venv_python_version: str = [
                line.strip()
                for line in config.read_text().split("\n")  # works on Windows as well
                if line.startswith("version")
            ][0].split(" ")[2]
        except IndexError:
            logger.debug(f"{VENV_CONFIG_FILE_NAME} in {venv_dir} " + "in bad format")
            return True

        current_version: str = (
            f"{sys.version_info.major}."
            + f"{sys.version_info.minor}.{sys.version_info.micro}"
        )
        comparison_result: int = compare_semver(
            current_version, venv_python_version, ignore_patch=True
        )

        rebuild_venv: bool = False
        if comparison_result == 1:
            # current version is greater than venv_version
            logger.debug(
                "Current python version is greater than "
                + "the python version used to create the venv"
            )
            rebuild_venv = True
        elif comparison_result == -1:
            # current version is less than venv_version
            logger.warning(
                f"Current Python version ({current_version}) "
                + f"is less than version {venv_python_version} used to "
                f"create the venv at {venv_dir}."
            )
            rebuild_venv = True

        if not rebuild_venv:
            # check that the replay version has not changed
            replay_version_file: Path = venv_dir / VENV_REPLAY_VERSION_FILE_NAME
            logger.debug(f"Reading {replay_version_file}")
            if not replay_version_file.exists():
                logger.debug(f"File {replay_version_file} does not exist")
                rebuild_venv = True
            else:
                venv_replay_version: str = replay_version_file.read_text()
                logger.debug(f"Venv replay version: {venv_replay_version}")
                if compare_semver(__version__, venv_replay_version) > 0:
                    logger.debug("Venv replay version is old")
                    rebuild_venv = True

        if rebuild_venv:
            logger.debug("Venv recreation required")
            return True
        else:
            logger.debug("Venv rebuild not required")
            return False

    def _verify_platform(self) -> str:
        """Verifies that the curent platform is supported. Returns
        True if the current platform is supported, otherwise raises
        an AstroPiExecutorRuntimeError.
        """
        if (
            sys.platform not in VenvResolver.SUPPORTED_PLATFORMS
            and not VenvResolver.ALLOW_UNSUPPORTED_PLATFORM
        ):
            message: str = os.linesep.join(
                [
                    f"{sys.platform} is not supported. The currently "
                    + "supported platforms are: "
                    + f"{VenvResolver.SUPPORTED_PLATFORMS}. To override this "
                    + "use the --allow-unsupported-platform option."
                ]
            )
            logger.error(message)
            raise AstroPiReplayRuntimeError(message)
        return sys.platform

    @staticmethod
    def _copy_venv(source: Union[str, Path], destination: Union[str, Path]):
        shutil.copytree(source, destination, symlinks=True)

        if sys.platform in ["linux", "darwin"]:
            # modify each shebang (and other paths) in
            # the bin directory to the new location

            for line in fileinput.FileInput(
                files=(
                    full_path
                    for parent, _, children in os.walk(Path(destination) / "bin")
                    for child in children
                    if not (full_path := Path(parent) / child).is_symlink()
                    and not full_path.name.endswith(".pyc")
                ),
                inplace=True,
            ):
                print(line.replace(str(source), str(destination)), end="")

        elif sys.platform in ["win32"]:
            # On Windows the executables contain hardcoded paths so
            # all the files in the Scripts directory must be read
            # as bytes to avoid decoding errors
            for file in (
                full_path
                for parent, _, children in os.walk(Path(destination) / "Scripts")
                for child in children
                if not (full_path := Path(parent) / child).is_symlink()
            ):
                with open(file, "rb") as f:
                    contents: bytes = f.read()
                with open(file, "wb") as f:
                    f.write(
                        contents.replace(
                            str(source).encode(), str(destination).encode()
                        )
                    )
        else:
            raise AstroPiReplayRuntimeError(f"Unsupported platform {sys.platform}")


"""
Set-ExecutionPolicy Bypass -Scope Process -Force
# and then, for example:
venv\\Scripts\\Activate.ps1
"""
