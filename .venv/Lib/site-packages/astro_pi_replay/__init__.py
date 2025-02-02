from pathlib import Path

__version__ = "1.2.0"
PROGRAM_CMD_NAME = "Astro-Pi-Replay"
PROGRAM_NAME = "astro_pi_replay"
PACKAGE_ROOT = Path(__file__).parent
LOGGING_FORMAT: str = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
ONLINE_NAME: str = PROGRAM_NAME.replace("_", "-") + "-online"
