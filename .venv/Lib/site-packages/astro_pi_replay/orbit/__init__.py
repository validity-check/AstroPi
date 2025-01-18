"orbit: Module for interfacing with the Astro Pi"
# from .motion_sensor import MotionSensor
from .telemetry import ephemeris
from .telemetry_adapter import ISS

__project__ = "orbit"
__version__ = "1.2.1"
__requires__ = ["gpiozero", "skyfield"]
__entry_points__: dict[str, list[str]] = {}
__scripts__: list[str] = []

__all__ = ["ephemeris", "ISS"]
