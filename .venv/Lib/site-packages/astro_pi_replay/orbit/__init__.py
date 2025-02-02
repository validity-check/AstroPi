"orbit: Module for interfacing with the Astro Pi"
from .telemetry import ephemeris
from .telemetry_adapter import ISS

__project__ = "orbit"
__version__ = "2.0.0"
__requires__ = ["skyfield"]
__entry_points__: dict[str, list[str]] = {}
__scripts__: list[str] = []

__all__ = ["ephemeris", "ISS"]
