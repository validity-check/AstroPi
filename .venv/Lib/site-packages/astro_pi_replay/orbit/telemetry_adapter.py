# import functools
import typing
from datetime import datetime, timezone
from typing import Callable, Optional

import skyfield.api
from skyfield.positionlib import ICRF, Barycentric, Geocentric
from skyfield.timelib import Time
from skyfield.toposlib import GeographicPosition

from astro_pi_replay.executor import AstroPiExecutor

from .telemetry import ISS as _ISS
from .telemetry import _timescale, coordinates


class EarthSatellite(skyfield.api.EarthSatellite):
    """Desired subclass type signature"""

    def coordinates(self) -> GeographicPosition:
        return coordinates(self)

    def at(self, x: Time) -> typing.Union[Barycentric, Geocentric, ICRF]:
        # Although the value is not used in this stub, it is nice to check
        # the type since it will fail if it is incorrect in the real
        # implementation.
        if not isinstance(x, Time):
            raise ValueError(
                "please provide the at() method with a Time "
                + f"instance as its argument, instead of the value {str(x)}"
            )
        new_t: Time = self._now(self.get_executor())
        return super().at(new_t)

    def _now(self, executor: AstroPiExecutor) -> Time:
        """
        Gets the relative time since the start from the executor
        and converts it.
        """
        new_time: datetime = executor.time_since_start()
        new_time = new_time.replace(tzinfo=timezone.utc)
        return _timescale.from_datetime(new_time)

    def set_executor(self, executor: AstroPiExecutor):
        self.executor = executor

    def get_executor(self) -> AstroPiExecutor:
        return self.executor


def get_patched_iss(
    executor: typing.Optional[AstroPiExecutor] = None,
) -> EarthSatellite:
    """
    Patches the timescale object used by the ISS EarthSatellite so that
    times are relative to the start time of the replayed experiment.
    The start time is stored in the metadata.json file
    """
    if executor is None:
        executor = AstroPiExecutor()

    b = _ISS()
    b.__class__ = EarthSatellite
    b = typing.cast(EarthSatellite, b)
    b.set_executor(executor)
    b.at = EarthSatellite.at.__get__(b)

    return b


ISS: Callable[[Optional[AstroPiExecutor]], EarthSatellite] = get_patched_iss

# Instead of doing at from the time given, instead do it from time_since_start
