import logging
import typing
from pathlib import Path

import skyfield.api
from skyfield.api import Loader, Timescale, load
from skyfield.jpllib import SpiceKernel
from skyfield.positionlib import Geocentric
from skyfield.timelib import Time
from skyfield.toposlib import GeographicPosition

from astro_pi_replay.resources import get_resource, get_tle

logger = logging.getLogger(__name__)
_BSP_FILE: Path = get_resource("de421.bsp")
_timescale: Timescale = load.timescale()


def coordinates(satellite: skyfield.api.EarthSatellite) -> GeographicPosition:
    """
    Return a Skyfield GeographicPosition object corresponding to the  Earth
    latitude and longitude beneath the current celestial position of the ISS.

    See: rhodesmill.org/skyfield/api-topos.html#skyfield.toposlib.GeographicPosition
    """
    now: Time = _timescale.now()
    rel_pos: Geocentric = typing.cast(Geocentric, satellite.at(now))
    return rel_pos.subpoint()


def load_ephemeris() -> SpiceKernel:
    loader: Loader = Loader(_BSP_FILE.parent, verbose=False)
    return typing.cast(SpiceKernel, loader(_BSP_FILE.name))


def load_iss() -> skyfield.api.EarthSatellite:
    tle_file = get_tle()
    loader: Loader = Loader(tle_file.parent, verbose=False)
    satellites: list[skyfield.api.EarthSatellite] = loader.tle_file(tle_file.name)
    iss = next((sat for sat in satellites if sat.name == "ISS (ZARYA)"), None)
    if iss is None:
        raise RuntimeError(f"Unable to retrieve ISS TLE data from {str(tle_file)}")

    # bind the `coordinates` function to the ISS object as a method
    setattr(iss, "coordinates", coordinates.__get__(iss, iss.__class__))
    return iss


# create ISS as a Skyfield EarthSatellite object
# See: rhodesmill.org/skyfield/api-satellites.html#skyfield.sgp4lib.EarthSatellite
ISS: typing.Callable[[], skyfield.api.EarthSatellite] = load_iss

# Expose ephemeris in the API
ephemeris: SpiceKernel = load_ephemeris()
