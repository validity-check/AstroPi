import os
from pathlib import Path

from skyfield.api import Loader, load

_tle_dir: Path = Path(os.environ.get("TLE_DIR", Path.home()))
_tle_filename = "iss.tle"
_tle_url = "http://celestrak.com/NORAD/elements/stations.txt"

_bsp_dir = os.environ.get("BSP_DIR", Path.home())
_bsp_421_filename = "de421.bsp"
_bsp_440s_filename = "de440s.bsp"

_timescale = load.timescale()


def _load_iss():
    """
    Retrieves ISS telemetry data from a local or remote TLE file and
    returns a Skyfield EarthSatellite object corresponding to the ISS.
    """
    loader = Loader(_tle_dir, verbose=False)
    try:
        # find telemetry data locally
        satellites = loader.tle_file(_tle_filename)
    except FileNotFoundError:
        pass
    else:
        iss = next((sat for sat in satellites if sat.name == "ISS (ZARYA)"), None)
        if iss is None:
            raise RuntimeError(
                f"Unable to retrieve ISS TLE data from {loader.path_to(_tle_filename)}"
            )
        return iss

    try:
        # find telemetry data remotely
        satellites = loader.tle_file(_tle_url)
        Path(_tle_dir / Path(_tle_url).name).rename(_tle_dir / _tle_filename)
    except Exception as e:
        print(e)
        pass
    else:
        iss = next((sat for sat in satellites if sat.name == "ISS (ZARYA)"), None)
        if iss is None:
            raise RuntimeError(f"Unable to retrieve ISS TLE data from {_tle_url}")
        return iss

    raise FileNotFoundError(
        "Unable to retrieve ISS TLE data: "
        + f"cannot find {loader.path_to(_tle_filename)} or download {_tle_url}."
    )


def load_iss():
    ISS = _load_iss()
    # bind the `coordinates` function to the ISS object as a method
    setattr(ISS, "coordinates", coordinates.__get__(ISS, ISS.__class__))
    return ISS


def coordinates(satellite):
    """
    Return a Skyfield GeographicPosition object corresponding to the  Earth
    latitude and longitude beneath the current celestial position of the ISS.

    See: rhodesmill.org/skyfield/api-topos.html#skyfield.toposlib.GeographicPosition
    """
    return satellite.at(_timescale.now()).subpoint()


def load_ephemeris(bsp_filename):
    loader = Loader(_bsp_dir, verbose=False)
    return loader(bsp_filename)


# create ISS as a Skyfield EarthSatellite object
# See: rhodesmill.org/skyfield/api-satellites.html#skyfield.sgp4lib.EarthSatellite
ISS = load_iss


# Expose ephemeris in the API
de421 = load_ephemeris(_bsp_421_filename)
de440s = load_ephemeris(_bsp_440s_filename)
ephemeris = de421
