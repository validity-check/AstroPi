import logging

import colorzero as c0

logger = logging.getLogger(__name__)


NAMED_COLORS = c0.tables.NAMED_COLORS
Red = c0.Red
Green = c0.Green
Blue = c0.Blue
Hue = c0.Hue
Lightness = c0.Lightness
Saturation = c0.Saturation


class Color(c0.Color):
    def __new__(cls, *args, **kwargs):
        logger.warn(
            "The picamera.color module and Color class are deprecated; "
            + "please use the colorzero library (same API) instead"
        )
        return c0.Color.__new__(cls, *args, **kwargs)
