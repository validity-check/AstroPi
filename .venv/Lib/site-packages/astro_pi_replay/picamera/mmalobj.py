# TODO copy the license from mmalobj.py in the real implementation
import io
import logging
import warnings
from collections import namedtuple
from fractions import Fraction

from astro_pi_replay.picamera.exc import PiCameraDeprecated, PiCameraValueError
from astro_pi_replay.picamera.streams import BufferIO

logger = logging.getLogger(__name__)


def open_stream(stream, output=True, buffering=65536):
    """
    This is the core of picamera's IO-semantics. It returns a tuple of a
    file-like object and a bool indicating whether the stream requires closing
    once the caller is finished with it.

    * If *stream* is a string, it is opened as a file object (with mode 'wb' if
      *output* is ``True``, and the specified amount of *bufffering*). In this
      case the function returns ``(stream, True)``.

    * If *stream* is a stream with a ``write`` method, it is returned as
      ``(stream, False)``.

    * Otherwise *stream* is assumed to be a writeable buffer and is wrapped
      with :class:`BufferIO`. The function returns ``(stream, True)``.
    """
    if isinstance(stream, bytes):
        stream = stream.decode("ascii")
    opened = isinstance(stream, str)
    if opened:
        stream = io.open(stream, "wb" if output else "rb", buffering)
    else:
        try:
            if output:
                stream.write
            else:
                stream.read
        except AttributeError:
            # Assume the stream is actually a buffer
            opened = True
            stream = BufferIO(stream)
            if output and not stream.writable:
                raise IOError("writeable buffer required for output")
    return (stream, opened)


def close_stream(stream, opened):
    """
    If *opened* is ``True``, then the ``close`` method of *stream* will be
    called. Otherwise, the function will attempt to call the ``flush`` method
    on *stream* (if one exists). This function essentially takes the output
    of :func:`open_stream` and finalizes the result.
    """
    if opened:
        stream.close()
    else:
        try:
            stream.flush()
        except AttributeError:
            pass


def to_resolution(value):
    """
    Converts *value* which may be a (width, height) tuple or a string
    containing a representation of a resolution (e.g. "1024x768" or "1080p") to
    a (width, height) tuple.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            # A selection from https://en.wikipedia.org/wiki/Graphics_display_resolution
            # Feel free to suggest additions
            w, h = {
                "VGA": (640, 480),
                "SVGA": (800, 600),
                "XGA": (1024, 768),
                "SXGA": (1280, 1024),
                "UXGA": (1600, 1200),
                "HD": (1280, 720),
                "FHD": (1920, 1080),
                "1080P": (1920, 1080),
                "720P": (1280, 720),
            }[value.strip().upper()]
        except KeyError:
            w, h = (int(i.strip()) for i in value.upper().split("X", 1))
    else:
        try:
            w, h = value
        except (TypeError, ValueError):
            raise PiCameraValueError("Invalid resolution tuple: %r" % value)
    return PiResolution(w, h)


def to_fraction(value, den_limit=65536):
    """
    Converts *value*, which can be any numeric type, an MMAL_RATIONAL_T, or a
    (numerator, denominator) tuple to a :class:`~fractions.Fraction` limiting
    the denominator to the range 0 < n <= *den_limit* (which defaults to
    65536).
    """
    try:
        # int, long, or fraction
        n, d = value.numerator, value.denominator
    except AttributeError:
        try:
            # float
            n, d = value.as_integer_ratio()
        except AttributeError:
            try:
                n, d = value.num, value.den
            except AttributeError:
                try:
                    # tuple
                    n, d = value
                    warnings.warn(
                        PiCameraDeprecated(
                            "Setting framerate or gains as a tuple is "
                            "deprecated; please use one of Python's many "
                            "numeric classes like int, float, Decimal, or "
                            "Fraction instead"
                        )
                    )
                except (TypeError, ValueError):
                    # try and convert anything else to a Fraction directly
                    value = Fraction(value)
                    n, d = value.numerator, value.denominator
    # Ensure denominator is reasonable
    if d == 0:
        raise PiCameraValueError("Denominator cannot be 0")
    elif d > den_limit:
        return Fraction(n, d).limit_denominator(den_limit)
    else:
        return Fraction(n, d)


class PiResolution(namedtuple("PiResolution", ("width", "height"))):
    """
    A :func:`~collections.namedtuple` derivative which represents a resolution
    with a :attr:`width` and :attr:`height`.

    .. attribute:: width

        The width of the resolution in pixels

    .. attribute:: height

        The height of the resolution in pixels

    .. versionadded:: 1.11
    """

    __slots__ = ()  # workaround python issue #24931

    def pad(self, width=32, height=16):
        """
        Returns the resolution padded up to the nearest multiple of *width*
        and *height* which default to 32 and 16 respectively (the camera's
        native block size for most operations). For example:

        .. code-block:: pycon

            >>> PiResolution(1920, 1080).pad()
            PiResolution(width=1920, height=1088)
            >>> PiResolution(100, 100).pad(16, 16)
            PiResolution(width=128, height=112)
            >>> PiResolution(100, 100).pad(16, 16)
            PiResolution(width=112, height=112)
        """
        return PiResolution(
            width=((self.width + (width - 1)) // width) * width,
            height=((self.height + (height - 1)) // height) * height,
        )

    def transpose(self):
        """
        Returns the resolution with the width and height transposed. For
        example:

        .. code-block:: pycon

            >>> PiResolution(1920, 1080).transpose()
            PiResolution(width=1080, height=1920)
        """
        return PiResolution(self.height, self.width)

    def __str__(self):
        return "%dx%d" % (self.width, self.height)


class PiFramerateRange(namedtuple("PiFramerateRange", ("low", "high"))):
    """
    This class is a :func:`~collections.namedtuple` derivative used to store
    the low and high limits of a range of framerates. It is recommended that
    you access the information stored by this class by attribute rather than
    position (for example: ``camera.framerate_range.low`` rather than
    ``camera.framerate_range[0]``).

    .. attribute:: low

        The lowest framerate that the camera is permitted to use (inclusive).
        When the :attr:`~picamera.PiCamera.framerate_range` attribute is
        queried, this value will always be returned as a
        :class:`~fractions.Fraction`.

    .. attribute:: high

        The highest framerate that the camera is permitted to use (inclusive).
        When the :attr:`~picamera.PiCamera.framerate_range` attribute is
        queried, this value will always be returned as a
        :class:`~fractions.Fraction`.

    .. versionadded:: 1.13
    """

    __slots__ = ()  # workaround python issue #24931

    def __new__(cls, low, high):
        return super(PiFramerateRange, cls).__new__(
            cls, to_fraction(low), to_fraction(high)
        )

    def __str__(self):
        return "%s..%s" % (self.low, self.high)


class PiSensorMode(
    namedtuple(
        "PiSensorMode", ("resolution", "framerates", "video", "still", "full_fov")
    )
):
    """
    This class is a :func:`~collections.namedtuple` derivative used to store
    the attributes describing a camera sensor mode.

    .. attribute:: resolution

        A :class:`PiResolution` specifying the size of frames output by the
        camera in this mode.

    .. attribute:: framerates

        A :class:`PiFramerateRange` specifying the minimum and maximum
        framerates supported by this sensor mode. Typically the low value is
        exclusive and high value inclusive.

    .. attribute:: video

        A :class:`bool` indicating whether or not the mode is capable of
        recording video. Currently this is always ``True``.

    .. attribute:: still

        A :class:`bool` indicating whether the mode can be used for still
        captures (cases where a capture method is called with
        ``use_video_port`` set to ``False``).

    .. attribute:: full_fov

        A :class:`bool` indicating whether the full width of the sensor
        area is used to capture frames. This can be ``True`` even when the
        resolution is less than the camera's maximum resolution due to binning
        and skipping. See :ref:`camera_modes` for a diagram of the available
        fields of view.
    """

    __slots__ = ()  # workaround python issue #24931

    def __new__(cls, resolution, framerates, video=True, still=False, full_fov=True):
        return super(PiSensorMode, cls).__new__(
            cls,
            resolution
            if isinstance(resolution, PiResolution)
            else to_resolution(resolution),
            framerates
            if isinstance(framerates, PiFramerateRange)
            else PiFramerateRange(*framerates),
            video,
            still,
            full_fov,
        )


PiCameraFraction = Fraction


class MMALBaseComponent:
    def __init__(self):
        self.enabled = False


class MMALCamera(MMALBaseComponent):
    pass


class MMALCameraInfo(MMALBaseComponent):
    pass


class MMALComponent(MMALBaseComponent):
    pass


class MMALSplitter(MMALComponent):
    pass


class MMALResizer(MMALComponent):
    pass


class MMALISPResizer(MMALComponent):
    pass


class MMALEncoder(MMALComponent):
    pass


class MMALVideoEncoder(MMALEncoder):
    pass


class MMALImageEncoder(MMALEncoder):
    pass


class MMALDecoder(MMALComponent):
    pass


class MMALVideoDecoder(MMALDecoder):
    pass


class MMALImageDecoder(MMALDecoder):
    pass


class MMALRenderer(MMALComponent):
    pass


class MMALNullSink(MMALComponent):
    pass


class MMALControlPort:
    def __init__(self, port: int):
        self.value: int = port
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class MMALPort(MMALControlPort):
    pass


class MMALVideoPort(MMALPort):
    pass


class MMALSubPicturePort(MMALPort):
    pass


class MMALAudioPort(MMALPort):
    pass


# TODO
# class MMALPortParams(TODO):
# pass


class MMALBaseConnection:
    pass


class MMALConnection(MMALBaseConnection):
    pass


class MMALBuffer:
    pass


class MMALQueue:
    pass


class MMALPool:
    pass


class MMALPortPool(MMALPool):
    pass


# TODO
# class MMALPythonPort(TODO):
#     pass
class MMALPythonBaseComponent:
    pass


class MMALPythonComponent(MMALPythonBaseComponent):
    pass


class MMALPythonConnection(MMALBaseConnection):
    pass


class MMALPythonSource(MMALPythonBaseComponent):
    pass


class MMALPythonTarget(MMALPythonComponent):
    pass
