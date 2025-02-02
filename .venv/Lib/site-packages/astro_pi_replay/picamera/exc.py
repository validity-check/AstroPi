# TODO just copy everything from the original.
class PiCameraError(Exception):
    """
    Base class for PiCamera errors.
    """


class PiCameraRuntimeError(RuntimeError):
    """
    Raised when an invalid sequence of operations is attempted with a
    :class:`PiCamera` object.
    """


class PiCameraValueError(PiCameraError, ValueError):
    """
    Raised when an invalid value is fed to a :class:`~PiCamera` object.
    """


class PiCameraWarning(Warning):
    """
    Base class for PiCamera warnings.
    """


class PiCameraDeprecated(PiCameraWarning, DeprecationWarning):
    """
    Raised when deprecated functionality in picamera is used.
    """


class PiCameraFallback(PiCameraWarning, RuntimeWarning):
    """
    Raised when picamera has to fallback on old functionality.
    """


class PiCameraNotRecording(PiCameraRuntimeError):
    """
    Raised when :meth:`~PiCamera.stop_recording` or
    :meth:`~PiCamera.split_recording` are called against a port which has no
    recording active.
    """


class PiCameraAlreadyRecording(PiCameraRuntimeError):
    """
    Raised when :meth:`~PiCamera.start_recording` or
    :meth:`~PiCamera.record_sequence` are called against a port which already
    has an active recording.
    """


class PiCameraMMALError(PiCameraError):
    """
    Raised when an MMAL operation fails for whatever reason.
    """


class PiCameraClosed(PiCameraRuntimeError):
    """
    Raised when a method is called on a camera which has already been closed.
    """
