import collections
from enum import Enum
from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite, WriteableBuffer


# Type synonyms
RGBC = tuple[int, int, int, int]
RGB = tuple[int, int, int]
RollPitchYawDict = dict[str, float]
XYZDict = dict[str, float]
InputEvent = collections.namedtuple("InputEvent", ("timestamp", "direction", "action"))
XYWH = tuple[float, float, float, float]
UV = tuple[int, int]


DEFAULT_ROLL_PITCH_YAW_DICT = {"roll": float(), "pitch": float(), "yaw": float()}
DEFAULT_RGB_TUPLE = (int(), int(), int())
DEFAULT_RGBC_TUPLE = (int(), int(), int(), int())
DEFAULT_X_Y_Z_DICT = {"x": float(), "y": float(), "z": float()}
DEFAULT_XYWH = (0.0, 0.0, 1.0, 1.0)
DEFAULT_CALLABLE = (
    lambda x: x
)  # TODO could use inspect module to check type annotations at runtime


class ExecutionMode(str, Enum):
    REPLAY = ("REPLAY",)
    LIVE = "LIVE"


# Picamera types
# output: Union[str, BinaryIO, np.ndarray],

# class SupportsWrite(Protocol):
#     __slots__ = ()
#     write: Callable[[bytes],None]

# class SupportsRead(Protocol):
#     __slots__ = ()
#     read: Callable[[None],bytes]


# https://docs.python.org/3/c-api/buffer.html

# format: Optional[str] = None
# see: https://peps.python.org/pep-0688/#python-level-buffer-protocol
IO_TYPE: TypeAlias = Union[
    bytes, str, "SupportsWrite", "SupportsRead", "WriteableBuffer"
]
