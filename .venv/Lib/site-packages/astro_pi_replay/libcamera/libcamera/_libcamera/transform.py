from typing import Literal, Union


def maybe_error():
    # Only throw error if configured to
    if True:
        raise RuntimeError("Not implemented")


class Transform:
    """
    Derived from:
        * src/py/py_transform.cpp
        * src/libcamera/transform.cpp
        * include/libcamera/transform.h
    """

    def __init__(
        self,
        rotation: int = 0,
        vflip: Union[bool, Literal[1], Literal[0]] = False,
        hflip: Union[bool, Literal[1], Literal[0]] = False,
        transpose: bool = False,
    ):
        self.rotation: int = rotation
        self.vflip: bool = bool(vflip)
        self.hflip: bool = bool(hflip)
        self.transpose: bool = transpose

    def __hash__(self) -> int:
        return hash((self.rotation, self.vflip, self.hflip, self.transpose))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return False
        return (
            self.rotation == other.rotation
            and self.vflip == other.vflip
            and self.hflip == other.hflip
            and self.transpose == other.transpose
        )

    def __repr__(self) -> str:
        if not self.vflip and not self.hflip:
            sub = "identity"
        elif self.vflip and self.hflip:
            sub = "hvflip"
        elif self.hflip:
            sub = "hflip"
        elif self.vflip:
            sub = "vflip"
        else:
            sub = ""

        classname: str = "libcamera.Transform"
        if len(sub) > 0:
            return f"<{classname} '{sub}'>"
        return f"<{classname} >"

    def compose(self):
        maybe_error()

    def invert(self):
        maybe_error()

    def inverse(self):
        maybe_error()
