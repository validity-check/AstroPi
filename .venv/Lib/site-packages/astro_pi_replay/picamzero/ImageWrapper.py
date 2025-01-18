from pathlib import Path
from typing import Union, Iterable, Optional

from PIL import Image
import numpy as np


class ImageWrapper:
    """
    Wrapper class for PIL.Image and
    PIL.Image.Image to assist with
    keeping exif data when converting
    to/from numpy arrays
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.image: Image.Image = Image.open(path)
        self.exif: Image.Exif = self.image.getexif()

    @staticmethod
    def open(path: Union[str,Path]) -> "ImageWrapper":
        return ImageWrapper(path)

    ####################
    # instance methods #
    ####################

    @property
    def size(self) -> Iterable[int]:
        return self.image.size

    @property
    def mode(self) -> str:
        return self.image.mode

    def convert(self, *args, **kwargs) -> "ImageWrapper":
        self.image = self.image.convert(*args, **kwargs)
        return self

    def resize(self, *args, **kwargs) -> "ImageWrapper":
        self.image = self.image.resize(*args,**kwargs)
        return self

    # this would normally be a static method
    # but we keep it an instance method to
    # keep the exif data
    def fromarray(self, array) -> "ImageWrapper":
        self.image = Image.fromarray(array)
        return self

    def toarray(self) -> np.ndarray:
        return np.array(self.image)

    def getexif(self) -> Image.Exif:
        return self.exif

    def save(self, fp, format=None, **params) -> None:
        if "exif" not in params:
            params["exif"] = self.exif
        self.image.save(fp, format, **params)

    def paste(self, im: Union["ImageWrapper",Image.Image],
              box=None, mask: Optional["ImageWrapper"]=None) -> None:
        _mask: Optional[Image.Image] = None if mask is None \
                else mask.image
        if isinstance(im, ImageWrapper):
            return self.image.paste(im.image, box, _mask)
        return self.image.paste(im, box, _mask)

