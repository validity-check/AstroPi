"""
Sources:
- Exif v2.2 spec:
    https://web.archive.org/web/20131019050323/http://www.exif.org/Exif2-2.PDF
- More concise descriptions of spec:
    https://web.archive.org/web/20230706014625/
    https://www.media.mit.edu/pia/Research/deepview/exif.html
- https://exiftool.org/TagNames/EXIF.html

The exif module uses https://plum-py.readthedocs.io/en/latest/ to mutate the EXIF tags
"""

import datetime
from collections import defaultdict

from PIL import Image

from astro_pi_replay.picamera.abstract_camera import PiCamera
from astro_pi_replay.picamera.exc import PiCameraValueError

DATETIME_STR_FORMAT: str = "%Y:%m:%d %H:%M:%S"

_IFD_TAGS_MAP: dict[str, int] = {
    "ImageWidth": 0x0100,
    "ImageLength": 0x0101,
    "BitsPerSample": 0x0102,
    "Compression": 0x0103,
    "PhotometricInterpretation": 0x0106,
    "ImageDescription": 0x010E,
    "Make": 0x010F,
    "Model": 0x0110,
    "StripOffsets": 0x0111,
    "Orientation": 0x0112,
    "SamplesPerPixel": 0x0115,
    "RowsPerString": 0x0116,  # sic (should be RowsPerStrip)
    "RowsPerStrip": 0x0116,
    "StripByteCounts": 0x0117,
    "Xresolution": 0x011A,  # sic
    "Yresolution": 0x011B,  # sic
    "PlanarConfiguration": 0x011C,
    "ResolutionUnit": 0x0128,
    "TransferFunction": 0x012D,
    "Software": 0x0131,
    "DateTime": 0x0132,  # often called ModifyDate
    "Artist": 0x013B,
    "WhitePoint": 0x013E,
    "PrimaryChromaticities": 0x013F,
    "JPEGInterchangeFormat": 0x0201,
    "JPEGInterchangeFormatLength": 0x0202,
    "YcbCrCoefficients": 0x0211,
    "YcbCrSubSampling": 0x0212,
    "YcbCrPositioning": 0x0213,
    "ReferenceBlackWhite": 0x0214,
    "Copyright": 0x8298,
}

_EXIF_TAGS_MAP: dict[str, int] = {
    "ExposureTime": 0x829A,
    "FNumber": 0x829D,
    "ExposureProgram": 0x8822,
    "SpectralSensitivity": 0x8824,
    "ISOSpeedRatings": 0x8827,
    "OECF": 0x8828,
    "ExifVersion": 0x9000,
    "DateTimeOriginal": 0x9003,
    "DateTimeDigitized": 0x9004,
    "ComponentsConfiguration": 0x9101,
    "CompressedBitsPerPixel": 0x9102,
    "ShutterSpeedValue": 0x9201,
    "ApertureValue": 0x9202,
    "BrightnessValue": 0x9203,
    "ExposureBiasValue": 0x9204,
    "MaxApertureValue": 0x9205,
    "SubjectDistance": 0x9206,
    "MeteringMode": 0x9207,
    "LightSource": 0x9208,
    "Flash": 0x9209,
    "FocalLength": 0x920A,
    "SubjectArea": 0x9214,
    "MakerNote": 0x927C,
    "UserComment": 0x9286,
    "SubSecTime": 0x9290,
    "SubSecTimeOriginal": 0x9291,
    "SubSecTimeDigitized": 0x9292,
    "FlashpixVersion": 0xA000,
    "ColorSpace": 0xA001,
    "PixelXDimension": 0xA002,
    "PixelYDimension": 0xA003,
    "RelatedSoundFile": 0xA004,
    "FlashEnergy": 0xA20B,
    "SpacialFrequencyResponse": 0xA20C,
    "FocalPlaneXResolution": 0xA20E,
    "FocalPlaneYResolution": 0xA20F,
    "FocalPlaneResolutionUnit": 0xA210,
    "SubjectLocation": 0xA214,
    "ExposureIndex": 0xA215,
    "SensingMethod": 0xA217,
    "FileSource": 0xA300,
    "SceneType": 0xA301,
    "CFAPattern": 0xA302,
    "CustomRendered": 0xA401,
    "ExposureMode": 0xA402,
    "WhiteBalance": 0xA403,
    "DigitalZoomRatio": 0xA404,
    "FocalLengthIn35mmFilm": 0xA405,
    "SceneCaptureType": 0xA406,
    "GainControl": 0xA407,
    "Contrast": 0xA408,
    "Saturation": 0xA409,
    "Sharpness": 0xA40A,
    "DeviceSettingDescription": 0xA40B,
    "SubjectDistanceRange": 0xA40C,
    "ImageUniqueID": 0xA420,
}

# https://exiftool.org/TagNames/GPS.html
_GPS_TAGS_MAP: dict[str, int] = {
    "GPSVersionID": 0x0000,
    "GPSLatitudeRef": 0x0001,
    "GPSLatitude": 0x0002,
    "GPSLongitudeRef": 0x0003,
    "GPSLongitude": 0x0004,
    "GPSAltitudeRef": 0x0005,
    "GPSAltitude": 0x0006,
    "GPSTimeStamp": 0x0007,
    "GPSSatellites": 0x0008,
    "GPSStatus": 0x0009,
    "GPSMeasureMode": 0x000A,
    "GPSDOP": 0x000B,
    "GPSSpeedRef": 0x000C,
    "GPSSpeed": 0x000D,
    "GPSTrackRef": 0x000E,
    "GPSTrack": 0x000F,
    "GPSImgDirectionRef": 0x0010,
    "GPSImgDirection": 0x0011,
    "GPSMapDatum": 0x0012,
    "GPSDestLatitudeRef": 0x0013,
    "GPSDestLatitude": 0x0014,
    "GPSDestLongitudeRef": 0x0015,
    "GPSDestLongitude": 0x0016,
    "GPSDestBearingRef": 0x0017,
    "GPSDestBearing": 0x0018,
    "GPSDestDistanceRef": 0x0019,
    "GPSDestDistance": 0x001A,
    "GPSProcessingMethod": 0x001B,
    "GPSAreaInformation": 0x001C,
    "GPSDateStamp": 0x001D,
    "GPSDifferential": 0x001E,
}

_EINT_TAGS_MAP: dict[str, int] = {
    "InteroperabilityIndex": 0x0001,
    "InteroperabilityVersion": 0x0002,
    "RelatedImageFileFormat": 0x1000,
    "RelatedImageWidth": 0x1001,
    "RelatedImageLength": 0x1002,
}

SUPPORTED_TAGS_MAP: dict[str, dict[str, int]] = {
    "IFD0": _IFD_TAGS_MAP,
    "IFD1": _IFD_TAGS_MAP,
    "EXIF": _EXIF_TAGS_MAP,
    "GPS": _GPS_TAGS_MAP,
    "EINT": _EINT_TAGS_MAP,
}

############
# FUNCTIONS
############


def get_tag_index(tag: str) -> int:
    """
    Returns the int index of the given tag, if it exists.
    Otherwise, raises a PiCameraValueError
    """
    split_tag: list[str] = tag.split(".")
    exception: PiCameraValueError = PiCameraValueError(f"Unsupported tag '{tag}")
    if len(split_tag) != 2:
        raise exception
    tag_map = SUPPORTED_TAGS_MAP[split_tag[0]]
    if tag_map is None:
        raise exception
    value = tag_map[split_tag[1]]
    if value is None:
        raise exception
    return value


# TODO refactor
def modify_exif_tags(
    current_exif_tags: Image.Exif, desired_exif_tags: dict[str, str]
) -> Image.Exif:
    """
    Merges the current_exif_tags with the desired_exif_tags, updating the
    datetime fields unless they are set in desired_exif_tags.
    Precedence is given to keys in the desired_exif_tags in case of a collision.
    """
    now: str = datetime.datetime.now().strftime(DATETIME_STR_FORMAT)
    tags_to_set: dict[str, str] = {
        "EXIF.DateTimeOriginal": now,
        "EXIF.DateTimeDigitized": now,
        "IFD0.DateTime": now,
    }
    if PiCamera._DEFAULT_EXIF_TAGS != desired_exif_tags:
        # merges, keeping keys from the right arg
        tags_to_set = tags_to_set | desired_exif_tags

    # ifd is (Image File Directory)
    grouped_by_ifd: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k, v in tags_to_set.items():
        mapped_k = k.split(".")[0]
        grouped_by_ifd[mapped_k].append((k, v))

    if len(grouped_by_ifd["IFD0"]) > 0:
        ifd0_ifd = current_exif_tags
        for tag, value in grouped_by_ifd["IFD0"]:
            ifd0_ifd[get_tag_index(tag)] = value
    if len(grouped_by_ifd["EXIF"]) > 0:
        exif_ifd = current_exif_tags.get_ifd(34665)  # ExifTags.IFD.Exif
        for tag, value in grouped_by_ifd["EXIF"]:
            exif_ifd[get_tag_index(tag)] = value

    if len(grouped_by_ifd["GPS"]) > 0:
        gps_ifd = current_exif_tags.get_ifd(34853)  # ExifTags.IFD.GPSInfo
        for gps_tag, value in grouped_by_ifd["GPS"]:
            gps_ifd[get_tag_index(gps_tag)] = value
    # TODO need to set the subsecond tags as well...
    return current_exif_tags
