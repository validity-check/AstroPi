import enum


# Derived from src/libcamera/control_ids_core.yaml
class AwbModeEnum(enum.Enum):
    Auto = 0
    Incandescent = 1
    Tungsten = 2
    Fluorescent = 3
    Indoor = 4
    Daylight = 5
    Cloudy = 6
    Custom = 7
