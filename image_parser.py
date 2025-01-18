from exif import Image
from datetime import datetime
from math import radians

def extract_image_data(absolute_path: str, return_coords: bool = True, return_time: bool = True):
    return_values = []
    with open(absolute_path, 'rb') as image_file:
        img = Image(image_file)

        if return_coords:
            longitude = img.get("gps_longitude")
            latitude = img.get("gps_latitude")
            coords = [latitude, longitude]
            return_values.append([radians(dms_to_dd(_)) for _ in coords])
            print(return_values)

        if return_time:
            time_str = img.get("datetime_original")
            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            return_values.append(time)
    
    return return_values

def dms_to_dd(dms):
    """
    Converts signed degrees minutes seconds to decimal degrees.
    """
    (d, m, s) = dms
    dd = d + float(m)/60 + float(s)/3600
    return dd