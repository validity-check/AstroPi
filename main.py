from datetime import datetime, timedelta
from time import sleep
from exif import Image
from math import radians
from pathlib import Path

from picamzero import Camera # type: ignore
from astro_pi_orbit import ISS
from sklearn.metrics.pairwise import haversine_distances


# Constant values
BASE_FOLDER = Path(__file__).parent.resolve()
EARTH_RADIUS = 6371000/1000

def main():
    # Create a variable to store the start time
    start_time = datetime.now()
    # Create a variable to store the current time
    # (these will be almost the same at the start)
    now_time = datetime.now()

    cam = Camera()
    iss = ISS()
    prev_image = None

    images_taken = 0
    total_speed = 0

    # Core timed 10 minute loop
    while now_time < start_time +  timedelta(minutes=10) and images_taken <= 40:
        file_name = BASE_FOLDER / ("image%s.jpg" % str(images_taken))
        cam.take_photo(file_name, gps_coordinates=get_gps_coordinates(iss))
        images_taken +=1

        image_data = extract_image_data(file_name)

        if prev_image:
            print(image_data[0], prev_image[0])
            distance = haversine_distances([image_data[0], prev_image[0]])[0][1] * EARTH_RADIUS
            time = timedelta.total_seconds(image_data[1] - prev_image[1])
            speed = distance/time
            total_speed += speed
            print(total_speed/(images_taken-1))
            write_data(total_speed, images_taken)

        prev_image = image_data
        sleep(1)
        now_time = datetime.now()

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

def write_data(total_speed, images_taken):
    with open("result.txt", "w", buffering=1) as results:
        result=total_speed/(images_taken-1)
        results.write("{:.4f}".format(result))

def get_gps_coordinates(iss):
    """
    Returns a tuple of latitude and longitude coordinates expressed
    in signed degrees minutes seconds.
    """
    point = iss.coordinates()
    return (point.latitude.signed_dms(), point.longitude.signed_dms())

if __name__=="__main__":
    main()