from datetime import datetime, timedelta
from time import sleep
from exif import Image
from math import radians
import os

from picamzero import Camera # type: ignore
# from astro_pi_orbit import ISS
from orbit import ISS

from sklearn.metrics.pairwise import haversine_distances


# Constant values
BASE_FOLDER = os.path.abspath(os.getcwd())
EARTH_RADIUS = 6371

def main():
    # Create a variable to store the start time
    start_time = datetime.now()
    # Create a variable to store the current time
    # (these will be almost the same at the start)
    now_time = datetime.now()

    cam = Camera()
    iss = ISS()

    # Store image previously taken to compare to current image 
    prev_image = None

    images_taken = 0
    total_speed = 0

    # Core timed 10 minute loop
    # Ensures program operates within time and photo limits
    while now_time < start_time +  timedelta(minutes=10) and images_taken <= 41:
        # file_name = BASE_FOLDER + ("\\image%s.jpg" % str(images_taken))
        file_name = "\\image%s.jpg" % str(images_taken)
        print(file_name)
        # Assign GPS coordinates of ISS to image we are taking
        cam.take_photo(file_name, gps_coordinates=get_gps_coordinates(iss))
        images_taken +=1

        image_data = extract_image_data(file_name)

        if prev_image:
            print(image_data[0], prev_image[0])
            # Multiply by earth radius to convert to kilometers
            distance = haversine_distances([image_data[0], prev_image[0]])[0][1] * EARTH_RADIUS
            time = timedelta.total_seconds(image_data[1] - prev_image[1])
            speed = distance/time
            total_speed += speed
            print(total_speed/(images_taken-1))
            write_data(total_speed, images_taken)

        prev_image = image_data

        # Waiting a second should improve accuracy by taking more spread out readings
        sleep(1)
        now_time = datetime.now()

def extract_image_data(absolute_path: str, return_coords: bool = True, return_time: bool = True):
    """
    Given path to an image, return coords and time (if each is specified to be returned)
    """
    return_values = []
    with open(absolute_path, 'rb') as image_file:
        img = Image(image_file)

        if return_coords:
            longitude = img.get("gps_longitude")
            latitude = img.get("gps_latitude")
            coords = [latitude, longitude]
            # Coords were stored in dms. This needs to be converted to radians for the haversine formula
            return_values.append([dms_to_rad(_) for _ in coords])
            print(return_values)

        if return_time:
            time_str = img.get("datetime_original")
            # Format time in a way that can be subtracted from another timestamp of the same format
            time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            return_values.append(time)
    
    return return_values

def dms_to_rad(dms):
    """
    Converts signed degrees minutes seconds to radians.
    """
    (d, m, s) = dms
    # Formula dms-> dd: d + m/60 + s/3600
    # The math module converts to radians
    rad = radians(d + float(m)/60 + float(s)/3600)
    return rad

def write_data(total_speed: int, images_taken: int):
    """
    Formats result to 5 significant figures
    """
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