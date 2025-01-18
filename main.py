from datetime import datetime, timedelta
import time

from picamzero import Camera # type: ignore
from astro_pi_orbit import ISS
from sklearn.metrics.pairwise import haversine_distances

import image_parser
import consts

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
        file_name = consts.BASE_FOLDER / ("image%s.jpg" % str(images_taken))
        cam.take_photo(file_name, gps_coordinates=get_gps_coordinates(iss))
        images_taken +=1

        image_data = image_parser.extract_image_data(file_name)

        if prev_image:
            print(image_data[0], prev_image[0])
            distance = haversine_distances([image_data[0], prev_image[0]]) * consts.EARTH_RADIUS
            time = timedelta.total_seconds(image_data[1] - prev_image[1])
            speed = distance/time
            total_speed += speed
            print(total_speed/(images_taken-1))

        prev_image = image_data
        time.sleep(1)
        now_time = datetime.now()

def write_data(total_velocity, images_taken):
    with open("results.txt", "w", buffering=1) as results:
        result=total_velocity/(images_taken-1)
        results.write(result)

def get_gps_coordinates(iss):
    """
    Returns a tuple of latitude and longitude coordinates expressed
    in signed degrees minutes seconds.
    """
    point = iss.coordinates()
    return (point.latitude.signed_dms(), point.longitude.signed_dms())

if __name__=="__main__":
    main()