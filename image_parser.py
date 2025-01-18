from exif import Image
from datetime import datetime
from math import radians

from consts import BASE_FOLDER

# class ImageSet:
#     def __init__(self, prev_image=None, curr_image=None):
#         self.prev_image = prev_image
#         self.curr_image = curr_image
    
#     def add_image(self, url: str):
#         absolute_path = BASE_FOLDER / url
#         if self.prev_image:
#             self.prev_image, self.curr_image = self.curr_image, absolute_path
#         else:
#             self.prev_image = absolute_path

def extract_image_data(absolute_path: str, return_coords: bool = True, return_time: bool = True):
    return_values = []
    with open(absolute_path, 'rb') as image_file:
        img = Image(image_file)

        if return_coords:
            longitude = img.get("gps_longitude")
            latitude = img.get("gps_latitude")
            coords = [latitude, longitude]
            return_values.append([radians(_) for _ in coords])

        if return_time:
            time_str = img.get("datetime_original")
            time = datetime.strptime(time_str, '%s')
            return_values.append(time)
    
    return return_values
