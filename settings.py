import numpy as np
from datetime import datetime

#############
# Debugging
#############
# debugging_mode = True
debugging_mode = False
debug_image_filename = 'IM-0001-0057.dcm'
debug_lung_position = 'left'
# debug_lung_position = 'right'

#############
# Image
#############
images_dirname = '/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/images/'
# patient_id = '10602422'
category = None
# category = 'in'
# category = 'ex'
# category = 'mix'
date = datetime.today().strftime('%Y%m%d-%H%M%S')[2:]
original_images_dirname = None
processed_images_dirname = None

image_height = None
image_width = None

z_value = None

# Use src/utils/hsv_thresholder.py to get the required upper and lower bounds
# No need to reshape, it is used to help PyCharm perform code static analysis.
# left_lung_lower_boundary = np.array([81, 0, 111], dtype='uint8').reshape(1, 3)
# left_lung_upper_boundary = np.array([103, 111, 195],
#                                     dtype='uint8').reshape(1, 3)
# right_lung_lower_boundary = np.array([6, 8, 84], dtype='uint8').reshape(1, 3)
# right_lung_upper_boundary = np.array([48, 155, 213],
#                                      dtype='uint8').reshape(1, 3)

# Segment entire lung
left_lung_lower_boundary = np.array([4, 0, 41], dtype='uint8').reshape(1, 3)
left_lung_upper_boundary = np.array([179, 255, 255],
                                    dtype='uint8').reshape(1, 3)
right_lung_lower_boundary = np.array([4, 0, 41], dtype='uint8').reshape(1, 3)
right_lung_upper_boundary = np.array([179, 255, 255],
                                     dtype='uint8').reshape(1, 3)

#############
# Web
#############
url_origin = 'https://ct.eny.li'
gif_filename = '2d.gif'
html_filename = '3d.html'
csv_filename = 'points.csv'
html_url = None
upload_cmd = None

#############
# Diaphragm
#############
diaphragm_points = None
# unit: cm
row_spacing = 0.3 * 10**-1
col_spacing = 0.28 * 10**-1
thickness = 1 * 10**-1

#############
# Color
#############
color_black = (0, 0, 0)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_red = (0, 0, 255)

# Blue and green
color_cyan = (255, 255, 0)
color_azure = (255, 127, 0)

# Green and red
color_yellow = (0, 255, 255)

# Blue and red
color_magenta = (255, 0, 255)
color_violet = (255, 0, 127)

color_purple = (173, 13, 106)

#############
# 3d vis
#############
camera = dict(up=dict(x=0, y=1, z=0),
              center=dict(x=0, y=0, z=0),
              eye=dict(x=0, y=1.25, z=-1.5))

#############
# Heart
#############
selected_x_value = None
selected_y_value = None
