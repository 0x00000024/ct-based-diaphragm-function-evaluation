import os
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd
from slice import handle_lung_slice
from colorama import Fore, init
import settings
from src.utils.image_utils import jpg2gif
from timeit import default_timer as timer
import plotly.express as px

start_time = timer()
images_dirname = '../images/'
original_images_dirname = 'original/'
images_basename = [
    f for f in sorted(listdir(settings.original_images_dirname))
    if isfile(join(settings.original_images_dirname, f))
]

init(autoreset=True)

color_image_example = cv2.imread(settings.original_images_dirname +
                                 settings.debug_image_filename)
settings.image_height = color_image_example.shape[0]
settings.image_width = color_image_example.shape[1]

for image_basename in images_basename:

    if settings.debugging_mode:
        if image_basename != settings.debug_image_filename:
            continue

    print(Fore.BLUE + 'Current slice image filename: ',
          settings.category + '/' + image_basename)
    handle_lung_slice(image_basename)

print(Fore.BLUE + 'Generate 3D visualization of diaphragm points...\n')
df = pd.DataFrame(
    settings.diaphragm_points,
    columns=['x_value', 'y_value', 'slice_interval', 'image_number'])
fig = px.scatter_3d(df,
                    x='x_value',
                    y='y_value',
                    z='slice_interval',
                    range_x=[0, 512],
                    range_y=[0, 512],
                    color='image_number')
fig.update_layout(scene_camera=settings.camera,
                  title=settings.patient_id + '/' + settings.category)
fig.data[0].marker.symbol = 'circle'
fig.data[0].marker.size = 2
fig.write_html(settings.processed_images_dirname + settings.html_filename)
# Save the diaphragm points to a CSV file
df.to_csv(settings.processed_images_dirname + settings.csv_filename)

end_time = timer()
elapsed_time = end_time - start_time
print(
    Fore.RED +
    f'The total time to execute the program this time is {elapsed_time} seconds\n'
)

if not settings.debugging_mode:
    print(Fore.BLUE + 'Generating GIF image to',
          settings.processed_images_dirname + settings.gif_filename + '\n')
    jpg2gif(image_dirname=settings.processed_images_dirname,
            output_filename=settings.gif_filename)

    print(Fore.BLUE + 'Uploading demo files to server...\n')
    os.system(settings.upload_cmd)
    print(Fore.BLUE + '2d demo:', settings.gif_url + '\n')
    print(Fore.BLUE + '3d demo:', settings.html_url + '\n')
    print(Fore.BLUE + 'points.csv:', settings.csv_url + '\n')

    print(Fore.GREEN + 'DONE!')
