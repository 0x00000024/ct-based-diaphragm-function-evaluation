import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List, Tuple
import cv2
import pandas as pd
from slice import handle_lung_slice
from colorama import Fore, init
import settings
from src.merger import merger
from src.utils.image_utils import jpg2gif
from timeit import default_timer as timer
import plotly.express as px

from src.volume import get_volume


def in_or_ex_analysis(patient_id: str, category: str,
                      images_basename: List[str]) -> Tuple[str, str, str]:
    settings.category = 'in' if category == 'in' else 'ex'
    settings.original_images_dirname = settings.images_dirname + 'original/' + patient_id + '/' + category + '/'
    settings.processed_images_dirname = settings.images_dirname + 'processed/' + patient_id + '/' + category + '/' + \
                                        settings.date + '/'
    Path(settings.processed_images_dirname).mkdir(parents=True, exist_ok=True)

    color_image_example = cv2.imread(settings.original_images_dirname +
                                     images_basename[0])
    settings.image_height = color_image_example.shape[0]
    settings.image_width = color_image_example.shape[1]

    for image_basename in images_basename:
        if settings.debugging_mode:
            if image_basename != settings.debug_image_filename:
                continue

        print(Fore.BLUE + 'Current slice image filename: ',
              category + '/' + image_basename)
        handle_lung_slice(image_basename)

    url_path = '/' + patient_id + '/' + category + '/' + settings.date + '/'
    gif_url = settings.url_origin + url_path + settings.gif_filename
    html_url = settings.url_origin + url_path + settings.html_filename
    csv_url = settings.url_origin + url_path + settings.csv_filename
    upload_cmd = 'cd ' + settings.images_dirname + 'processed && ls -al && find ' + patient_id + '/' + category + '/' +\
                 settings.date + '/' + ' -name "*.gif" -o -name "*.html" -o -name "*.csv" | xargs tar cfP - |' +\
                 ' ssh root@ct.eny.li tar xfP - -C /var/www/html && rm -rf * && echo'

    print(Fore.BLUE + 'Generate 3D visualization of diaphragm points...\n')
    df_3d = pd.DataFrame(
        settings.diaphragm_points,
        columns=['x_value', 'y_value', 'slice_interval', 'image_number'])
    fig = px.scatter_3d(df_3d,
                        x='x_value',
                        y='y_value',
                        z='slice_interval',
                        range_x=[0, settings.image_height],
                        range_y=[0, settings.image_width],
                        color='image_number')
    fig.update_layout(scene_camera=settings.camera,
                      title=patient_id + '/' + category)
    fig.data[0].marker.symbol = 'circle'
    fig.data[0].marker.size = 2
    fig.write_html(settings.processed_images_dirname + settings.html_filename)
    # Save the diaphragm points to a CSV file
    df_3d.to_csv(settings.processed_images_dirname + settings.csv_filename)

    if not settings.debugging_mode:
        print(Fore.BLUE + 'Generating GIF image to',
              settings.processed_images_dirname + settings.gif_filename + '\n')
        jpg2gif(image_dirname=settings.processed_images_dirname,
                output_filename=settings.gif_filename)

        print(Fore.BLUE + 'Uploading demo files to server...\n')
        os.system(upload_cmd)
        print(Fore.BLUE + '2d demo:', gif_url + '\n')
        print(Fore.BLUE + '3d demo:', html_url + '\n')
        print(Fore.BLUE + 'points.csv:', csv_url + '\n')

    return gif_url, html_url, csv_url


df = pd.read_csv('test/input.csv', dtype={'patient_id': str, 'start_image_number': int, 'stop_image_number': int,
                                          'in_2d': str, 'in_3d': str, 'in_csv': str, 'ex_2d': str, 'ex_3d': str,
                                          'ex_csv': str, 'merged_3d': 'str', 'volume (cm^3)': float,
                                          'longest dist (cm)': float, 'shortest dist (cm)': float})

for index, row in df.iterrows():
    patient_id = row['patient_id']
    start_image_number = row['start_image_number']
    stop_image_number = row['stop_image_number']
    print('patient_id', patient_id)
    print('start_image_number', start_image_number)
    print('stop_image_number', stop_image_number)

    start_time = timer()

    images_basename = []
    for i in range(start_image_number, stop_image_number + 1, 1):
        images_basename.append(f"IM-0001-{i:04d}.jpg")

    settings.initial_slice_interval = 2
    settings.diaphragm_points = None
    gif_url, html_url, csv_url = in_or_ex_analysis(
        patient_id=patient_id, category='in', images_basename=images_basename)
    df.at[index, 'in_2d'] = gif_url
    df.at[index, 'in_3d'] = html_url
    df.at[index, 'in_csv'] = csv_url
    in_csv = csv_url

    settings.initial_slice_interval = 2
    settings.diaphragm_points = None
    gif_url, html_url, csv_url = in_or_ex_analysis(
        patient_id=patient_id, category='ex', images_basename=images_basename)
    df.at[index, 'ex_2d'] = gif_url
    df.at[index, 'ex_3d'] = html_url
    df.at[index, 'ex_csv'] = csv_url
    ex_csv = csv_url

    #############
    # Merger
    #############
    category = 'mix'
    settings.processed_images_dirname = settings.images_dirname + 'processed/' + patient_id + '/' + category + '/' + \
                                        settings.date + '/'
    Path(settings.processed_images_dirname).mkdir(parents=True, exist_ok=True)

    merger(in_csv, ex_csv, patient_id, category)

    url_path = '/' + patient_id + '/' + category + '/' + settings.date + '/'
    html_url = settings.url_origin + url_path + settings.html_filename
    print(Fore.BLUE + '3d demo:', html_url + '\n')
    df.at[index, 'merged_3d'] = html_url

    #############
    # Volume
    #############
    volume, longest_dist, shortest_dist = get_volume(in_csv, ex_csv)

    df.at[index, 'volume (cm^3)'] = volume
    df.at[index, 'longest dist (cm)'] = longest_dist
    df.at[index, 'shortest dist (cm)'] = shortest_dist

    end_time = timer()
    elapsed_time = end_time - start_time
    print(
        Fore.RED +
        f'The total time to process the {patient_id} is {elapsed_time} seconds\n'
    )

    df.to_csv('test/output.csv')
