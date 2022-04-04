import os
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from slice import handle_lung_slice
from colorama import Fore
import settings
from src.max_height_width import get_in_or_ex_left_or_right_max_height_width
from src.merger import merger
from src.utils.image_utils import jpg2gif
import pydicom as dicom
from timeit import default_timer as timer
import plotly.express as px
from src.area import get_area


def in_or_ex_analysis(patient_id: str, category: str,
                      images_basename: List[str]) -> None:
    settings.category = 'in' if category == 'in' else 'ex'
    settings.original_images_dirname = settings.images_dirname + 'original/' + patient_id + '/' + category + '/'
    settings.processed_images_dirname = settings.images_dirname + 'processed/' + patient_id + '/' + category + '/' + \
                                        settings.date + '/'
    Path(settings.processed_images_dirname).mkdir(parents=True, exist_ok=True)

    image_path = settings.original_images_dirname + images_basename[0]
    dicom_file_dataset = dicom.dcmread(image_path)
    # settings.row_spacing = float(dicom_file_dataset[0x28,
    #                                                 0x30].value[0]) * 10**-1
    # settings.col_spacing = float(dicom_file_dataset[0x28,
    #                                                 0x30].value[1]) * 10**-1

    if category == 'in':
        settings.in_row_spacing = float(
            dicom_file_dataset[0x28, 0x30].value[0]) * 10**-1
        settings.in_col_spacing = float(
            dicom_file_dataset[0x28, 0x30].value[1]) * 10**-1
    if category == 'ex':
        settings.ex_row_spacing = float(
            dicom_file_dataset[0x28, 0x30].value[0]) * 10**-1
        settings.ex_col_spacing = float(
            dicom_file_dataset[0x28, 0x30].value[1]) * 10**-1

    settings.image_height = int(dicom_file_dataset[0x28, 0x10].value)
    settings.image_width = int(dicom_file_dataset[0x28, 0x11].value)

    # for image_basename in images_basename:
    #     if settings.debugging_mode:
    #         if image_basename != settings.debug_image_filename:
    #             continue
    #
    #     print(Fore.BLUE + 'Current slice image filename: ',
    #           category + '/' + image_basename)
    #     handle_lung_slice(image_basename)
    #
    # url_path = '/' + patient_id + '/' + category + '/' + settings.date + '/'
    # gif_url = settings.url_origin + url_path + settings.gif_filename
    # html_url = settings.url_origin + url_path + settings.html_filename
    # csv_url = settings.url_origin + url_path + settings.csv_filename
    # upload_cmd = 'cd ' + settings.images_dirname + 'processed && ls -al && find ' + patient_id + '/' + category + '/' + \
    #              settings.date + '/' + ' -name "*.gif" -o -name "*.html" -o -name "*.csv" | xargs tar cfP - |' + \
    #              ' ssh root@ct.eny.li tar xfP - -C /var/www/html && rm -rf * && echo'
    #
    # print(Fore.BLUE + 'Generate 3D visualization of diaphragm points...\n')
    # df_3d = pd.DataFrame(
    #     settings.diaphragm_points,
    #     columns=['x_value', 'y_value', 'z_value', 'image_number', 'category'])
    # fig = px.scatter_3d(df_3d,
    #                     x='x_value',
    #                     y='y_value',
    #                     z='z_value',
    #                     range_x=[0, settings.image_height],
    #                     range_y=[0, settings.image_width],
    #                     color='category')
    # fig.update_layout(scene_camera=settings.camera,
    #                   title=patient_id + '/' + category)
    # fig.data[0].marker.symbol = 'circle'
    # fig.data[0].marker.size = 2
    # fig.write_html(settings.processed_images_dirname + settings.html_filename)
    # # Save the diaphragm points to a CSV file
    # df_3d.to_csv(settings.processed_images_dirname + settings.csv_filename)
    #
    # if not settings.debugging_mode:
    #     print(Fore.BLUE + 'Generating GIF image to',
    #           settings.processed_images_dirname + settings.gif_filename + '\n')
    #     jpg2gif(image_dirname=settings.processed_images_dirname,
    #             output_filename=settings.gif_filename)
    #
    #     print(Fore.BLUE + 'Uploading demo files to server...\n')
    #     os.system(upload_cmd)
    #     print(Fore.BLUE + '2d demo:', gif_url + '\n')
    #     print(Fore.BLUE + '3d demo:', html_url + '\n')
    #     print(Fore.BLUE + 'points.csv:', csv_url + '\n')
    #
    # return gif_url, html_url, csv_url


def main() -> None:
    df = pd.read_csv('test/input.csv',
                     dtype={
                         'patient_id': str,
                         'in_csv': str,
                         'ex_csv': str,
                         'row_spacing (cm)': float,
                         'col_spacing (cm)': float,
                         'in_left_lung_top_y_value': int,
                         'in_left_lung_bottom_y_value': int,
                         'in_left_lung_max_height (cm)': float,
                         'in_left_lung_max_height_image_index': int,
                         'in_left_lung_left_x_value': int,
                         'in_left_lung_right_x_value': int,
                         'in_left_lung_max_width (cm)': float,
                         'in_left_lung_max_width_image_index': int,
                         'in_right_lung_top_y_value': int,
                         'in_right_lung_bottom_y_value': int,
                         'in_right_lung_max_height (cm)': float,
                         'in_right_lung_max_height_image_index': int,
                         'in_right_lung_left_x_value': int,
                         'in_right_lung_right_x_value': int,
                         'in_right_lung_max_width (cm)': float,
                         'in_right_lung_max_width_image_index': int,
                         'ex_left_lung_top_y_value': int,
                         'ex_left_lung_bottom_y_value': int,
                         'ex_left_lung_max_height (cm)': float,
                         'ex_left_lung_max_height_image_index': int,
                         'ex_left_lung_left_x_value': int,
                         'ex_left_lung_right_x_value': int,
                         'ex_left_lung_max_width (cm)': float,
                         'ex_left_lung_max_width_image_index': int,
                         'ex_right_lung_top_y_value': int,
                         'ex_right_lung_bottom_y_value': int,
                         'ex_right_lung_max_height (cm)': float,
                         'ex_right_lung_max_height_image_index': int,
                         'ex_right_lung_left_x_value': int,
                         'ex_right_lung_right_x_value': int,
                         'ex_right_lung_max_width (cm)': float,
                         'ex_right_lung_max_width_image_index': int,
                     })

    for index, row in df.iterrows():
        patient_id = row['patient_id']
        # start_image_number = row['start_image_number']
        # stop_image_number = row['stop_image_number']
        settings.row_spacing = row['row_spacing (cm)']
        settings.col_spacing = row['col_spacing (cm)']
        print('patient_id', patient_id)
        # print('start_image_number', start_image_number)
        # print('stop_image_number', stop_image_number)

        start_time = timer()

        # images_basename = []
        # for i in range(start_image_number, stop_image_number + 1, 1):
        #     images_basename.append(f"IM-0001-{i:04d}.dcm")
        #
        # settings.diaphragm_points = None
        # in_or_ex_analysis(patient_id=patient_id, category='in', images_basename=images_basename)
        #
        # settings.diaphragm_points = None
        # in_or_ex_analysis( patient_id=patient_id, category='ex', images_basename=images_basename)

        #############
        # Merger
        #############
        # category = 'mix'
        # settings.processed_images_dirname = settings.images_dirname + 'processed/' + patient_id + '/' + category + '/' + \
        #                                     settings.date + '/'
        # Path(settings.processed_images_dirname).mkdir(parents=True,
        #                                               exist_ok=True)
        #
        # merger(in_csv, ex_csv, patient_id, category)
        #
        # url_path = '/' + patient_id + '/' + category + '/' + settings.date + '/'
        # html_url = settings.url_origin + url_path + settings.html_filename
        # print(Fore.BLUE + '3d demo:', html_url + '\n')
        # df.at[index, 'merged_3d'] = html_url

        #############
        # Area
        #############
        # in_left_area, in_right_area, ex_left_area, ex_right_area = get_area(
        #     in_csv, ex_csv)
        #
        # df.at[index, 'in_left_area (cm^2)'] = in_left_area
        # df.at[index, 'in_right_area (cm^2)'] = in_right_area
        # df.at[index, 'ex_left_area (cm^2)'] = ex_left_area
        # df.at[index, 'ex_right_area (cm^2)'] = ex_right_area
        #
        # end_time = timer()
        # elapsed_time = end_time - start_time
        # print(
        #     Fore.RED +
        #     f'The total time to process the {patient_id} is {elapsed_time} seconds\n'
        # )

        #############
        # Max height and width
        #############
        in_csv = row['in_csv']
        ex_csv = row['ex_csv']
        print('first group')
        in_left_lung_top_y_value, in_left_lung_bottom_y_value, in_left_lung_max_height, in_left_lung_max_height_image_index, in_left_lung_left_x_value, in_left_lung_right_x_value, in_left_lung_max_width, in_left_lung_max_width_image_index = get_in_or_ex_left_or_right_max_height_width(
            position='left', csv_url=in_csv)
        print('second group')
        in_right_lung_top_y_value, in_right_lung_bottom_y_value, in_right_lung_max_height, in_right_lung_max_height_image_index, in_right_lung_left_x_value, in_right_lung_right_x_value, in_right_lung_max_width, in_right_lung_max_width_image_index = get_in_or_ex_left_or_right_max_height_width(
            position='right', csv_url=in_csv)
        print('third group')
        ex_left_lung_top_y_value, ex_left_lung_bottom_y_value, ex_left_lung_max_height, ex_left_lung_max_height_image_index, ex_left_lung_left_x_value, ex_left_lung_right_x_value, ex_left_lung_max_width, ex_left_lung_max_width_image_index = get_in_or_ex_left_or_right_max_height_width(
            position='left', csv_url=ex_csv)
        print('forth group')
        ex_right_lung_top_y_value, ex_right_lung_bottom_y_value, ex_right_lung_max_height, ex_right_lung_max_height_image_index, ex_right_lung_left_x_value, ex_right_lung_right_x_value, ex_right_lung_max_width, ex_right_lung_max_width_image_index = get_in_or_ex_left_or_right_max_height_width(
            position='right', csv_url=ex_csv)

        df.at[index, 'in_left_lung_top_y_value'] = in_left_lung_top_y_value
        df.at[index, 'in_left_lung_bottom_y_value'] = in_left_lung_bottom_y_value
        df.at[index, 'in_left_lung_max_height (cm)'] = in_left_lung_max_height
        df.at[index, 'in_left_lung_max_height_image_index'] = in_left_lung_max_height_image_index
        df.at[index, 'in_left_lung_left_x_value'] = in_left_lung_left_x_value
        df.at[index, 'in_left_lung_right_x_value'] = in_left_lung_right_x_value
        df.at[index, 'in_left_lung_max_width (cm)'] = in_left_lung_max_width
        df.at[index, 'in_left_lung_max_width_image_index'] = in_left_lung_max_width_image_index

        df.at[index, 'in_right_lung_top_y_value'] = in_right_lung_top_y_value
        df.at[index, 'in_right_lung_bottom_y_value'] = in_right_lung_bottom_y_value
        df.at[index, 'in_right_lung_max_height (cm)'] = in_right_lung_max_height
        df.at[index, 'in_right_lung_max_height_image_index'] = in_right_lung_max_height_image_index
        df.at[index, 'in_right_lung_left_x_value'] = in_right_lung_right_x_value
        df.at[index, 'in_right_lung_right_x_value'] = in_right_lung_right_x_value
        df.at[index, 'in_right_lung_max_width (cm)'] = in_right_lung_max_width
        df.at[index, 'in_right_lung_max_width_image_index'] = in_right_lung_max_width_image_index

        df.at[index, 'ex_left_lung_top_y_value'] = ex_left_lung_top_y_value
        df.at[index, 'ex_left_lung_bottom_y_value'] = ex_left_lung_bottom_y_value
        df.at[index, 'ex_left_lung_max_height (cm)'] = ex_left_lung_max_height
        df.at[index, 'ex_left_lung_max_height_image_index'] = ex_left_lung_max_height_image_index
        df.at[index, 'ex_left_lung_left_x_value'] = ex_left_lung_left_x_value
        df.at[index, 'ex_left_lung_right_x_value'] = ex_left_lung_right_x_value
        df.at[index, 'ex_left_lung_max_width (cm)'] = ex_left_lung_max_width
        df.at[index, 'ex_left_lung_max_width_image_index'] = ex_left_lung_max_width_image_index

        df.at[index, 'ex_right_lung_top_y_value'] = ex_right_lung_top_y_value
        df.at[index, 'ex_right_lung_bottom_y_value'] = ex_right_lung_bottom_y_value
        df.at[index, 'ex_right_lung_max_height (cm)'] = ex_right_lung_max_height
        df.at[index, 'ex_right_lung_max_height_image_index'] = ex_right_lung_max_height_image_index
        df.at[index, 'ex_right_lung_left_x_value'] = ex_right_lung_left_x_value
        df.at[index, 'ex_right_lung_right_x_value'] = ex_right_lung_right_x_value
        df.at[index, 'ex_right_lung_max_width (cm)'] = ex_right_lung_max_width
        df.at[index, 'ex_right_lung_max_width_image_index'] = ex_right_lung_max_width_image_index

        end_time = timer()
        elapsed_time = end_time - start_time
        print(
            Fore.RED +
            f'The total time to process the {patient_id} is {elapsed_time} seconds\n'
        )

        df.to_csv('test/output.csv')


if __name__ == '__main__':
    main()
