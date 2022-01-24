import os
import plotly.express as px
import pandas as pd
import settings
from colorama import Fore
from src.utils.debugger import my_debugger, var_info
from src.utils.geometry_utils import get_two_extreme_points


def merger(in_csv: str, ex_csv: str, patient_id: str, category: str) -> None:
    df_in = pd.read_csv(in_csv)
    # print(df_in.head())

    df_ex = pd.read_csv(ex_csv)
    # print(df_ex.head())

    start_image_num = df_in.iloc[0]['image_number']
    end_image_num = df_in.iloc[-1]['image_number']
    # end_image_num = start_image_num

    dist_list = []
    total_pixel_num = 0
    while start_image_num <= end_image_num:
        curr_image_df_in = df_in.loc[df_in['image_number'] == start_image_num]
        curr_image_df_ex = df_ex.loc[df_ex['image_number'] == start_image_num]

        in_left_extreme_point, in_right_extreme_point = get_two_extreme_points(
            curr_image_df_in)
        ex_left_extreme_point, ex_right_extreme_point = get_two_extreme_points(
            curr_image_df_ex)

        offset = [
            ex_left_extreme_point[0] - in_left_extreme_point[0],
            ex_left_extreme_point[1] - in_left_extreme_point[1]
        ]

        cut_off_point_index = df_ex.loc[(
            (df_ex['x_value'] == 0) &
            (df_ex['image_number']
             == start_image_num))]['Unnamed: 0'].tolist()[0]

        print(cut_off_point_index)
        print('offset')
        print(offset)
        # TODO: Improve performance by reduce assignment operation
        # Update the calibrated df_ex
        df_ex.loc[((df_ex['image_number'] == start_image_num) & (df_ex['Unnamed: 0'] < cut_off_point_index)),
                  'x_value'] = curr_image_df_ex['x_value'] - offset[0]
        # df_ex.loc[df_ex['image_number'] == start_image_num, 'x_value'] = 0
        df_ex.loc[((df_ex['image_number'] == start_image_num) & (df_ex['Unnamed: 0'] < cut_off_point_index)),
                  'y_value'] = curr_image_df_ex['y_value'] - offset[1]
        # df_ex.loc[df_ex['image_number'] == start_image_num, 'y_value'] = 0

        # break
        # print('offset' , offset)
        #     Fore.GREEN +
        #     f'The number of pixels between the diaphragm at different positions of the #{start_image_num} slice: '
        #     + str(curr_pixel_num))
        #
        # total_pixel_num += curr_pixel_num

        start_image_num += 1

    frames = [df_in, df_ex]

    df_merge = pd.concat(frames)
    print(df_merge.head())

    fig = px.scatter_3d(
        df_merge,
        x='x_value',
        y='y_value',
        z='slice_interval',
        # range_x=[0, settings.image_width],
        range_x=[0, 512],
        # range_y=[0, settings.image_height],
        range_y=[0, 512],
        color='image_number')

    fig.update_layout(scene_camera=settings.camera,
                      title=patient_id + '/' + category)
    fig.data[0].marker.symbol = 'circle'
    fig.data[0].marker.size = 1
    # fig.write_html(settings.processed_images_dirname + settings.html_filename)
    fig.write_html('/Users/ethan/Downloads/1.html')

    print('done')
    # print(Fore.BLUE + 'Uploading demo files to server...\n')
    # upload_cmd = 'cd ' + settings.images_dirname + 'processed && ls -al && find ' + patient_id + '/' + category + '/' +\
    #              settings.date + '/' + ' -name "*.gif" -o -name "*.html" -o -name "*.csv" | xargs tar cfP - |' + \
    #              ' ssh root@ct.eny.li tar xfP - -C /var/www/html && rm -rf * && echo'
    # os.system(upload_cmd)


merger(in_csv='https://ct.eny.li/10532242/in/220111-191052/points.csv',
       ex_csv='https://ct.eny.li/10532242/ex/220111-191052/points.csv',
       patient_id='10532242',
       category='mix')
