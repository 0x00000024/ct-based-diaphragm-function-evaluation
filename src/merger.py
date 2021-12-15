import os
import plotly.express as px
import pandas as pd
import settings
from colorama import Fore


def merger(in_csv: str, ex_csv: str, patient_id: str, category: str) -> None:
    df_in = pd.read_csv(in_csv)
    print(df_in.head())

    df_ex = pd.read_csv(ex_csv)
    print(df_ex.head())

    frames = [df_in, df_ex]

    df_merge = pd.concat(frames)
    print(df_merge.head())

    fig = px.scatter_3d(df_merge,
                        x='x_value',
                        y='y_value',
                        z='slice_interval',
                        range_x=[0, settings.image_width],
                        range_y=[0, settings.image_height],
                        color='image_number')

    fig.update_layout(scene_camera=settings.camera, title=patient_id + '/' + category)
    fig.data[0].marker.symbol = 'circle'
    fig.data[0].marker.size = 1
    fig.write_html(settings.processed_images_dirname + settings.html_filename)

    print(Fore.BLUE + 'Uploading demo files to server...\n')
    upload_cmd = 'cd ' + settings.images_dirname + 'processed && ls -al && find ' + patient_id + '/' + category + '/' +\
                 settings.date + '/' + ' -name "*.gif" -o -name "*.html" -o -name "*.csv" | xargs tar cfP - |' + \
                 ' ssh root@ct.eny.li tar xfP - -C /var/www/html && rm -rf * && echo'
    os.system(upload_cmd)
