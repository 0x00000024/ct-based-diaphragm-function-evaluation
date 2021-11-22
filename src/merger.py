import os
import plotly.express as px
import pandas as pd
import settings
from colorama import Fore

in_csv_url = 'https://ct.eny.li/10017162/in/211109-113136/points.csv'
ex_csv_url = 'https://ct.eny.li/10017162/ex/211109-113230/points.csv'

df_in = pd.read_csv(in_csv_url)
print(df_in.head())

df_ex = pd.read_csv(ex_csv_url)
print(df_ex.head())

frames = [df_in, df_ex]

df_merge = pd.concat(frames)
print(df_merge.head())
fig = px.scatter_3d(df_merge,
                    x='x_value',
                    y='y_value',
                    z='slice_interval',
                    range_x=[0, 512],
                    range_y=[0, 512],
                    color='image_number')

fig.update_layout(scene_camera=settings.camera,
                  title=settings.patient_id + '/' + settings.category)
fig.data[0].marker.symbol = 'circle'
fig.data[0].marker.size = 1
fig.write_html(settings.processed_images_dirname + settings.html_filename)
print(Fore.BLUE + 'Uploading demo files to server...\n')
os.system(settings.upload_cmd)
print(Fore.BLUE + '3d demo:', settings.html_url + '\n')
