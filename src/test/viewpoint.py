import plotly.express as px
import pandas as pd
import settings
from src.utils.debugger import my_debugger, var_info

df_ex = pd.read_csv(
    '/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/images/processed/'
    + settings.patient_id + '/expiration/points.csv')
print(df_ex.head())

df_in = pd.read_csv(
    '/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/images/processed/'
    + settings.patient_id + '/inspiration/points.csv')
print(df_in.head())

frames = [df_in, df_ex]

df_merge = pd.concat(frames)
print(df_merge.head())
fig = px.scatter_3d(df_merge,
                    x='x_value',
                    y='y_value',
                    z='slice_interval',
                    color='image_number')

fig.data[0].marker.symbol = 'circle'
fig.data[0].marker.size = 1

name = 'default'
# Default parameters which are used when `layout.scene.camera` is not provided
camera = dict(up=dict(x=0, y=1, z=0),
              center=dict(x=0, y=0, z=0),
              eye=dict(x=0, y=1.25, z=-1.5))


fig.update_layout(scene_camera=camera, title=name)

fig.write_html(
    '/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/images/processed/'
    + settings.patient_id + '/merged.html')

print('Done!!')
