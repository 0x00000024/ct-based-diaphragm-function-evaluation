import plotly.express as px
import pandas as pd
import settings

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

fig.write_html(
    '/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/images/processed/'
    + settings.patient_id + '/merged.html')
