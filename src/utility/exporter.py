import pandas as pd
from pandas import DataFrame
from colorama import Fore
import plotly.express as px
from pyntcloud import PyntCloud


class Exporter:

    def __init__(self, df: DataFrame, result_dir_path: str, filename: str) -> None:
        self._df: DataFrame = df.drop_duplicates()
        self._result_dir_path: str = result_dir_path
        self._filename: str = filename
        self.export_as_ply()

    def export_as_interactive_html(self) -> None:
        fig = px.scatter_3d(self._df,
                            x='x',
                            y='y',
                            z='z',
                            color='x',
                            range_x=[0, 350],
                            range_y=[0, 350],
                            range_z=[0, 350])
        fig.data[0].marker.symbol = 'circle'
        fig.data[0].marker.size = 1
        html_path: str = f'{self._result_dir_path}/{self._filename}.html'
        fig.write_html(html_path)
        print(Fore.GREEN + f'Exported interactive HTML file to {html_path}')

    def export_as_ply(self) -> None:
        cloud: PyntCloud = PyntCloud(pd.DataFrame(self._df))
        ply_path: str = f'{self._result_dir_path}/{self._filename}.ply'
        cloud.to_file(ply_path)
        print(Fore.GREEN + f'Exported PLY file to {ply_path}')
