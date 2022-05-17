from os import listdir
from os.path import join, abspath
from typing import List
import pandas as pd
from colorama import Fore
from src.slice.base import read_dicom_file_dataset, Slice
from src.slice.metadata import Metadata
import plotly.express as px
from pathlib import Path


class Respiration:

    def __init__(self, patient_id: str, category: str, image_dir_path: str) -> None:
        self._patient_id: str = patient_id
        self._category: str = category
        self._image_dir_path: str = abspath(join(image_dir_path, patient_id, category))
        self._images_path: List[str] = self._get_images_path()
        self._result_dir_path: str = self._get_result_dir_path()
        self._image_num: int = len(self._images_path)
        self._load_first_slice_metadata()
        self._lung_3d_coordinates: List[List[float]] = []

    def _load_first_slice_metadata(self) -> None:
        self._metadata: Metadata = Metadata(
            dicom_file_dataset=read_dicom_file_dataset(self._images_path[0]))

    def _get_images_path(self) -> List[str]:
        return [
            join(self._image_dir_path, f)
            for f in sorted(listdir(self._image_dir_path))
            if f.endswith(".dcm")
        ]

    def _get_result_dir_path(self) -> str:
        path = abspath(f'../result/{self._patient_id}/{self._category}')
        Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def analyze_slice(self) -> None:
        lung_3d_coordinates: List[List[float]] = []
        for image_path in self._images_path:
            _slice: Slice = Slice(image_path, self._metadata)
            lung_3d_coordinates += _slice.extract_substance_3d_coordinates(
                lower_bound=-700, upper_bound=-600)
        self._lung_3d_coordinates = lung_3d_coordinates

    def plotly(self) -> None:
        df = pd.DataFrame(self._lung_3d_coordinates, columns=['x', 'y', 'z'])
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='x')
        fig.data[0].marker.symbol = 'circle'
        fig.data[0].marker.size = 1
        fig.write_html(f'{self._result_dir_path}/3d_visualization.html')
        df.to_json(f'{self._result_dir_path}/data.json')
        print(Fore.GREEN + f'Results are stored in {self._result_dir_path}')
