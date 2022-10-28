from os import listdir
from os.path import join, abspath
from typing import List
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from pyntcloud import PyntCloud
from src.respiration.mask import Mask
from src.respiration.surface import Surface
from src.respiration.surface_fitting import SurfaceFitting
from src.slice.base import read_dicom_file_dataset, Slice
from src.slice.metadata import Metadata
from src.utility.checker import Checker
from src.utility.exporter import Exporter
from src.utility.server import Server


class Respiration:

    def __init__(self, patient_id: str, category: str, image_dir_path: str,
                 roi: str) -> None:
        self._patient_id: str = patient_id
        self._category: str = category
        self._image_dir_path: str = abspath(
            join(image_dir_path, patient_id, category))
        self._images_path: List[str] = self._get_images_path()
        self._result_dir_path: str = self._get_result_dir_path()
        self._image_num: int = len(self._images_path)
        self._load_first_slice_metadata()
        self._lung_3d_coordinates: List[List[float]] = []
        self._lung_df: DataFrame = pd.DataFrame()
        self._lung_base_df: DataFrame = pd.DataFrame()
        self._lung_base_mask_df: DataFrame = pd.DataFrame()
        self._largest_z_value: int = 0
        self._lung_3d_point_cloud: str = "1.lung_3d_point_cloud"
        self._lung_base_3d_point_cloud_filename: str = "2.lung_base_3d_point_cloud"
        self._server: Server = Server("rp.eny.li", "4005", "root",
                                      self._result_dir_path)
        self._roi: str = roi
        self._alternative_roi: str = "thorax" if roi == "lung" else "lung"

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

    def calculate_lung_points_coordinates(self) -> None:
        lung_3d_coordinates: List[List[float]] = []
        for image_path in self._images_path:
            lung_3d_coordinates += Slice(
                image_path, self._metadata).extract_substance_3d_coordinates(
                    lower_bound=-700, upper_bound=-600, roi=self._roi)

        lung_df: DataFrame = pd.DataFrame(lung_3d_coordinates,
                                          columns=['x', 'y', 'z'])
        Exporter(lung_df, self._result_dir_path, self._lung_3d_point_cloud)
        # Checker(self._result_dir_path, self._lung_3d_point_cloud)

    def extract_lung_base(self) -> str:
        Checker(self._result_dir_path, self._lung_3d_point_cloud)
        lung_df: DataFrame = PyntCloud.from_file(
            f'{self._result_dir_path}/{self._lung_3d_point_cloud}.ply').points
        df: DataFrame = lung_df.sort_values(['x',
                                             'y']).drop_duplicates(['x', 'y'],
                                                                   keep='last')
        range_stride: int = 40
        range_median: int = df['z'].mode()[0]
        range_left: int = range_median - range_stride
        range_right: int = lung_df["z"].max()
        df = df[df['z'].between(range_left, range_right)]
        self._lung_base_df = df
        Exporter(self._lung_base_df, self._result_dir_path,
                 self._lung_base_3d_point_cloud_filename)
        Checker(self._result_dir_path, self._lung_base_3d_point_cloud_filename)
        # print(
        #     Fore.GREEN +
        #     f'{self._roi} mask is being used, do you need to change the mask?')
        # if input("Please enter Y/N: ").lower() == 'y':
        #     return self._alternative_roi
        self._server.upload_file(
            f'{self._lung_base_3d_point_cloud_filename}.ply')
        return '0'

    def synthesize_surface_mesh(self) -> None:
        mask: Mask = Mask(self._result_dir_path, self._lung_base_df,
                          self._category,
                          self._lung_base_3d_point_cloud_filename, self._server)
        mask.squash_z_axis()
        mask.run_cgal_program()
        mask_df: DataFrame = mask.get_mask_range_df()

        surface: Surface = Surface(self._result_dir_path, self._server)
        surface.download_surface_intermediate_files()
        SurfaceFitting(
            surface.contact_surface_3d_point_cloud_outlier_removed_filename,
            surface.result_dir_path).call_matlab_program()
        surface.surface_df = surface.reconstruct_surface_points()
        surface.get_masked_contact_surface_3d_point_cloud(mask_df)
        surface.calculate_surface_mesh_area()
        surface.calculate_average_height_surface_mesh()
