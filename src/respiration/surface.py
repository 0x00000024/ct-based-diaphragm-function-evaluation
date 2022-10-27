import os
import shutil
from typing import List
import pandas as pd
from pandas import DataFrame
from pyntcloud import PyntCloud
from src.utility.checker import Checker
from src.utility.exporter import Exporter
from src.utility.server import Server


def drop_not_in_range_row(surface_df: DataFrame, x: float, y_left: float,
                          y_right: float) -> DataFrame:
    for index, row in surface_df.iterrows():
        if row['x'] == x:
            if row['y'] < y_left or row['y'] > y_right:
                surface_df = surface_df.drop([index])
    return surface_df


class Surface:

    def __init__(self, result_dir_path: str, server: Server) -> None:
        self.result_dir_path: str = result_dir_path
        self._lung_base_mesh_filename: str = "6.lung_base_mesh"
        self._contact_surface_3d_point_cloud_filename: str = "7.lung_diaphragm_contact_surface_3d_point_cloud"
        self.contact_surface_3d_point_cloud_outlier_removed_filename: str = "8.lung_diaphragm_contact_surface_3d_point_cloud_outlier_removed(manually)"
        self._contact_surface_3d_point_cloud_coordinate_g_filename: str = "9.lung_diaphragm_contact_surface_3d_point_cloud_coordinate_g"
        self._contact_surface_3d_point_cloud_coordinate_gx_filename: str = "9.lung_diaphragm_contact_surface_3d_point_cloud_coordinate_gx"
        self._contact_surface_3d_point_cloud_coordinate_gy_filename: str = "9.lung_diaphragm_contact_surface_3d_point_cloud_coordinate_gy"
        self._contact_surface_3d_point_cloud_uniform_distribution_filename: str = "10.lung_diaphragm_contact_surface_3d_point_cloud_uniform_distribution"
        self._contact_surface_mesh_filename: str = "11.lung_diaphragm_contact_surface_mesh(manually)"
        self._server: Server = server
        self.surface_df: DataFrame = pd.DataFrame()

    def download_surface_intermediate_files(self) -> None:
        self._server.download_file(f'{self._lung_base_mesh_filename}.ply')
        self._server.download_file(
            f'{self._contact_surface_3d_point_cloud_filename}.ply')
        shutil.copy(
            f'{self.result_dir_path}/{self._contact_surface_3d_point_cloud_filename}.ply',
            f'{self.result_dir_path}/{self.contact_surface_3d_point_cloud_outlier_removed_filename}.ply'
        )
        Checker(self.result_dir_path, self.contact_surface_3d_point_cloud_outlier_removed_filename)

    def reconstruct_surface_points(self) -> DataFrame:
        g_df: DataFrame = pd.read_csv(
            f'{self.result_dir_path}/{self._contact_surface_3d_point_cloud_coordinate_g_filename}.csv',
            header=None)
        gx_df: DataFrame = pd.read_csv(
            f'{self.result_dir_path}/{self._contact_surface_3d_point_cloud_coordinate_gx_filename}.csv',
            header=None)
        gy_df: DataFrame = pd.read_csv(
            f'{self.result_dir_path}/{self._contact_surface_3d_point_cloud_coordinate_gy_filename}.csv',
            header=None)

        g = g_df.values.tolist()
        gx = gx_df.values.tolist()[0]
        gy = gy_df.values.tolist()[0]

        result: List = []
        for i in range(len(gx)):
            for j in range(len(gy)):
                result.append([gx[j], gy[i], g[i][j]])

        df: DataFrame = pd.DataFrame(result, columns=list('xyz'))

        return df

    def get_masked_contact_surface_3d_point_cloud(self,
                                                  mask_df: DataFrame) -> None:
        x_lower: float = mask_df.iloc[0]['x']
        x_upper: float = mask_df.iloc[-1]['x']
        surface_df: DataFrame = self.surface_df[
            (self.surface_df['x'] >= x_lower) &
            (self.surface_df['x'] <= x_upper)]

        for index, row in mask_df.iterrows():
            x: float = row['x']
            y_left: float = row['y_left']
            y_right: float = row['y_right']
            surface_df: DataFrame = drop_not_in_range_row(
                surface_df, x, y_left, y_right)

        surface_df[list("xy")] = surface_df[list("xy")].astype(float)
        Exporter(
            surface_df, self.result_dir_path,
            self._contact_surface_3d_point_cloud_uniform_distribution_filename)
        shutil.copy(
            f'{self.result_dir_path}/{self._contact_surface_3d_point_cloud_uniform_distribution_filename}.ply',
            f'{self.result_dir_path}/{self._contact_surface_mesh_filename}.ply')

    def calculate_surface_mesh_area(self) -> None:
        # f = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # f.write(
        #     "(sleep 10 && osascript -e 'tell application \"System Events\" to keystroke \"m\" using {control down, option down, command down, shift down}') &"
        # )
        # f.close()
        # os.system('bash ' + f.name)
        Checker(self.result_dir_path, self._contact_surface_mesh_filename)
        self._server.upload_file(f'{self._contact_surface_mesh_filename}.ply')
        area: float = float(
            self._server.run_remote_cmd(
                "/root/ct/11-area",
                f'\\"{self._contact_surface_mesh_filename}.ply\\"',
                True).split('\n')[1])
        os.system(f'echo "{area}" | pbcopy')
        print(f'area: {area}')

        self._server.run_remote_cmd(
            "/root/ct/11-area",
            f'\\"4.lung_base_mask_mesh(manually).ply\\"',
            True)

        # self._server.result_dir_path = f'{str(Path(self.result_dir_path).parent.absolute())}/in'
        # self._server.run_remote_cmd(
        #     "/root/ct/11-area",
        #     f'\\"4.lung_base_mask_mesh(manually).ply\\"',
        #     True)

    def calculate_average_height_surface_mesh(self) -> None:
        surface_df: DataFrame = PyntCloud.from_file(
            f'{self.result_dir_path}/{self._contact_surface_3d_point_cloud_uniform_distribution_filename}.ply'
        ).points
        z_mean: float = surface_df['z'].mean()
        with open("../result/log.csv", "a") as log_file:
            log_file.write("{},{}\n".format(self.result_dir_path, z_mean))
