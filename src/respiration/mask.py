import shutil
from pathlib import Path
import numpy as np
from pandas import DataFrame
from pyntcloud import PyntCloud
from src.utility.checker import Checker
from src.utility.exporter import Exporter
from src.utility.server import Server


class Mask:

    def __init__(self, result_dir_path: str, lung_base_df: DataFrame,
                 category: str, lung_base_3d_point_cloud_filename: str,
                 server: Server) -> None:
        self._result_dir_path: str = result_dir_path
        self._in_result_dir_path: str = f'{str(Path(self._result_dir_path).parent.absolute())}/in'
        self._lung_base_df: DataFrame = lung_base_df
        self._category: str = category
        self._lung_base_3d_point_cloud_filename: str = lung_base_3d_point_cloud_filename
        self._lung_base_mask_3d_point_cloud_filename: str = "3.lung_base_mask_3d_point_cloud"
        self._lung_base_mask_mesh_filename: str = "4.lung_base_mask_mesh(manually)"
        self._lung_base_mask_l2r_3d_point_cloud: str = "5.lung_base_mask_l2r_3d_point_cloud"
        self._lung_base_mask_r2l_3d_point_cloud: str = "5.lung_base_mask_r2l_3d_point_cloud"
        self._server: Server = server

    def squash_z_axis(self) -> None:
        df: DataFrame = self._lung_base_df
        df = df.drop(labels='z', axis='columns')
        df[list("xy")] = np.round(df[list("xy")] * 1).astype(int).astype(float)
        df['z'] = 0.0
        df = df.drop_duplicates()
        Exporter(df, self._result_dir_path,
                 self._lung_base_mask_3d_point_cloud_filename)

        shutil.copy(
            f'{self._result_dir_path}/{self._lung_base_mask_3d_point_cloud_filename}.ply',
            f'{self._result_dir_path}/{self._lung_base_mask_mesh_filename}.ply'
        )
        Checker(self._result_dir_path, self._lung_base_mask_mesh_filename)
        self._server.upload_file(f'{self._lung_base_mask_mesh_filename}.ply')

        # if self._category == "in":
        #     shutil.copy(
        #         f'{self._result_dir_path}/{self._lung_base_mask_3d_point_cloud_filename}.ply',
        #         f'{self._result_dir_path}/{self._lung_base_mask_mesh_filename}.ply'
        #     )
        #     Checker(self._result_dir_path, self._lung_base_mask_mesh_filename)
        # if self._category == "ex":
        #     shutil.copy(
        #         f'{self._in_result_dir_path}/{self._lung_base_mask_mesh_filename}.ply',
        #         f'{self._result_dir_path}/{self._lung_base_mask_mesh_filename}.ply'
        #     )
        # self._server.upload_file(f'{self._lung_base_mask_mesh_filename}.ply')

        # if self._category == "ex":
        #     df: DataFrame = PyntCloud.from_file(f'{self._result_dir_path}/2.lung_base_3d_point_cloud.ply').points
        #
        #     df = df.drop(labels='z', axis='columns')
        #     df[list("xy")] = np.round(df[list("xy")] * 1).astype(int).astype(float)
        #     df['z'] = 0.0
        #     df = df.drop_duplicates()
        #     Exporter(df, self._result_dir_path, self._lung_base_mask_3d_point_cloud_filename)
        #     shutil.copy(
        #         f'{self._result_dir_path}/{self._lung_base_mask_3d_point_cloud_filename}.ply',
        #         f'{self._result_dir_path}/{self._lung_base_mask_mesh_filename}.ply'
        #     )
        #     f = tempfile.NamedTemporaryFile(mode='w', delete=False)
        #     f.write(
        #         "(sleep 10 && osascript -e 'tell application \"System Events\" to keystroke \"n\" using {control down, option down, command down, shift down}') &"
        #     )
        #     f.close()
        #     os.system('bash ' + f.name)
        #     Checker(self._result_dir_path, self._lung_base_mask_mesh_filename)
        #     self._server.upload_file(f'{self._lung_base_mask_mesh_filename}.ply')

    def run_cgal_program(self) -> None:
        self._server.run_remote_cmd(
            "/root/ct/5-8-auto",
            f'\'{self._lung_base_3d_point_cloud_filename}.ply\' \'{self._lung_base_mask_mesh_filename}.ply\'',
            False)

        self._server.download_file(
            f'{self._lung_base_mask_l2r_3d_point_cloud}.ply')
        self._server.download_file(
            f'{self._lung_base_mask_r2l_3d_point_cloud}.ply')

        # if self._category == "in":
        #     self._server.download_file(
        #         f'{self._lung_base_mask_l2r_3d_point_cloud}.ply')
        #     self._server.download_file(
        #         f'{self._lung_base_mask_r2l_3d_point_cloud}.ply')
        # if self._category == "ex":
        #     shutil.copy(
        #         f'{self._in_result_dir_path}/{self._lung_base_mask_l2r_3d_point_cloud}.ply',
        #         f'{self._result_dir_path}/{self._lung_base_mask_l2r_3d_point_cloud}.ply'
        #     )
        #     shutil.copy(
        #         f'{self._in_result_dir_path}/{self._lung_base_mask_r2l_3d_point_cloud}.ply',
        #         f'{self._result_dir_path}/{self._lung_base_mask_r2l_3d_point_cloud}.ply'
        #     )

        # if self._category == "ex":
        #     self._server.run_remote_cmd(
        #         "/root/ct/5-8-auto",
        #         f'\'{self._lung_base_3d_point_cloud_filename}.ply\' \'{self._lung_base_mask_mesh_filename}.ply\'',
        #         False)
        #     self._server.download_file(
        #         f'{self._lung_base_mask_l2r_3d_point_cloud}.ply')
        #     self._server.download_file(
        #         f'{self._lung_base_mask_r2l_3d_point_cloud}.ply')

    def get_mask_range_df(self) -> DataFrame:

        def get_dataframe(filename: str) -> DataFrame:
            df_one_side: DataFrame = PyntCloud.from_file(
                f'{self._result_dir_path}/{filename}.ply').points
            df_one_side.drop('z', axis=1, inplace=True)
            df_one_side = df_one_side.sort_values(['x']).drop_duplicates()
            return df_one_side

        df_l2r: DataFrame = get_dataframe(
            self._lung_base_mask_l2r_3d_point_cloud)
        df_r2l: DataFrame = get_dataframe(
            self._lung_base_mask_r2l_3d_point_cloud)
        df: DataFrame = df_l2r.merge(right=df_r2l, left_on='x', right_on='x')
        df = df.rename(columns={'y_x': 'y_left', 'y_y': 'y_right'})

        for index, row in df.iterrows():
            if row['y_left'] > row['y_right']:
                df.drop([index], inplace=True)

        return df
