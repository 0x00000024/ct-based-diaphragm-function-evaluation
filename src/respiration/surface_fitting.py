from pathlib import Path
from colorama import Fore
from src.utility.os_system_wrapper import OsSystemWrapper


class SurfaceFitting:

    def __init__(self,
                 contact_surface_3d_point_cloud_outlier_removed_filename: str,
                 result_dir_path: str) -> None:
        self._matlab_path: str = "/Applications/MATLAB_R2021a.app/bin/maci64/MATLAB"
        self._options: str = "-nodisplay -nodesktop -r"
        self._result_dir_path: str = result_dir_path
        self._project_dir_path: str = str(
            Path(self._result_dir_path).parent.parent.parent.absolute())
        self._matlab_dir_path: str = f'{self._project_dir_path}/tools'
        self._matlab_filename: str = "lung_base_gridfit"
        self._contact_surface_3d_point_cloud_outlier_removed_filename: str = contact_surface_3d_point_cloud_outlier_removed_filename

    def call_matlab_program(self) -> None:
        command: str = f'\'cd {self._matlab_dir_path}; {self._matlab_filename}("{self._result_dir_path}", "{self._contact_surface_3d_point_cloud_outlier_removed_filename}.ply"); exit\''
        print(
            Fore.BLUE +
            f'Calling MATLAB program: {self._matlab_path} {self._options} {command}'
        )
        OsSystemWrapper(f'{self._matlab_path} {self._options} {command}')
        print(Fore.BLUE + f'MATLAB program is finished')
