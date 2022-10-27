from colorama import Fore
from src.utility.os_system_wrapper import OsSystemWrapper


class Checker:

    def __init__(self, result_dir_path: str, filename: str) -> None:
        self._meshlab: str = "/Applications/MeshLab2022.02.app/Contents/MacOS/meshlab"
        self._result_dir_path: str = result_dir_path
        self._filename: str = filename
        self.open_with_meshlab()

    def open_with_meshlab(self) -> None:
        print(Fore.BLUE + f'Manually modifying {self._filename} with MeshLab')
        OsSystemWrapper(
            f'{self._meshlab} "{self._result_dir_path}/{self._filename}.ply"')
        print(Fore.BLUE + f'MeshLab is closed')
