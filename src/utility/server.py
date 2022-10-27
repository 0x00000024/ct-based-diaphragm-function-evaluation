from colorama import Fore
import tempfile
from src.utility.os_system_wrapper import OsSystemWrapper


class Server:

    def __init__(self, host: str, port: str, user: str,
                 result_dir_path: str) -> None:
        self._host: str = host
        self._port: str = port
        self._user: str = user
        self.result_dir_path: str = result_dir_path

    def run_remote_cmd(self, program_path: str, arguments: str,
                       return_result: bool) -> str:
        print(Fore.YELLOW + f'Running {program_path} {self.result_dir_path} {arguments}')
        tmp_file_path: str = ''
        redirection_symbol: str = ''
        if return_result:
            redirection_symbol = '>'
            tmp_file_path = tempfile.NamedTemporaryFile().name
        OsSystemWrapper(
            f'ssh {self._user}@{self._host} -p {self._port} "{program_path} {self.result_dir_path} {arguments}" {redirection_symbol} {tmp_file_path}'
        )
        print(Fore.YELLOW + f'Finished running {program_path} {arguments}')

        if return_result:
            f = open(tmp_file_path, "r")
            return f.read()

        return '0'

    def upload_file(self, filename: str) -> None:
        print(Fore.BLUE + f'Uploading {filename}')
        OsSystemWrapper(
            f'ssh {self._user}@{self._host} -p {self._port} "mkdir --parents {self.result_dir_path}"'
        )
        OsSystemWrapper(
            f'scp -P {self._port} "{self.result_dir_path}/{filename}" {self._user}@{self._host}:{self.result_dir_path}'
        )
        print(Fore.BLUE + f'Uploaded {filename}')

    def download_file(self, filename: str) -> None:
        print(Fore.MAGENTA + f'Downloading {filename}')
        OsSystemWrapper(
            f'scp -P {self._port} {self._user}@{self._host}:{self.result_dir_path}/{filename} {self.result_dir_path}/{filename}'
        )
        print(Fore.MAGENTA + f'Downloaded {filename}')
