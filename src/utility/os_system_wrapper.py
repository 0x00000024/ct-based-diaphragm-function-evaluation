from colorama import Fore
import os
import sys


class OsSystemWrapper:

    def __init__(self, command: str) -> None:
        self._command: str = command
        self._run()

    def _run(self) -> None:
        return_code: int = os.system(self._command)
        if return_code != 0:
            print(Fore.RED + f'Failed to run {self._command}')
            sys.exit(1)
