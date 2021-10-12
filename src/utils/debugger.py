import inspect
import sys
from typing import Tuple
import numpy as np
from colorama import Fore


def my_debugger(var: Tuple[str, any]) -> None:
    var_name, var_value = var
    print(Fore.GREEN + '-------------------------------------------------')

    print(Fore.GREEN + 'Name:')
    print(var_name)

    print(Fore.GREEN + 'Type:')
    var_type = type(var_value)
    print(var_type)

    if var_type is np.ndarray:
        print(Fore.GREEN + 'Shape:')
        print(var_value.shape)

    if var_type is list:
        print(Fore.GREEN + 'Length:')
        print(len(var_value))

    np.set_printoptions(threshold=sys.maxsize)
    print(Fore.GREEN + 'Value:')
    print(var_value)

    print(Fore.GREEN + '-------------------------------------------------')


def var_info(var: any) -> Tuple[str, any]:
    for fi in reversed(inspect.stack()):
        names = [
            variable_name
            for variable_name, variable_val in fi.frame.f_locals.items()
            if variable_val is var
        ]
        if len(names) > 0:
            return names[0], var
