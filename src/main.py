from colorama import Fore
from src.patient.base import Patient
from timeit import default_timer as timer


def main() -> None:
    start_time = timer()

    p = Patient('10003382', '../images')
    p_in = p.get_inhalation()
    p_in.analyze_slice()
    p_in.plotly()

    end_time = timer()
    elapsed_time = end_time - start_time
    print(Fore.RED + f'The total time to process is {elapsed_time} seconds\n')


if __name__ == '__main__':
    main()
