from pandas import DataFrame
import pandas as pd
from colorama import Fore
from src.patient.base import Patient
from timeit import default_timer as timer

def main() -> None:
    df: DataFrame = pd.read_csv('../test/input.csv', dtype={
        'patient_id': str,
    })
    for patient_id in df['patient_id'].tolist():
        print(Fore.RED + f'Processing {patient_id}')
        start_time = timer()

        Patient(patient_id, '../images')

        end_time = timer()
        elapsed_time = end_time - start_time
        print(Fore.RED + f'The total time to process is {elapsed_time} seconds\n')


if __name__ == '__main__':
    main()
