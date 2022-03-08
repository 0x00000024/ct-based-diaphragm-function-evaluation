from typing import Tuple
import pandas as pd
from colorama import Fore
from src.utils.geometry_utils import get_left_right_base_area


def get_in_or_ex_area(df: pd.DataFrame) -> Tuple[float, float]:
    start_image_num = df.iloc[0]['image_number']
    end_image_num = df.iloc[-1]['image_number']

    left_lung_total_base_area = 0
    right_lung_total_base_area = 0
    while start_image_num <= end_image_num:
        curr_slice_df = df.loc[df['image_number'] == start_image_num]

        left_lung_base_area, right_lung_base_area = get_left_right_base_area(curr_slice_df)

        left_lung_total_base_area += left_lung_base_area
        right_lung_total_base_area += left_lung_base_area

        start_image_num += 1

    return left_lung_total_base_area, right_lung_total_base_area


def get_area(in_csv: str, ex_csv: str) -> Tuple[float, float, float, float]:
    dtype = {'image_number': int, 'x_value': int, 'y_value': int}
    df_in = pd.read_csv(in_csv, dtype=dtype)
    df_ex = pd.read_csv(ex_csv, dtype=dtype)

    in_left_area, in_right_area = get_in_or_ex_area(df_in)
    ex_left_area, ex_right_area = get_in_or_ex_area(df_ex)

    print(Fore.GREEN + 'The area (cm^2) under the left lung during inhalation',
          in_left_area)

    print(Fore.GREEN + 'The area (cm^2) under the left lung during inhalation',
          in_right_area)

    print(Fore.GREEN + 'The area (cm^2) under the left lung during exhalation',
          ex_left_area)

    print(Fore.GREEN + 'The area (cm^2) under the left lung during exhalation',
          ex_right_area)

    return in_left_area, in_right_area, ex_left_area, ex_right_area
