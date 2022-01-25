from typing import Tuple
import pandas as pd
import settings
from colorama import Fore
from src.utils.geometry_utils import get_number_of_points


def get_in_or_ex_area(df: pd.DataFrame) -> Tuple[int, int]:
    start_image_num = df.iloc[0]['image_number']
    end_image_num = df.iloc[-1]['image_number']

    total_left_pixel_num = 0
    total_right_pixel_num = 0
    while start_image_num <= end_image_num:
        curr_slice_df = df.loc[df['image_number'] == start_image_num]

        left_lung_pixel_num, right_lung_pixel_num = get_number_of_points(
            curr_slice_df)
        total_left_pixel_num += left_lung_pixel_num
        total_right_pixel_num += right_lung_pixel_num

        start_image_num += 1

    one_pixel_area = settings.row_spacing * settings.thickness

    return total_left_pixel_num * one_pixel_area, total_right_pixel_num * one_pixel_area


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
