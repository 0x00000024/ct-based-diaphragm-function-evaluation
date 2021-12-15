from typing import Tuple
import pandas as pd
import settings
from colorama import Fore


def get_volume(in_csv: str, ex_csv: str) -> Tuple[float, float, float]:
    dtype = {'image_number': int, 'x_value': int, 'y_value': int}
    df_in = pd.read_csv(in_csv, dtype=dtype)
    df_ex = pd.read_csv(ex_csv, dtype=dtype)

    start_image_num = df_in.iloc[0]['image_number']
    end_image_num = df_in.iloc[-1]['image_number']

    dist_list = []
    total_pixel_num = 0
    while start_image_num <= end_image_num:
        curr_image_df_in = df_in.loc[df_in['image_number'] == start_image_num]
        curr_image_df_ex = df_ex.loc[df_ex['image_number'] == start_image_num]
        print("curr_image_df_in['x_value']", curr_image_df_in['x_value'])
        print("curr_image_df_ex['x_value']", curr_image_df_ex['x_value'])
        start_x_value = max(curr_image_df_in['x_value'].iloc[0],
                            curr_image_df_ex['x_value'].iloc[0])
        stop_x_value = min(curr_image_df_in['x_value'].iloc[-1],
                           curr_image_df_ex['x_value'].iloc[-1])

        curr_in_dict = {}
        curr_ex_dict = {}
        for index, row in curr_image_df_in.iterrows():
            curr_in_dict[row['x_value']] = row['y_value']
        for index, row in curr_image_df_ex.iterrows():
            curr_ex_dict[row['x_value']] = row['y_value']

        curr_pixel_num = 0
        for x in range(start_x_value, stop_x_value, 1):
            if x in curr_in_dict and x in curr_ex_dict:
                if curr_ex_dict[x] < curr_in_dict[x]:
                    print(
                        Fore.RED +
                        '[Error] The position of the diaphragm during exhalation is below the position of the diaphragm'
                        ' during inhalation')
                else:
                    dist = curr_ex_dict[x] - curr_in_dict[x]
                    dist_list.append(dist)
                    curr_pixel_num += dist
            else:
                print(Fore.YELLOW + '[Info] Omitted frame:', x)

        print(
            Fore.GREEN +
            f'The number of pixels between the diaphragm at different positions of the #{start_image_num} slice: '
            + str(curr_pixel_num))

        total_pixel_num += curr_pixel_num

        start_image_num += 1

    volume = total_pixel_num * settings.row_spacing * settings.col_spacing * settings.thickness
    longest_dist = max(dist_list) * settings.col_spacing
    shortest_dist = min(dist_list) * settings.col_spacing
    print(
        Fore.GREEN +
        'The volume (cm^3) between the diaphragms in different positions:',
        volume)

    print(
        Fore.GREEN +
        'The longest distance (cm) between the diaphragms in different positions:',
        longest_dist)
    print(
        Fore.GREEN +
        'The shortest distance (cm) between the diaphragms in different positions:',
        shortest_dist)

    return volume, longest_dist, shortest_dist
