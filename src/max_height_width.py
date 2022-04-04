import sys
from typing import Tuple, Dict
import pandas as pd
from nptyping import NDArray
import settings
from src.utils.geometry_utils import get_left_right_base_area


def get_in_or_ex_left_or_right_max_height_width(
        position: str,
        csv_url: str) -> Tuple[int, int, float, int, int, int, float, int]:
    data_type = {'image_number': int, 'x_value': int, 'y_value': int}
    df = pd.read_csv(csv_url, dtype=data_type)

    start_image_num = df.iloc[0]['image_number']
    end_image_num = df.iloc[-1]['image_number']

    settings.image_number = start_image_num
    curr_slice_df = df.loc[df['image_number'] == start_image_num]
    settings.z_value = curr_slice_df.iloc[0]['z_value'] * 10**-1
    start_image_num += 1

    top_y_value = None
    bottom_y_value = None
    max_height = 0
    max_height_image_index = None
    left_x_value = None
    right_x_value = None
    max_width = 0
    max_width_image_index = None

    while start_image_num <= end_image_num:
        curr_slice_df = df.loc[df['image_number'] == start_image_num]

        # sorted_contour_points = contour_points[contour_points[:, 0].argsort()]

        print(curr_slice_df.sort_values('y_value'))

        cut_off_point_index = curr_slice_df.loc[curr_slice_df['x_value'] ==
                                                0]['Unnamed: 0'].tolist()[0]

        curr_slice_df = curr_slice_df[curr_slice_df['x_value'] != 0]

        x_values = []
        y_values = []
        for index, row in curr_slice_df.iterrows():
            if position == 'left':
                # Left lung
                if index < cut_off_point_index:
                    x_values.append(row['x_value'])
                    y_values.append(row['y_value'])
            else:
                # Right lung
                if index > cut_off_point_index:
                    x_values.append(row['x_value'])
                    y_values.append(row['y_value'])

        x_values.sort()
        y_values.sort()

        local_top_y_value = y_values[-1]
        local_bottom_y_value = y_values[0]
        local_max_height = (local_top_y_value - local_bottom_y_value) * settings.row_spacing
        local_left_x_value = x_values[0]
        local_right_x_value = x_values[-1]
        local_max_width = (local_right_x_value - local_left_x_value) * settings.col_spacing

        if local_max_height > max_height:
            top_y_value = local_top_y_value
            bottom_y_value = local_bottom_y_value
            max_height = local_max_height
            max_height_image_index = start_image_num

        if local_max_width > max_width:
            left_x_value = local_left_x_value
            right_x_value = local_right_x_value
            max_width = local_max_width
            max_width_image_index = start_image_num

        start_image_num += 1

    return top_y_value, bottom_y_value, max_height, max_height_image_index, left_x_value, right_x_value, max_width, max_width_image_index
