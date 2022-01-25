import math
from typing import List, Tuple
import numpy as np
import typing
import pandas as pd
from nptyping import NDArray
from numpy import vstack, ones
from numpy.linalg import lstsq


def get_slope_between_two_points(x1: float, y1: float, x2: float,
                                 y2: float) -> float:
    if y2 - y1 == 0 or x2 - x1 == 0:
        return -1
    return (y2 - y1) / (x2 - x1)


def get_distance_between_two_points(x1: float, y1: float, x2: float,
                                    y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def get_middle_point_index(position: str,
                           contour_points: NDArray[(typing.Any, 2), np.int32],
                           start_index: int, stop_index: int) -> int:
    step = 1 if position is 'left' else -1
    middle_point_x_value = int(
        (contour_points[start_index][0] + contour_points[stop_index][0]) / 2)
    for i in range(start_index, stop_index - 1, step):
        if contour_points[i][0] <= middle_point_x_value <= contour_points[i +
                                                                          1][0]:
            return i
    return -1


def get_two_equation_least_squares_solution(
        points: NDArray[(2, 2), np.int32]) -> Tuple[int, int]:
    x_coords, y_coords = zip(*points)
    a = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(a, y_coords)[0]
    return round(m), round(c)


def linear_interpolation(
        points: NDArray[(2, 2),
                        np.int32]) -> NDArray[(typing.Any, 2), np.int32]:
    m, c = get_two_equation_least_squares_solution(points)
    point_a = points[0]
    point_a_x = point_a[0]
    point_b = points[1]
    point_b_x = point_b[0]
    step = 1 if point_a_x - point_b_x < 0 else -1

    # Use list instead of ndarray to avoid unnecessary memory allocation
    points_list = []
    for x in range(point_a_x + 1, point_b_x, step):
        y = m * x + c
        points_list.append([x, y])

    # No need to reshape, it is used to help PyCharm perform code static analysis
    return np.asarray(points_list, dtype=np.int32).reshape(len(points_list), 2)


# Get the number of points at the bottom of the left and right lungs
def get_number_of_points(curr_slice_df: pd.DataFrame) -> Tuple[int, int]:
    cut_off_point_index = curr_slice_df.loc[curr_slice_df['x_value'] == 0]['Unnamed: 0'].tolist()[0]

    curr_slice_df = curr_slice_df[curr_slice_df['x_value'] != 0]

    left_lung_dict = {}
    right_lung_dict = {}

    for index, row in curr_slice_df.iterrows():
        # Left lung
        if index < cut_off_point_index:
            if row['x_value'] not in left_lung_dict:
                left_lung_dict[row['x_value']] = 1
            else:
                left_lung_dict[row['x_value']] += 1
        # Right lung
        if index > cut_off_point_index:
            if row['x_value'] not in right_lung_dict:
                right_lung_dict[row['x_value']] = 1
            else:
                right_lung_dict[row['x_value']] += 1

    return len(left_lung_dict.keys()), len(right_lung_dict.keys())
