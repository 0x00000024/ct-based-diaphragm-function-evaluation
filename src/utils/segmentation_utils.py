import sys
import typing
from typing import List, Tuple
import numpy as np
from nptyping import NDArray
from numpy import int32
import settings
from src.utils.geometry_utils import get_slope_between_two_points


def handle_contour_points_data_structure(
    original_data_list: List[NDArray[(typing.Any, 1, 2), np.int32]],
    max_size_index: int,
) -> NDArray[(typing.Any, 2), np.int32]:

    selected_contour_ndarray = original_data_list[max_size_index]

    points_number = selected_contour_ndarray.shape[0]

    return selected_contour_ndarray.reshape(points_number, 2)


# For the left lung, we need the largest x value
# For the right lung, we need the smallest x value
def get_index_of_x_extremum_in_contours(
        position: str, contour_points: NDArray[(typing.Any, 2),
                                               np.int32]) -> int:

    if position is 'left':
        row_number = -1
    elif position is 'right':
        row_number = 0
    else:
        raise ValueError('The value of the position needs to be left or right')

    # Sort all points based on x-axis by ascending order
    sorted_contour_points = contour_points[contour_points[:, 0].argsort()]

    index = None
    target_number = sorted_contour_points[row_number][0]
    for i, point in enumerate(contour_points):
        if int(target_number) == int(point[0]):
            index = i
            break

    return index


def get_index_of_y_maximum_in_contours(
        position: str, contour_points: NDArray[(typing.Any, 2),
                                               np.int32]) -> int:

    # Sort all points based on y-axis by ascending order
    sorted_contour_points = contour_points[contour_points[:, 1].argsort()]

    total_index = contour_points.shape[0]

    def find_index_closest_to_edge(x: int) -> int:
        index_candidates = {}
        for i, point in enumerate(contour_points):
            if int(x) == int(point[1]):
                index_candidates[i] = point[0]

        result = min(index_candidates, key=index_candidates.get) if position is 'left'\
            else max(index_candidates, key=index_candidates.get)

        return int(result)

    def is_valid_index(test_index: int) -> bool:
        if test_index == sys.maxsize:
            return False

        x = contour_points[test_index][0]
        if test_index == 140:
            print('test index: ', test_index)
            print('x: ', x)

        if position is 'left' and 0 <= x < int(settings.image_width / 4):
            return True

        if position is 'right' and (int(settings.image_width / 4 * 3) < x <=
                                    settings.image_width):
            return True

        return False

    # row_number = 0 if position is 'left' else -1
    row_number = -1
    index = sys.maxsize
    while not is_valid_index(index):
        target_number = sorted_contour_points[row_number][1]
        index = find_index_closest_to_edge(target_number)
        row_number -= 1

        if abs(row_number) > total_index:
            raise ValueError('row_number > total_index')

    return index


def handle_multiple_contours(
    contours: List[NDArray[(typing.Any, 1, 2), np.int32]],
) -> Tuple[List[NDArray[(typing.Any, 1, 2), np.int32]], int]:

    # When multiple contours are extracted, select the contour with the most points
    max_size = 0
    max_size_index = 0
    for i in range(len(contours)):
        current_contour_size = contours[i].size
        if current_contour_size > max_size:
            max_size = current_contour_size
            max_size_index = i

    # Overwrite the other contours with point(0, 0)
    for i in range(len(contours)):
        if max_size_index == i:
            continue
        # noinspection PyTypeChecker
        contours[i] = np.asarray([[[0, 0]]], dtype=int32)

    return contours, max_size_index


def get_monotonic_change_points_index(position: str,
                                      contour_points: NDArray[(typing.Any, 2),
                                                              np.int32],
                                      start_index: int,
                                      stop_index: int) -> List[int]:
    monotonic_change_points_index = []
    step = 1 if position is 'left' else -1
    slope_list = []
    first_point = [
        contour_points[start_index][0], contour_points[start_index][1]
    ]
    for i in range(start_index + step, stop_index, step):
        second_point = [contour_points[i][0], contour_points[i][1]]
        slope = get_slope_between_two_points(first_point[0], first_point[1],
                                             second_point[0], second_point[1])
        if slope == -1:
            continue

        if len(slope_list) > 0:
            if (slope_list[len(slope_list) - 1] > 0) != (slope > 0):
                monotonic_change_points_index.append(i - step)

        slope_list.append(slope)

        first_point = [contour_points[i][0], contour_points[i][1]]

    return monotonic_change_points_index


def get_magic_slope_point_index(position: str,
                                contour_points: NDArray[(typing.Any, 2),
                                                        np.int32],
                                start_index: int, stop_index: int) -> int:

    step = 1 if position is 'left' else -1
    first_point = [
        contour_points[start_index][0], contour_points[start_index][1]
    ]
    slope_dict = {}
    for i in range(start_index + step, stop_index, step):
        second_point = [contour_points[i][0], contour_points[i][1]]
        slope = get_slope_between_two_points(first_point[0], first_point[1],
                                             second_point[0], second_point[1])
        if slope == -1:
            continue

        slope_dict[i] = slope

        first_point = [contour_points[i][0], contour_points[i][1]]

    for index, slope in slope_dict.items():
        if abs(slope) >= 1.5:
            return index

    if len(slope_dict) >= 1:
        return list(slope_dict)[-1]

    return -1
