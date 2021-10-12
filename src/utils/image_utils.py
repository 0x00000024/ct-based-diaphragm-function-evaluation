from pathlib import Path

import cv2
from typing import Tuple
import imageio
import numpy as np
import typing
from nptyping import NDArray
from os import listdir
from os.path import isfile, join
import settings


def get_grayscale_image(image_filename: str) -> np.ndarray:
    image = cv2.imread(image_filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def print_image_info(image: NDArray[(typing.Any, 2), np.int32]) -> None:
    print('--------------------------------------------------------')
    print('Image Dimension    : ', image.shape)
    print('Image Height       : ', image.shape[0])
    print('Image Width        : ', image.shape[1])
    print('Number of Channels : ', image.shape[2])
    print('--------------------------------------------------------')


def get_color_upper_lower_boundaries(
    position: str
) -> Tuple[NDArray[(3, ), np.uint8], NDArray[(3, ), np.uint8]]:
    if position is 'left':
        return settings.left_lung_lower_boundary, settings.left_lung_upper_boundary
    if position is 'right':
        return settings.right_lung_lower_boundary, settings.right_lung_upper_boundary

    raise ValueError('The value of the string needs to be left or right')


def convert_part_of_image_to_black(
        image: NDArray[(typing.Any, 2), np.int32], row_start_index: int,
        row_stop_index: int, column_start_index: int,
        column_stop_index: int) -> NDArray[(typing.Any, 2), np.int32]:

    for i in range(row_start_index, row_stop_index + 1, 1):
        for j in range(column_start_index, column_stop_index + 1, 1):
            image[i][j] = 0

    return image


def merge_left_right_images(
        left_image: NDArray[(typing.Any, 2),
                            np.int32], right_image: NDArray[(typing.Any, 2),
                                                            np.int32],
        row_start_index: int, row_stop_index: int, column_start_index: int,
        column_stop_index: int) -> NDArray[(typing.Any, 2), np.int32]:

    for i in range(row_start_index, row_stop_index, 1):
        for j in range(column_start_index, column_stop_index, 1):
            left_image[i][j] = right_image[i][j]

    return left_image


def jpg2gif(image_dirname: str, output_filename: str) -> None:
    jpg_images = []
    jpg_images_basename = [
        f for f in sorted(listdir(image_dirname))
        if isfile(join(image_dirname, f)) and f.endswith('.jpg')
    ]
    for jpg_image_basename in jpg_images_basename:
        jpg_images.append(
            imageio.imread(settings.processed_images_dirname +
                           jpg_image_basename))
    Path(image_dirname).mkdir(parents=True, exist_ok=True)
    imageio.mimsave(image_dirname + output_filename, jpg_images)
