from pathlib import Path
import cv2
from typing import Tuple
import imageio
import numpy as np
import typing
from nptyping import NDArray
from os import listdir
from os.path import isfile, join
from pydicom import FileDataset
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
) -> Tuple[NDArray[(3,), np.uint8], NDArray[(3,), np.uint8]]:
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


def get_z_value(dicom_file_dataset: FileDataset) -> float:
    image_position_patient = dicom_file_dataset[0x20, 0x32].value
    image_orientation_patient = dicom_file_dataset[0x20, 0x37].value

    normal = [0, 0, 0]

    normal[0] = image_orientation_patient[1] * image_orientation_patient[
        5] - image_orientation_patient[2] * image_orientation_patient[4]
    normal[1] = image_orientation_patient[2] * image_orientation_patient[
        3] - image_orientation_patient[0] * image_orientation_patient[5]
    normal[2] = image_orientation_patient[0] * image_orientation_patient[
        4] - image_orientation_patient[1] * image_orientation_patient[3]

    z_value = 0
    for i in range(3):
        z_value += normal[i] * image_position_patient[i]

    return z_value


def is_margin_row(row) -> bool:
    for i in range(0, row.shape[0] - 1, 2):
        a = row[i]
        b = row[i + 1]
        for j in range(a.shape[0]):
            if a[j] != b[j]:
                return False
    return True


# Get non-margin start and stop index
def get_non_margin_row_index(image, index: int, step: int) -> int:
    while is_margin_row(image[index]):
        index += step
    return index
