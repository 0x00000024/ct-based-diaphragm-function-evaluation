import os
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
import pydicom
from nptyping import NDArray, Int, Shape
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from archived2.utils.debugger import my_debugger, var_info


def main() -> None:
    image_path: str = '/Users/ethan/Downloads/4.dcm'
    dicom_file_dataset: FileDataset | DicomDir = pydicom.dcmread(image_path)

    row_spacing: float = float(dicom_file_dataset[0x28, 0x30].value[0]) * 10**-1
    rows: int = int(dicom_file_dataset[0x28, 0x10].value)
    columns: int = int(dicom_file_dataset[0x28, 0x11].value)
    pixel_data: NDArray[Shape[rows, columns], Int] = dicom_file_dataset.pixel_array
    rescale_intercept: int = int(dicom_file_dataset[0x28, 0x1052].value)
    rescale_slope: int = int(dicom_file_dataset[0x28, 0x1053].value)
    slice_thickness: float = float(dicom_file_dataset[0x18, 0x50].value)

    gray_image: NDArray[Shape[rows, columns], Int] = np.empty(shape=(rows, columns))

    # gray_image: NDArray[Shape[rows, columns], Int] = np.empty(shape=(rows, columns))
    # hu: NDArray[Shape[rows, columns], Int] = np.zeros(shape=(rows, columns))

    # # Calculate HU value
    # for i in range(rows):
    #     for j in range(columns):
    #         # print(pixel_data[i][j])
    #         hu[i][j] = pixel_data[i][j] * rescale_slope + rescale_intercept
    #
    # # Set all pixels to white
    # for i in range(rows):
    #     for j in range(columns):
    #         gray_image[i][j] = 255
    #
    # # Select all pixels
    # hu_coordinate = []
    # for i in range(rows):
    #     for j in range(columns):
    #         if -700 < hu[i][j] < -600:
    #             hu_coordinate.append([i, j])
    #
    # # Set selected pixels to black
    # for hu_coordinate_index in range(len(hu_coordinate)):
    #     i = hu_coordinate[hu_coordinate_index][0]
    #     j = hu_coordinate[hu_coordinate_index][1]
    #     gray_image[i][j] = 0
    #
    # cv2.imshow('sample image', gray_image)

    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing image


if __name__ == '__main__':
    main()
