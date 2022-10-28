import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future, wait, ProcessPoolExecutor
from typing import List
import numpy as np
from nptyping import NDArray, Shape, Int8, Int16
from src.slice.metadata import Metadata


class HounsfieldUnitThresholder:

    def __init__(self, lower_bound: int, upper_bound: int, z_value,
                 metadata: Metadata, denoised_data: NDArray[Shape["*, *"],
                                                            Int8]) -> None:
        self._metadata: Metadata = metadata
        self._rows: int = self._metadata.rows
        self._columns: int = self._metadata.columns
        self._rescale_slope: int = self._metadata.rescale_slope
        self._rescale_intercept = self._metadata.rescale_intercept
        rows: int = self._rows
        columns: int = self._columns
        self._pixel_data: NDArray[Shape[rows, columns],
                                  Int16] = self._metadata.pixel_data
        self._denoised_data: NDArray[Shape[rows, columns], Int8] = denoised_data
        self._lower_bound: int = lower_bound
        self._upper_bound: int = upper_bound
        self._z_value: int = z_value

    def extract_substance_3d_coordinates(self) -> List[List[float]]:
        rows: int = self._rows
        columns: int = self._columns
        hu_data: NDArray[Shape[rows, columns],
                         Int16] = np.zeros(shape=(rows, columns))
        substance_3d_coordinates: List[List[float]] = []
        step = 1
        for i in range(0, rows, step):
            for j in range(0, columns, step):
                # Use denoised data as mask
                if self._denoised_data[i][j] == 0:
                    continue
                hu_data[i][j] = self._pixel_data[i][
                    j] * self._rescale_slope + self._rescale_intercept
                if self._lower_bound <= hu_data[i][j] <= self._upper_bound:
                    substance_3d_coordinates.append([
                        i * self._metadata.row_spacing,
                        j * self._metadata.col_spacing, self._z_value
                    ])
        return substance_3d_coordinates
