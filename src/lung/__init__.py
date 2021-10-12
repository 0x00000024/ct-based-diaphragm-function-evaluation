import typing
import numpy as np
from nptyping import NDArray
from typing import Union
from .segmentation import contour, diaphragm
import settings
from src.utils.geometry_utils import linear_interpolation
from src.utils.debugger import my_debugger, var_info


class Lung:
    def __init__(self, image: NDArray[(typing.Any, 2), np.int32],
                 position: str) -> None:
        self.image: NDArray[(typing.Any, 2), np.int32] = image
        self.position: str = position
        self.contour_points: Union[NDArray[(typing.Any, 2), np.int32],
                                   None] = None
        self.diaphragm_points: Union[NDArray[(typing.Any, 2), np.int32],
                                     None] = None

    def get_lung_contour_points(self, draw: bool) -> None:
        contour.get_lung_contour_points(self, draw=draw)

    def extract_diaphragm_points(self, draw: bool) -> None:
        diaphragm.extract_diaphragm_points(self, draw=draw)

    def add_diaphragm_points_to_global_variable(self,
                                                image_number: int) -> None:
        if self.diaphragm_points is None or self.diaphragm_points.shape[0] == 0:
            return

        # Linear interpolation
        result = []
        first_index = 0
        second_point = None
        for second_index in range(1, len(self.diaphragm_points), 1):
            first_point = self.diaphragm_points[first_index]
            second_point = self.diaphragm_points[second_index]
            if first_point[0] == second_point[0]:
                result.append(first_point)
                first_index = second_index
                continue
            interpolated_array = linear_interpolation(
                points=np.vstack((first_point, second_point)).reshape(2, 2))
            result.append(first_point)
            for points in interpolated_array:
                result.append(points)
            first_index = second_index

        result.append(second_point)
        result = np.asarray(result)

        # Adjust the y-axis coordinate
        for i in range(len(result)):
            result[i][1] = settings.image_height - result[i][1]

        # Add slice interval column
        result = np.hstack(
            (result, np.full((result.shape[0], 1), settings.initial_slice_interval)))

        # Add image number column
        result = np.hstack((result, np.full((result.shape[0], 1),
                                            image_number)))

        if settings.diaphragm_points is None:
            settings.diaphragm_points = result
        else:
            settings.diaphragm_points = np.concatenate(
                [settings.diaphragm_points, result])
