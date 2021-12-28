import cv2
import numpy as np
import settings
from src.utils.debugger import my_debugger, var_info
from src.utils.geometry_utils import get_middle_point_index, linear_interpolation
from src.utils.segmentation_utils import get_index_of_x_extremum_in_contours, get_index_of_y_maximum_in_contours, \
    get_monotonic_change_points_index, get_magic_slope_point_index


def extract_diaphragm_points(self, draw: bool):
    x_extremum_index = get_index_of_x_extremum_in_contours(
        position=self.position, contour_points=self.contour_points)
    y_maximum_index = get_index_of_y_maximum_in_contours(
        position=self.position, contour_points=self.contour_points)

    if settings.debugging_mode:
        print('x extremum index', x_extremum_index)
        print('y maximum index', y_maximum_index)
        print('x extremum', self.contour_points[x_extremum_index])
        print('y largest', self.contour_points[y_maximum_index])

    start_index = x_extremum_index if self.position is 'right' else y_maximum_index
    stop_index = y_maximum_index + 1 if self.position is 'right' else x_extremum_index + 1

    monotonic_stop_index = None
    if start_index > stop_index:
        if self.position == 'left':
            # When the starting point of the contour is to the right of x extremum point
            stop_index = self.contour_points.shape[0]
            monotonic_stop_index = self.contour_points.shape[0]
        if self.position == 'right':
            # When the starting point of the contour is to the left of x extremum point
            start_index = 0
            monotonic_stop_index = 0
    else:
        monotonic_stop_index = x_extremum_index + 1 if self.position == 'left' else x_extremum_index - 1

    points_between_extreme_points = self.contour_points[start_index:stop_index]

    monotonic_change_points_index = get_monotonic_change_points_index(
        position=self.position,
        contour_points=self.contour_points,
        start_index=y_maximum_index,
        stop_index=monotonic_stop_index,
    )

    if len(monotonic_change_points_index) >= 2:
        if self.position is 'left':
            stop_index = monotonic_change_points_index[1] + 1
        else:
            start_index = monotonic_change_points_index[1]
    else:
        middle_point_index = get_middle_point_index(
            position=self.position,
            contour_points=self.contour_points,
            start_index=y_maximum_index,
            stop_index=monotonic_stop_index)

        index = get_magic_slope_point_index(
            position=self.position,
            contour_points=self.contour_points,
            start_index=middle_point_index,
            stop_index=monotonic_stop_index,
        )
        if self.position is 'left':
            stop_index = index
        else:
            start_index = index

    diaphragm_points = self.contour_points[start_index:stop_index]

    print('2nd mono pnt')
    print(self.contour_points[start_index])
    print('stop')
    print(self.contour_points[stop_index])

    self.diaphragm_points = np.array(diaphragm_points).reshape(
        diaphragm_points.shape[0], 2)

    if draw:
        # Line between two extreme points - Cyan
        cv2.polylines(img=self.image,
                      pts=[points_between_extreme_points],
                      isClosed=False,
                      color=settings.color_cyan,
                      thickness=2)

        # Diaphragm line - Yellow
        cv2.polylines(img=self.image,
                      pts=[diaphragm_points],
                      isClosed=False,
                      color=settings.color_yellow,
                      # color=settings.color_blue,
                      thickness=2)

        # The point with the x extreme value - Green
        cv2.circle(img=self.image,
                   center=(self.contour_points[x_extremum_index][0],
                           self.contour_points[x_extremum_index][1]),
                   radius=1,
                   color=settings.color_green,
                   thickness=3)

        # The point with the largest y value - Green
        cv2.circle(img=self.image,
                   center=(self.contour_points[y_maximum_index][0],
                           self.contour_points[y_maximum_index][1]),
                   radius=1,
                   color=settings.color_green,
                   thickness=3)

        # Monotonic change points - Black / Purple
        for i, point_index in enumerate(monotonic_change_points_index):
            rendered_color = settings.color_black if i == 1 else settings.color_purple
            cv2.circle(img=self.image,
                       center=(self.contour_points[point_index][0],
                               self.contour_points[point_index][1]),
                       radius=1,
                       color=rendered_color,
                       thickness=5)

        # Contour points - Azure
        for point in self.diaphragm_points:
            cv2.circle(img=self.image,
                       center=(point[0], point[1]),
                       radius=1,
                       color=settings.color_azure,
                       thickness=1)
