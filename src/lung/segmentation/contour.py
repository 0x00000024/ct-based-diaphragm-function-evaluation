import math
import cv2
import settings
from src.utils.image_utils import get_color_upper_lower_boundaries, convert_part_of_image_to_black
from src.utils.segmentation_utils import handle_contour_points_data_structure, handle_multiple_contours


def get_lung_contour_points(self, draw: bool) -> None:

    # Rewrite the value of the left half of the image directly to 0, which is black,
    # to remove the left lung that we don't need after color-based segmentation.
    if self.position is 'left':
        self.image = convert_part_of_image_to_black(
            image=self.image,
            row_start_index=0,
            row_stop_index=settings.image_height - 1,
            column_start_index=math.floor(settings.image_height / 2) - 1,
            column_stop_index=settings.image_height - 1)
    if self.position is 'right':
        self.image = convert_part_of_image_to_black(
            image=self.image,
            row_start_index=0,
            row_stop_index=settings.image_height - 1,
            column_start_index=0,
            column_stop_index=math.floor(settings.image_height / 2) - 1)

    lower_boundary, upper_boundary = get_color_upper_lower_boundaries(
        self.position)

    mask = cv2.inRange(self.image, lower_boundary, upper_boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    max_size_index = 0
    if len(contours) > 1:
        contours, max_size_index = handle_multiple_contours(contours)

    if draw:
        cv2.drawContours(self.image, contours, -1, 255, 2)

    self.contour_points = handle_contour_points_data_structure(
        original_data_list=contours, max_size_index=max_size_index)
