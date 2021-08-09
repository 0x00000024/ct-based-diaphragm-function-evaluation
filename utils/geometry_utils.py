import math


def get_distance_between_two_points(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_middle_point_index(contour_points, image_height):
    for index in range(math.ceil(len(contour_points) / 2), len(contour_points)):
        middle_point_y_coordinate = contour_points[index][1]
        if middle_point_y_coordinate > image_height / 2:
            return index
    return -1
