import numpy as np
import sys
import cv2
from colorama import Fore
from colorama import init

from utils.geometry_utils import get_distance_between_two_points, get_middle_point_index

init(autoreset=True)

image = cv2.imread('images/IM-0001-0163-gray.jpg')
original = image.copy()

# Get dimensions of image
image_dimensions = image.shape

# Height, width, number of channels in image
image_height = image.shape[0]
image_width = image.shape[1]
image_channels = image.shape[2]

print('Image Dimension    : ', image_dimensions)
print('Image Height       : ', image_height)
print('Image Width        : ', image_width)
print('Number of Channels : ', image_channels)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0], dtype="uint8")
upper = np.array([0, 0, 228], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)

contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Segment the big lung
area = np.zeros(len(contours))
big_lung = {'index': None, 'contour': None, 'area': None}
small_lung = {'index': None, 'contour': None, 'area': None, 'diaphragm_points': []}
for i in range(len(contours)):
    area[i] = cv2.contourArea(contours[i])
big_lung['index'] = np.argmax(area)
big_lung['contour'] = contours[big_lung['index']]
big_lung['area'] = area[big_lung['index']]
cv2.drawContours(original, contours, big_lung['index'], (49, 49, 238), 2)

# Segment the small lung
area[big_lung['index']] = 0
small_lung['index'] = np.argmax(area)
small_lung['contour'] = contours[small_lung['index']]
small_lung['area'] = area[small_lung['index']]
cv2.drawContours(original, contours, small_lung['index'], 255, 2)


# Contour data format
# [[[184  57]]
#  [[181  60]]
#  [[200  57]]]

def draw_diaphragm_curve(one_side_lung_contour):
    contour_points = []
    for point in one_side_lung_contour:
        point = point[0]
        x_coordinate = point[0]
        y_coordinate = point[1]
        contour_points.append([x_coordinate, y_coordinate])

    # Sort all points based on x-axis by ascending order
    contour_points = sorted(contour_points, key=lambda k: [k[0], k[1]])

    middle_point_index = get_middle_point_index(contour_points, image_height)
    if middle_point_index == -1:
        sys.exit('Call get_middle_point_index failure')

    middle_point = contour_points[middle_point_index]
    print('Middle point: ', middle_point)

    diaphragm_points = []

    def draw_half_diaphragm_curve(start_index, stop_index, step, last_point_x_coordinate, last_point_y_coordinate):
        for index in range(start_index, stop_index, step):
            current_x_coordinate = contour_points[index][0]
            current_y_coordinate = contour_points[index][1]
            print(Fore.BLUE + 'Current point: ', current_x_coordinate, current_y_coordinate)
            distance = get_distance_between_two_points(x1=current_x_coordinate, y1=current_y_coordinate,
                                                       x2=last_point_x_coordinate, y2=last_point_y_coordinate)
            print(Fore.BLUE + 'Last point: ', last_point_x_coordinate, last_point_y_coordinate)
            print(Fore.GREEN + 'Distance: ', distance)
            print('')
            if distance > 20:
                last_point_x_coordinate += step
                print(Fore.RED + 'SKIPPED\n')
                continue

            diaphragm_points.append([current_x_coordinate, current_y_coordinate])
            last_point_x_coordinate = current_x_coordinate
            last_point_y_coordinate = current_y_coordinate

    draw_half_diaphragm_curve(start_index=middle_point_index - 1,
                              stop_index=-1,
                              step=-1,
                              last_point_x_coordinate=middle_point[0],
                              last_point_y_coordinate=middle_point[1])

    draw_half_diaphragm_curve(start_index=middle_point_index,
                              stop_index=len(contour_points),
                              step=1,
                              last_point_x_coordinate=middle_point[0],
                              last_point_y_coordinate=middle_point[1])
    print('Middle point index: ', middle_point_index)

    print('Length of diaphragm points: ', len(diaphragm_points))

    diaphragm_points = sorted(diaphragm_points, key=lambda k: [k[0], k[1]])
    diaphragm_points = np.array(diaphragm_points)
    cv2.polylines(original, [diaphragm_points], isClosed=False, color=(0, 255, 0), thickness=2)


draw_diaphragm_curve(small_lung['contour'])
draw_diaphragm_curve(big_lung['contour'])

# cv2.imshow('mask', mask)
# cv2.imshow('opening', opening)
cv2.imshow('original', original)
cv2.waitKey()
