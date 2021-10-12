import cv2

import settings
from src.utils.image_utils import get_color_upper_lower_boundaries

image_path = "/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/images/original/10157947/inspiration/IM-0001-0150.jpg"
color_image = cv2.imread(image_path)

# cv2.imshow('RGB', color_image)
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
lower_boundary, upper_boundary = get_color_upper_lower_boundaries('right')

mask = cv2.inRange(color_image, lower_boundary, upper_boundary)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)


cv2.imshow('opening', opening)
cv2.imshow('kernel', kernel)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
