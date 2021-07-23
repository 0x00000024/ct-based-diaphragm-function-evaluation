import numpy as np
import cv2


image = cv2.imread('images/original/IM-0001-0305.jpg')
original = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower = np.array([0, 0, 114], dtype="uint8")
# upper = np.array([4, 7, 255], dtype="uint8")
lower = np.array([0, 0, 148], dtype="uint8")
upper = np.array([0, 0, 154], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=0)

contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

area = 0
for contour in contours:
    area += cv2.contourArea(contour)
    cv2.drawContours(original, [contour], 0, (49, 49, 238), 2)

# cv2.imwrite('images/contoured/Contoured-' + image_filename, original)

print('Area(pixels):', area)
cv2.imshow('mask', mask)
cv2.imshow('opening', opening)
cv2.imshow('original', original)
cv2.waitKey()
