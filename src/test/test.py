import cv2
import numpy as np

# Let's load a simple image with 3 black squares
from src.utils.debugger import var_info, my_debugger

image = cv2.imread(
    '/images/original/IM-0001-0160.jpg'
)
image_backup = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_boundary = np.array([12, 0, 103], dtype='uint8').reshape(1, 3)
upper_boundary = np.array([179, 177, 220], dtype='uint8').reshape(1, 3)
mask = cv2.inRange(src=image, lowerb=lower_boundary, upperb=upper_boundary)
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
opening = cv2.morphologyEx(src=mask,
                           op=cv2.MORPH_DILATE,
                           kernel=kernel,
                           iterations=1)

out = cv2.cvtColor(image_backup, cv2.COLOR_BGR2BGRA)

# Wipe out all unnecessary pixels
for i in range(len(opening)):
    for j in range(len(opening[i])):
        out[i][j] = opening[i][j] and out[i][j]

out[:, :, 3] = opening

cv2.imshow('out', out)

cv2.imwrite(
    '/Users/ethan/test/CBDFE/ct-based-diaphragm-function-evaluation/out.png',
    out)

cv2.waitKey(0)
cv2.destroyAllWindows()
