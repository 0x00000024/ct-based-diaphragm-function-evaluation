import cv2 as cv
import numpy as np
from nptyping import NDArray, Shape, Int8
from src.slice.utility import ClockwiseAngleAndDistance
from colorama import Fore


class Denoiser:

    def __init__(self, rows, columns, pixel_data) -> None:
        self._rows: int = rows
        self._columns: int = columns
        self._pixel_data: NDArray[Shape[rows, columns], Int8] = pixel_data
        kernel_size: int = 13
        self._kernel: NDArray[Shape[kernel_size, kernel_size], Int8] = np.ones(
            (kernel_size, kernel_size), np.uint8)
        self._thorax_mask: NDArray[Shape[rows, columns], Int8] | None = None
        self._thorax_roi: NDArray[Shape[rows, columns], Int8] | None = None
        self._lung_mask: NDArray[Shape[rows, columns], Int8] | None = None
        self._lung_roi: NDArray[Shape[rows, columns], Int8] | None = None

    def _get_largest_contour_mask(
        self, pixel_data: NDArray[Shape["*, *"],
                                  Int8]) -> NDArray[Shape["*, *"], Int8]:
        rows: int = self._rows
        columns: int = self._columns

        # Use Suzuki's contour tracing algorithm to find the largest contour in the binary image.
        contours: tuple = cv.findContours(image=pixel_data,
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE)
        contours: tuple = contours[0] if len(contours) == 2 else contours[1]
        largest_contour: NDArray = max(contours, key=cv.contourArea)

        # Initialize an image with a pixel value of 0,
        # and fill the largest contour with a pixel value of 255 to make a mask.
        mask: NDArray[Shape[rows, columns],
                      Int8] = np.zeros_like(self._pixel_data)
        cv.drawContours(image=mask,
                        contours=[largest_contour],
                        contourIdx=0,
                        color=255,
                        thickness=-1)

        return mask

    def _get_inverted_mask(
        self, pixel_data: NDArray[Shape["*, *"],
                                  Int8]) -> NDArray[Shape["*, *"], Int8]:
        rows: int = self._rows
        columns: int = self._columns

        # Apply erosion morphological transformation with 10 X 10 kernel and 1 iteration.
        eroded_mask: NDArray[Shape[rows, columns],
                             Int8] = cv.erode(src=pixel_data,
                                              kernel=self._kernel,
                                              iterations=1)

        # Apply an inverse binary thresholding to the mask.
        _, inverted_mask = cv.threshold(src=eroded_mask,
                                        thresh=127,
                                        maxval=255,
                                        type=cv.THRESH_BINARY_INV)

        return inverted_mask

    def set_thorax_as_roi(self) -> NDArray[Shape["*, *"], Int8]:
        rows: int = self._rows
        columns: int = self._columns

        # Apply binary thresholding,
        # if pixel intensity is greater than the 127, value set to 255 (white), else set to 0 (black).
        _, thresh = cv.threshold(src=self._pixel_data,
                                 thresh=127,
                                 maxval=255,
                                 type=cv.THRESH_BINARY)

        # Smooth the largest contour.
        smoothed_image: NDArray[Shape[rows, columns],
                                Int8] = cv.morphologyEx(src=thresh,
                                                        op=cv.MORPH_OPEN,
                                                        kernel=self._kernel)

        # Get largest contour mask.
        self._thorax_mask = self._get_largest_contour_mask(smoothed_image)

        # Use bitwise and operation and mask to construct ROI.
        return cv.bitwise_and(self._pixel_data, self._thorax_mask)

    def set_lung_as_roi(self) -> NDArray[Shape["*, *"], Int8]:
        if self._thorax_roi is None:
            self._thorax_roi = self.set_thorax_as_roi()

        rows: int = self._rows
        columns: int = self._columns

        # Apply binary thresholding,
        # if the pixel intensity is greater than 127, set the value to 255 (white), otherwise set to 0 (black).
        _, thresh = cv.threshold(src=self._thorax_roi,
                                 thresh=127,
                                 maxval=255,
                                 type=cv.THRESH_BINARY)

        # Get the inverted mask used to extract the lung area.
        thorax_inverted_mask = self._get_inverted_mask(self._thorax_mask)

        # Use bitwise or operation and inverse mask to extract the lung area.
        lung_roi: NDArray[Shape[rows, columns],
                          Int8] = cv.bitwise_or(src1=thresh,
                                                src2=thorax_inverted_mask)

        # Apply the inverse binary thresholding to change the pixel value of the lung area to 255 (white).
        _, lung_roi = cv.threshold(src=lung_roi,
                                   thresh=127,
                                   maxval=255,
                                   type=cv.THRESH_BINARY_INV)

        # If no lungs are detected, return to full black mask.
        if (lung_roi == 0).all():
            return lung_roi

        # Smooth the largest contour.
        smoothed_image: NDArray[Shape[rows, columns],
                                Int8] = cv.morphologyEx(src=lung_roi,
                                                        op=cv.MORPH_CLOSE,
                                                        kernel=self._kernel,
                                                        iterations=2)

        # Use Suzuki's contour tracing algorithm to find the all contours in the binary image.
        contours, _ = cv.findContours(image=smoothed_image,
                                      mode=cv.RETR_TREE,
                                      method=cv.CHAIN_APPROX_SIMPLE)
        points = []
        for contour in contours:
            for ctr in contour:
                for pts in ctr:
                    try:
                        if pts.shape[0] == 2:
                            points.append(pts)
                        else:
                            raise Exception
                    except Exception as e:
                        print(
                            Fore.YELLOW,
                            f'Points with more than two coordinates: {pts}, {e}'
                        )

        # Get the center point of the contour points
        # by calculating the arithmetic mean of the x and y of all the points.
        center_point = np.array(points).mean(axis=0)

        # Sort a list of 2D coordinates by a clockwise angle.
        clock_ang_dist = ClockwiseAngleAndDistance(center_point)
        sorted_points = sorted(points, key=clock_ang_dist)
        merged_contour = np.array(sorted_points).reshape(
            (-1, 1, 2)).astype(np.int32)

        # Draw the reordered contour.
        cv.drawContours(image=smoothed_image,
                        contours=[merged_contour],
                        contourIdx=0,
                        color=255,
                        thickness=-1)

        lung_mask: NDArray[Shape[rows, columns],
                           Int8] = self._get_largest_contour_mask(
                               smoothed_image)

        # Enlarge the mask of the lungs to preserve the lung boundaries.
        self._lung_mask = cv.dilate(src=lung_mask,
                                    kernel=self._kernel,
                                    iterations=1)

        return cv.bitwise_and(self._pixel_data, self._lung_mask)

    def set_heart_as_roi(self) -> NDArray[Shape["*, *"], Int8]:
        if self._lung_roi is None:
            self._lung_roi = self.set_lung_as_roi()

        rows: int = self._rows
        columns: int = self._columns

        # Get the inverted mask used to extract non-lung areas in the thorax.
        lung_inverted_mask: NDArray[Shape[rows, columns],
                                    Int8] = self._get_inverted_mask(
                                        self._lung_mask)

        return cv.bitwise_and(self._thorax_mask, lung_inverted_mask)
