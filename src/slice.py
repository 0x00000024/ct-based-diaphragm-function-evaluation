import math
import cv2
from colorama import Fore

import settings
from lung import Lung
from utils.image_utils import merge_left_right_images


def handle_lung_slice(image_basename: str) -> None:
    image_path = settings.original_images_dirname + image_basename
    color_image = cv2.imread(image_path)
    # color_image_backup = color_image.copy()
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    left_lung = Lung(image=color_image, position='left')
    right_lung = Lung(image=color_image.copy(), position='right')

    if (settings.debugging_mode and settings.debug_lung_position
            == 'left') or not settings.debugging_mode:
        print(Fore.BLUE + 'Finding the contours of the left lung...')
        left_lung.get_lung_contour_points(draw=True)
        print(Fore.BLUE +
              'Extracting the diaphragm points of the left lung...')
        left_lung.extract_diaphragm_points(draw=True)

    if (settings.debugging_mode and settings.debug_lung_position
            == 'right') or not settings.debugging_mode:
        print(Fore.BLUE + 'Finding the contours of the right lung...')
        right_lung.get_lung_contour_points(draw=True)
        print(Fore.BLUE +
              'Extracting the diaphragm points of the right lung...')
        right_lung.extract_diaphragm_points(draw=True)

    if not settings.debugging_mode:
        print(Fore.BLUE + 'Merging images of left and right lungs...')
        merged_image = merge_left_right_images(
            left_image=left_lung.image,
            right_image=right_lung.image,
            row_start_index=0,
            row_stop_index=settings.image_height,
            column_start_index=math.floor(settings.image_height / 2),
            column_stop_index=settings.image_height)

    # np.set_printoptions(threshold=sys.maxsize)

    if settings.debugging_mode:
        if settings.debug_lung_position == 'left':
            cv2.imshow('left lung image', left_lung.image)
            image_number = int(image_basename.split('.')[0][-3:])
            left_lung.add_diaphragm_points_to_global_variable(
                image_number=image_number)
        if settings.debug_lung_position == 'right':
            cv2.imshow('right lung image', right_lung.image)
        # cv2.imshow('entire lung image', merged_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not settings.debugging_mode:
        image_number = int(image_basename.split('.')[0][-3:])
        left_lung.add_diaphragm_points_to_global_variable(
            image_number=image_number)
        right_lung.add_diaphragm_points_to_global_variable(
            image_number=image_number)

        settings.initial_slice_interval += settings.slice_interval

        print(Fore.BLUE + 'Output the result image to',
              settings.processed_images_dirname + image_basename + '\n')
        cv2.imwrite(settings.processed_images_dirname + image_basename,
                    merged_image)
