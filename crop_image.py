import cv2
from os import listdir
from os.path import isfile, join


def crop_image(dir_path, filename):
    frame = cv2.imread(dir_path + '/' + filename)
    x = 210
    y = 340
    w = h = 100
    crop_img = frame[y:y + h, x:x + w]
    cv2.imwrite(dir_path + '/cropped/' + 'Cropped-' + filename, crop_img)


image_dir_path = 'images'
image_filename_list = [f for f in listdir(image_dir_path) if isfile(join(image_dir_path, f))]

for image_filename in image_filename_list:
    print(image_filename)
    crop_image(image_dir_path, image_filename)
