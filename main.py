from contour import calculate_diaphragm_cross_sectional_area
from os import listdir
from os.path import isfile, join
from lung_seg import get_lung_slice_volume

image_dir_path = 'images/original/'
image_filename_list = [f for f in listdir(image_dir_path) if isfile(join(image_dir_path, f))]

pixel_area = 0.28 * 10 ** -3 * 0.30 * 10 ** -3
slice_thickness = 600 * 10 ** -6

print('')
print('The area of a pixel (unit: m^2): ', pixel_area)
print('The area of a pixel (unit: cm^2): ', pixel_area * 10 ** 4)
print('Slice Thickness (unit: m): ', slice_thickness)
print('Slice Thickness (unit: cm): ', slice_thickness * 10 ** 2)
print('')

complete_lung_volume = get_lung_slice_volume(image_dir_path, 'IM-0001-0305.jpg')
print('[INFO] Complete lung volume: ', complete_lung_volume)

total_diaphragm_volume = 0
for image_filename in image_filename_list:
    current_lung_slice_volume = get_lung_slice_volume(image_dir_path, image_filename)
    if current_lung_slice_volume > complete_lung_volume:
        print('[ERROR] ' + image_filename + ' | Volume: ' + str(current_lung_slice_volume) +
              '  > complete lung volume: ' + str(complete_lung_volume))
    else:
        current_diaphragm_slice_volume = complete_lung_volume - current_lung_slice_volume
        print('The complete lung volume (unit: cm^3): ', complete_lung_volume)
        print('The volume of the current slice of the diaphragm (unit: cm^3): ', current_diaphragm_slice_volume)
        print('')
        total_diaphragm_volume += current_diaphragm_slice_volume

print('Total diaphragm volume (unit: cm^3):', total_diaphragm_volume)
