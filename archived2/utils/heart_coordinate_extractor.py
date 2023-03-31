from os.path import exists
import pandas as pd
import settings
import cv2


def store_coordinates(dirname: str, image_basename: str, x: int, y: int) -> None:
    heart_file_path = dirname + 'heart.csv'

    if not exists(heart_file_path):
        with open(heart_file_path, 'w') as f:
            f.write('image_filename,x,y')

    with open(heart_file_path, "a") as f:
        f.write(f'\n{image_basename},{x},{y}')


def extract(original_images_dirname: str, category: str, image_basename: str) -> None:
    image_path = original_images_dirname + category + '/' + image_basename
    print('image_path', image_path)

    # Read the image
    img = cv2.imread(image_path, 1)

    # Display the image
    cv2.imshow('image', img)

    # Display the coordinates of the points clicked on the image
    def click_event(event, x, y, flags, params):
        # Check for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            settings.selected_x_value = x
            settings.selected_y_value = y
            print(x, y)

            # Display the coordinates on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
            cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)

    # Wait for a key to be pressed to exit
    cv2.waitKey(0)
    store_coordinates(original_images_dirname + category + '/', image_basename, settings.selected_x_value,
                      settings.selected_y_value)
    cv2.destroyAllWindows()


def main() -> None:
    df = pd.read_csv('../test/heart.csv',
                     dtype={
                         'patient_id': str,
                         'start_image_number': int,
                         'stop_image_number': int,
                     })

    for index, row in df.iterrows():
        patient_id = row['patient_id']
        start_image_number = row['start_image_number']
        stop_image_number = row['stop_image_number']
        original_images_dirname = settings.images_dirname + 'original/' + patient_id + '/'

        for i in range(start_image_number, stop_image_number + 1, 1):
            image_basename = f"IM-0001-{i:04d}.jpg"
            extract(original_images_dirname, 'in', image_basename)
            extract(original_images_dirname, 'ex', image_basename)


if __name__ == "__main__":
    main()
