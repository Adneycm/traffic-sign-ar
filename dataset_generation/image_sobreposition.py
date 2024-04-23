import cv2 as cv
import numpy as np
import shutil
import os
from PIL import Image
import sys

INPUT_DIR_TRAFFIC_SIGN = 'dataset_generation/labeled/traffic_signs'
INPUT_DIR_COVER = 'dataset_generation/labeled/tree_branchs'

OUTPUT_BASE = 'dataset_generation/train_dataset'
WINDOW_NAME = 'traffic_signs_covered? (y/*)'

MAX_HEIGHT = 500
MAX_WIDTH = 500


def resize_image(image):
    height, width, _ = image.shape
    ratio = height / width

    if height > width:
        new_width = round(MAX_HEIGHT / ratio)
        resized_image = cv.resize(image, (new_width, MAX_HEIGHT))
    else:
        new_height = round(MAX_WIDTH * ratio)
        resized_image = cv.resize(image, (MAX_WIDTH, new_height))

    return resized_image

def overlay_transparent(img_front, img_back):
    img_back_copy = img_back.copy()

    # for i in range(img_back_copy.shape[0]):
    #     for j in range(img_back_copy.shape[1]):
    #         if img_front[i][j][3] != 0: # Checking if the alpha(transparent) chanel is not transparent
    #             img_back_copy[i][j] = img_front[i][j][:3]

    nonzero_alpha_indices = np.where(img_front[:, :, 3] != 0)
    img_back_copy[nonzero_alpha_indices[0], nonzero_alpha_indices[1]] = img_front[nonzero_alpha_indices[0], nonzero_alpha_indices[1], :3]
    return img_back_copy

def main():
    cv.namedWindow(WINDOW_NAME)

    output_dir = os.path.join(OUTPUT_BASE)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name_traffic_sign in sorted(os.listdir(INPUT_DIR_TRAFFIC_SIGN)):
        input_path = os.path.join(INPUT_DIR_TRAFFIC_SIGN, name_traffic_sign)
        traffic_sign = cv.imread(input_path)

        resized_traffic_sign = resize_image(traffic_sign)


        for name_cover in sorted(os.listdir(INPUT_DIR_COVER)):
            input_path = os.path.join(INPUT_DIR_COVER, name_cover)
            cover = cv.imread(input_path, -1)

            resized_cover = resize_image(cover)


            resized_cover = cv.resize(resized_cover, (resized_traffic_sign.shape[1], resized_traffic_sign.shape[0]))
            

            overlay = overlay_transparent(resized_cover, resized_traffic_sign)
            cv.imshow(WINDOW_NAME, overlay)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        break

        # cv.imshow(WINDOW_NAME, resized_image_traffic_sign)

        # while True:
        #     if cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE):
        #         key = cv.waitKey(1000) # um segundo
        #     else:
        #         key = ord('q')

        #     if key != -1:
        #         break

        # if key == ord('q'):
        #     break

        # if key == ord('y'):
        #     output_path = os.path.join(output_dir, name)
        #     shutil.copy(input_path, output_path)

    cv.destroyWindow(WINDOW_NAME)
    return 0

if __name__ == '__main__':
    sys.exit(main())