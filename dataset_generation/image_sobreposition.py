import cv2 as cv
import numpy as np
import shutil
import os
from PIL import Image
import sys
from PIL import Image as PILImage

INPUT_DIR_TRAFFIC_SIGN = 'dataset_generation/labeled/traffic_signs'
INPUT_DIR_COVER = 'dataset_generation/labeled/tree_branchs'

OUTPUT_BASE = 'dataset_generation/train_dataset'
WINDOW_NAME = 'traffic_signs_covered? (y/*)'

MAX_HEIGHT = 500
MAX_WIDTH = 500


def resize_image(image):
    img_pil = PILImage.fromarray(image)
    img_pil_rgba = img_pil.convert("RGBA")  # Convert to RGBA mode to handle alpha channel
    img_pil_resized = img_pil_rgba.resize((MAX_WIDTH, MAX_HEIGHT), PILImage.ANTIALIAS)
    resized_image = np.array(img_pil_resized)
    return resized_image

def overlay_transparent(img_front, img_back):
    # Ensure the front image has 4 channels (RGBA)
    if img_front.shape[2] == 3:
        img_front = np.concatenate([img_front, np.ones((img_front.shape[0], img_front.shape[1], 1), dtype=img_front.dtype) * 255], axis=-1)

    # Resize the front image to match the size of the back image
    img_front_pil = Image.fromarray(img_front).resize((img_back.shape[1], img_back.shape[0]), Image.ANTIALIAS).convert("RGBA")
    img_back_pil = Image.fromarray(img_back).convert("RGBA")

    # Composite the images
    combined = Image.alpha_composite(img_back_pil, img_front_pil)

    # Convert back to array and drop alpha channel for display purposes
    overlay = np.array(combined)[:, :, :3]
    return overlay

def main():
    output_img_number = 51
    rotation_step = 10

    output_dir = os.path.join(OUTPUT_BASE)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_traffic_signs = len(os.listdir(INPUT_DIR_TRAFFIC_SIGN))
    sign_counter = 0

    for name_traffic_sign in sorted(os.listdir(INPUT_DIR_TRAFFIC_SIGN)):
        sign_counter += 1
        input_path = os.path.join(INPUT_DIR_TRAFFIC_SIGN, name_traffic_sign)
        traffic_sign = cv.imread(input_path)
        resized_traffic_sign = resize_image(traffic_sign)
        print(f"Processing traffic sign {sign_counter} of {total_traffic_signs}: {name_traffic_sign}")

        total_covers = len(os.listdir(INPUT_DIR_COVER))
        cover_counter = 0

        for name_cover in sorted(os.listdir(INPUT_DIR_COVER)):
            cover_counter += 1
            input_path = os.path.join(INPUT_DIR_COVER, name_cover)
            cover = cv.imread(input_path, -1)
            resized_cover = resize_image(cover)
            resized_cover = cv.resize(resized_cover, (resized_traffic_sign.shape[1], resized_traffic_sign.shape[0]))
            print(f"  Applying cover {cover_counter} of {total_covers}: {name_cover}")

            for rotation_angle in range(0, 360, rotation_step):
                rotated_cover = Image.fromarray(resized_cover)
                rotated_cover = rotated_cover.rotate(rotation_angle, expand=True)
                rotated_cover = np.array(rotated_cover)
                overlay = overlay_transparent(rotated_cover, resized_traffic_sign)

                # Save the image automatically
                output_path = os.path.join(output_dir, f"{output_img_number:06d}.png")
                cv.imwrite(output_path, overlay)
                output_img_number += 1
                print(f"    Saved rotated image at {rotation_angle} degrees as {output_img_number:06d}.png")

    print("All images processed and saved.")
    return 0

if __name__ == '__main__':
    sys.exit(main())