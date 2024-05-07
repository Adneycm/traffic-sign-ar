from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import torch
from PIL import ImageDraw
import cv2 as cv
import numpy as np

# Load the model
model = YOLO('./model/best.pt')  

def resize_image(image, width, height):
    img_pil = Image.fromarray(image)
    img_pil_resized = img_pil.resize((width, height), Image.Resampling.LANCZOS)  # Updated for newer Pillow versions
    return np.array(img_pil_resized)

def overlay_transparent(front_img, back_img, x_offset, y_offset):
    # Get the dimensions of the foreground image
    h, w = front_img.shape[:2]

    # Ensure the foreground image fits within the background image
    if y_offset + h > back_img.shape[0] or x_offset + w > back_img.shape[1]:
        raise ValueError("The overlay image is too large for the background image at given position")

    # Extract the region of interest (ROI) from the background image where the foreground will be placed
    roi = back_img[y_offset:y_offset+h, x_offset:x_offset+w]

    # Extract the alpha mask of the foreground image and create the inverse mask
    alpha_foreground = front_img[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_foreground

    # For each channel (RGB), perform the alpha blending
    for c in range(3):
        roi[:, :, c] = (alpha_foreground * front_img[:, :, c] +
                        alpha_background * roi[:, :, c])

    # Place the modified ROI back into the original image
    back_img[y_offset:y_offset+h, x_offset:x_offset+w] = roi

    return back_img



def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert image to RGB
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize to the input size expected by the model
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path):
    img = load_image(image_path)
    with torch.no_grad():  # Turn off gradient computation
        results = model(img)
    print("Box Data Example:", results[0].boxes.xyxy[0])
    print("Full Boxes Object:", results[0].boxes)  # See if this prints more useful info
    print("Results object keys and types:", {key: type(value) for key, value in results[0].__dict__.items()})
    return results

# def draw_boxes(image, results):
#     draw = ImageDraw.Draw(image)
#     result = results[0]
#     data = result.boxes.data  # This should contain [x1, y1, x2, y2, conf, cls_id]

#     # Iterate over all detections
#     for detection in data:
#         # Extract coordinates and confidence, class directly
#         x1, y1, x2, y2, conf, cls_id = detection
#         x1, y1, x2, y2, conf, cls_id = x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls_id.item()

#         # Get class label using class ID
#         label = f"{result.names[int(cls_id)]} {conf:.2f}"
#         # Draw rectangle and label on the image
#         draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
#         draw.text((x1, y1), label, fill='red')

def draw_boxes_with_overlay(image_path, results, overlay_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    overlay = cv.imread(overlay_path, cv.IMREAD_UNCHANGED)  # Ensure overlay is loaded with alpha

    result = results[0]
    data = result.boxes.data  # Contains [x1, y1, x2, y2, conf, cls_id]

    for detection in data:
        x1, y1, x2, y2 = map(int, detection[:4])
        # Resize overlay image to fit bounding box
        resized_overlay = cv.resize(overlay, (x2-x1, y2-y1), interpolation=cv.INTER_AREA)
        # Overlay image within the bounding box
        image = overlay_transparent(resized_overlay, image, x1, y1)

    cv.imshow('Detected', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('output_with_overlay.jpg', image)


# Example usage
image_path = "./dataset_generation/train_dataset/000000.png"
results = predict(image_path)  # Assuming predict function returns detection results
# image = Image.open(image_path)
# image = draw_boxes(image, results)
# image.show() 
overlay_path = "./TrafficSigns/stop.png"
draw_boxes_with_overlay(image_path, results, overlay_path)