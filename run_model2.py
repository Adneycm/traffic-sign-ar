from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import torch
from PIL import ImageDraw
import cv2 as cv
import numpy as np

# Load the model
model = YOLO('./model/best.pt')  


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


def draw_boxes(image_path, results):
    original_image = Image.open(image_path)
    draw = ImageDraw.Draw(original_image)
    width, height = original_image.size  # Original dimensions
    result = results[0]
    data = result.boxes.data

    for detection in data:
        x1, y1, x2, y2, conf, cls_id = detection
        if conf >= 0.3:
            # Scale bounding box back to original image size
            x1 = x1.item() * width / 416
            y1 = y1.item() * height / 416
            x2 = x2.item() * width / 416
            y2 = y2.item() * height / 416

            label = f"{result.names[int(cls_id)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1), label, fill='red')

    # Convert PIL image to a format suitable for OpenCV
    np_image = np.array(original_image)
    np_image = cv.cvtColor(np_image, cv.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    cv.imshow('Detected', np_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# # Usage example
# image_path = './dataset_generation/test/santoAmaro.png'
# #image_path = './dataset_generation/train_dataset/000038.jpg'
# results = predict(image_path)
# #image = Image.open(image_path)
# draw_boxes(image_path, results)


#-----------------------------------------------------------------------------------------------------#
#                                                                                                     #
#                               AQUI FAZ OVERLAY COM A PLACA PARE                                     # 
##                                                                                                    #
#-----------------------------------------------------------------------------------------------------

def resize_image(image, width, height):
    img_pil = Image.fromarray(image)
    img_pil_resized = img_pil.resize((width, height), Image.Resampling.LANCZOS)  # Updated for newer Pillow versions
    return np.array(img_pil_resized)

def overlay_transparent(front_img, back_img, x_offset, y_offset):
    h, w = front_img.shape[:2]
    if y_offset + h > back_img.shape[0] or x_offset + w > back_img.shape[1]:
        raise ValueError("The overlay image is too large for the background image at given position")
    roi = back_img[y_offset:y_offset+h, x_offset:x_offset+w]
    alpha_foreground = front_img[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_foreground
    for c in range(3):
        roi[:, :, c] = (alpha_foreground * front_img[:, :, c] +
                        alpha_background * roi[:, :, c])
    back_img[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    return back_img

def draw_boxes_with_overlay(image_path, results, overlay_path):
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    original_height, original_width = image.shape[:2]
    overlay = cv.imread(overlay_path, cv.IMREAD_UNCHANGED)
    result = results[0]
    data = result.boxes.data
    for detection in data:
        x1, y1, x2, y2, conf, cls_id = detection
        if conf >= 0.3:  # Check if the detection confidence is high enough
            # Scale bounding box to original dimensions
            x1 = int(x1 * original_width / 416)
            x2 = int(x2 * original_width / 416)
            y1 = int(y1 * original_height / 416)
            y2 = int(y2 * original_height / 416)

            # Resize overlay image to fit bounding box
            resized_overlay = cv.resize(overlay, (x2 - x1, y2 - y1), interpolation=cv.INTER_AREA)
            # Overlay image within the bounding box
            image = overlay_transparent(resized_overlay, image, x1, y1)

    cv.imshow('Detected and overlay', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('output_with_overlay.jpg', image)

# # Example usage
# #image_path = "./dataset_generation/train_dataset/000000.png"
# results = predict(image_path)
# overlay_path = "./TrafficSigns/stop.png"
# draw_boxes_with_overlay(image_path, results, overlay_path)


#-----------------------------------------------------------------------------------------------------#
#                                          VÍDEO                                                      #
#                               AQUI FAZ OVERLAY COM A PLACA PARE                                     # 
##                                                                                                    #
#-----------------------------------------------------------------------------------------------------

def draw_boxes_with_overlay(frame, results, overlay_path):
    original_height, original_width = frame.shape[:2]
    overlay = cv.imread(overlay_path, cv.IMREAD_UNCHANGED)
    result = results[0]
    data = result.boxes.data
    for detection in data:
        x1, y1, x2, y2, conf, cls_id = detection
        if conf >= 0.3:  # Check if the detection confidence is high enough
            print("--------------------------")
            print("A box was detected")
            print("x1:", x1, "conf:", conf)
            print("--------------------------")
            # Scale bounding box to original dimensions
            x1 = int(x1 * original_width / 416)
            x2 = int(x2 * original_width / 416)
            y1 = int(y1 * original_height / 416)
            y2 = int(y2 * original_height / 416)

            # Resize overlay image to fit bounding box
            resized_overlay = cv.resize(overlay, (x2 - x1, y2 - y1), interpolation=cv.INTER_AREA)
            # Overlay image within the bounding box
            frame = overlay_transparent(resized_overlay, frame, x1, y1)
    return frame

def load_image_from_array(frame):
    # Convert OpenCV frame (BGR) to a PIL Image (RGB)
    image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize to the input size expected by the model
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(frame):
    img = load_image_from_array(frame)
    with torch.no_grad():  # Turn off gradient computation
        results = model(img)
    return results

def video_processing(video_path, overlay_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or cannot fetch the frame.")
            break
        results = predict(frame)
        frame_with_overlay = draw_boxes_with_overlay(frame, results, overlay_path)
        
        cv.imshow('Video with Detection', frame_with_overlay)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv.destroyAllWindows()

# Example usage for a video file
video_path = './dataset_generation/test/videos/gaivota2.mp4'
overlay_path = "./TrafficSigns/stop.png"
video_processing(video_path, overlay_path)