import cv2
import os
import random
from ultralytics import YOLO

# Constants
MODEL_PATH = "models/yolov8n.pt"
IMAGE_FOLDER = os.path.join("Data", "raw")
NUM_IMAGES = 10

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

def use_yolo(image_path: str) -> None:
    """Process an image with YOLO detection, draw bounding boxes, and display the result."""
    if not os.path.exists(image_path):
        print(f"Warning: '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image '{image_path}'.")
        return

    # Run YOLO detection
    results = model(image)

    # Check if detections exist
    if not results or not results[0].boxes.data.size:
        print(f"No detections for '{image_path}'.")
        return

    # Extract bounding boxes and draw them
    detections = results[0].boxes.data.cpu().numpy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Object {int(cls)} ({conf:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result with the image name as window title
    window_title = os.path.basename(image_path)
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)

def main():
    # Choose a random start index ensuring we have a range
    start_index = random.randint(1, 50-NUM_IMAGES)
    for img_id in range(start_index, start_index + NUM_IMAGES):
        image_path = os.path.join(IMAGE_FOLDER, f"{img_id}.jpg")
        use_yolo(os.path.abspath(image_path))

if __name__ == "__main__":
    main()
