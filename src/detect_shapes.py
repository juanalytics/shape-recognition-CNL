from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load the YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' is the fastest model

# Load an image
image_path = "Data/raw/220.jpg"  # Replace with your image path
image_path = os.path.abspath(image_path)
image = cv2.imread(image_path)

# Check if the image loaded successfully
if image_path is None:
    raise ValueError(f"❌ Could not load image from {image_path}")
else:
    print(f"✅ Image loaded successfully: {image.shape}")  # (height, width, channels)


# Run YOLO detection
results = model(image)

# Extract bounding boxes
detections = results[0].boxes.data.cpu().numpy()

# Draw bounding boxes on the image
for det in detections:
    x1, y1, x2, y2, conf, cls = det  # Bounding box coordinates
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # Draw rectangle around detected objects
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"Object {int(cls)} ({conf:.2f})"
    
    # Put label
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display the output
cv2.imwrite("yolo_output.jpg", image)
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()