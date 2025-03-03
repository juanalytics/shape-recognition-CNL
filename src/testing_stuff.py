from ultralytics import YOLO

# Constants
MODEL_PATH = "yolov8n.pt"

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

for name in model.names:
    print(name, model.names[name])


# Count total layers
num_layers = len(list(model.model.modules()))

print(f"🔍 YOLOv8n has {num_layers} layers.")