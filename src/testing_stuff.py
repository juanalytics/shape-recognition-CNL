from ultralytics import YOLO
import torch

# Constants
MODEL_PATH = "yolov8n.pt"

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

for name in model.names:
    print(name, model.names[name])


# Count total layers
num_layers = len(list(model.model.modules()))

print(f"🔍 YOLOv8n has {num_layers} layers.")
print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))
print("Allocated Memory:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
print("Reserved Memory:", torch.cuda.memory_reserved(0) / 1024**2, "MB")