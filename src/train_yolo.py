from ultralytics import YOLO
import os

# Load pretrained YOLOv8 model (fine-tuning from COCO)
model = YOLO("yolov8n.pt")  # Use "yolov8n.yaml" for training from scratch

for i, param in enumerate(model.model.parameters()):
    if i < 50:  # Freeze first 50 layers
        param.requires_grad = False

# Path to dataset YAML file
dataset_yaml = "shapes.yaml"

# Training parameters
epochs = 50
batch_size = 8
image_size = 640

model.train(
    data=dataset_yaml,
    epochs=epochs,
    batch=batch_size,
    imgsz=image_size,
    save=True,  # Automatically saves best model
    project="runs/train",  # Output directory
    name="shapes_finetune"  # Experiment name
)

results = model.val(data=dataset_yaml, split="test")
print(results)

# Save the best model manually (optional)
model.export(format="pt")  # Saves as best.pt

# Load trained model for further testing
trained_model = YOLO("runs/train/shapes_finetune/weights/best.pt")
