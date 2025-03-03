import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from ultralytics import YOLO

def main():
    # Set a random seed for reproducibility
    torch.manual_seed(42)

    # Load YOLO model
    model = YOLO("models/best2222.pt")

    # Set dataset YAML path dynamically
    dataset_yaml = os.path.abspath("D:/Repos/shape-recognition-CNL/shapes.yaml")

    epochs = 25
    base_batch_size = 8   # Safe starting value for GTX 1050 Ti
    max_batch_size = 32   # Upper limit to test
    best_batch_size = base_batch_size  # Default to start

    # Test for VRAM limitations using FP16 and one-cycle LR (cos_lr=True) with initial frozen layers
    for batch in range(base_batch_size, max_batch_size + 1, 8):
        try:
            print(f"Testing batch size: {batch}")
            model.train(
                data=dataset_yaml,
                epochs=1,
                batch=batch,
                imgsz=640,
                device="cuda",
                half=True,           # Enable FP16 precision
                workers=8,
                cos_lr=True,         # Use One-Cycle LR policy
                lr0=0.005,
                freeze=10,           # Freeze first 10 layers during initial tests
                mosaic=True,         # Enable mosaic augmentation
                mixup=0.0,
                verbose=False
            )
            best_batch_size = batch  # If it works, set as best batch size
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory at batch size {batch}. Using {best_batch_size}.")
                break

    print(f"Starting final training with batch size {best_batch_size}")

    # Final training with optimized settings. Unfreeze all layers by setting freeze=0.
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=best_batch_size,  # Best batch size determined from testing
        imgsz=640,
        device="cuda",
        half=True,              # Enable FP16 precision for speed and lower memory usage
        workers=8,
        lr0=0.005,               # Initial learning rate
        lrf=0.001,              # Final learning rate (for cosine decay)
        optimizer="AdamW",
        momentum=0.937,
        weight_decay=0.0005,
        patience=10,            # Early stopping if no improvement for 10 epochs
        cos_lr=True,            # One-Cycle LR policy
        freeze=0,               # Unfreeze all layers to allow fine-tuning
        mosaic=True,            # Enable mosaic augmentation
        mixup=0.0,
        verbose=True,
        seed=42
    )

    # Run post-training validation automatically.
    print("Starting post-training validation...")
    val_results = model.val()

    # Save best model weights (assuming Ultralytics saves best.pt in runs/detect/train)
    best_model_path = os.path.join("runs", "detect", "train", "best.pt")
    print(f"Training complete. Best model saved at: {best_model_path}")

    # Create a log file for training details.
    log_file = "training_log.txt"
    best_epoch = results.best_epoch if hasattr(results, "best_epoch") else "N/A"
    best_map50 = results.best_map50 if hasattr(results, "best_map50") else "N/A"
    early_stopped = results.early_stop if hasattr(results, "early_stop") else False

    with open(log_file, "w") as f:
        f.write("Training completed.\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best mAP50: {best_map50}\n")
        if early_stopped:
            f.write(f"Early stopping was triggered at epoch {best_epoch} due to no improvement in 10 epochs.\n")
        else:
            f.write("Training completed all epochs.\n")
        f.write("Post-training validation results:\n")
        f.write(str(val_results) + "\n")

    print(f"Training log saved at {log_file}")

if __name__ == "__main__":
    main()
