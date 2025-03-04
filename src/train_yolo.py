import os
import torch
import traceback
from datetime import datetime
from ultralytics import YOLO

def setup_environment():
    """Sets up the environment variables and dynamically selects the best GPU."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if torch.cuda.is_available():
        # Select GPU with the most free memory
        gpu_memory = [(i, torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
        best_gpu = min(gpu_memory, key=lambda x: x[1])[0]  # Select GPU with least used memory
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        device = f"cuda:{best_gpu}"
        print(f"✅ Using GPU {best_gpu} ({torch.cuda.get_device_name(best_gpu)})")
    else:
        print("⚠ No CUDA-compatible GPU found. Running on CPU.")
        device = "cpu"

    return device

def determine_batch_size():
    # """Sets batch size based on available VRAM."""
    # if torch.cuda.is_available():
    #     total_memory = torch.cuda.get_device_properties(0).total_memory
    # else:
    #     total_memory = 0

    # if total_memory > 12e9:  # GPUs with more than 12GB VRAM
    #     return 64
    # elif total_memory > 8e9:  # If GPU has more than 8GB VRAM
    #     return 32
    # elif total_memory > 4e9:  # If GPU has more than 4GB VRAM
    #     return 16
    # else:  # Low VRAM GPUs or CPU
    #     return 8
    return 32

def train_yolo():
    """Trains YOLO with optimized settings."""
    device = setup_environment()
    torch.manual_seed(42)

    model_path = "models/best3.pt"
    dataset_yaml = os.path.abspath("D:/Repos/shape-recognition-CNL/shapes.yaml")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Check path and try again.")
        return

    print(f"🔍 Loading model from: {model_path}")
    model = YOLO(model_path)

    epochs = 100  # Set high, early stopping will handle it
    best_batch_size = determine_batch_size()

    print(f"🚀 Starting training with batch size: {best_batch_size} on {device}")

    try:
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=best_batch_size,
            imgsz=640,
            device=device,
            half=True if "cuda" in device else False,  # Use FP16 only if on GPU
            workers=8,
            lr0=0.003,  # Lower LR for stability
            lrf=0.0005,  # Final LR for gradual refinement
            optimizer="AdamW",
            momentum=0.937,
            weight_decay=0.0005,
            patience=15,  # Early stopping
            cos_lr=True,
            freeze=0,  # Unfreeze all layers
            mosaic=True,
            mixup=0.1,
            verbose=True,
            seed=42
        )

        # Run validation
        print("📊 Running post-training validation...")
        val_results = model.val()

        # Save training logs
        save_training_log(results, val_results)

        print(f"✅ Training complete. Best model saved at: {results.save_dir}")

    except Exception as e:
        error_message = f"❌ Training failed due to error: {str(e)}"
        print(error_message)
        with open("training_error_log.txt", "w") as f:
            f.write(error_message + "\n")
            f.write(traceback.format_exc())  # Log full traceback

def save_training_log(results, val_results):
    """Logs training results into a timestamped file to prevent overwriting."""
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"training_log_{log_timestamp}.txt"

    best_epoch = getattr(results, "best_epoch", "N/A")
    best_map50 = getattr(val_results, "mAP_50", "N/A")  # Fix: Correctly extract validation mAP
    early_stopped = getattr(results, "early_stop", False)

    with open(log_file, "w") as f:
        f.write(f"Training completed on {log_timestamp}.\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best mAP50: {best_map50}\n")
        if early_stopped:
            f.write(f"⚠ Early stopping triggered at epoch {best_epoch} due to no improvement in 15 epochs.\n")
        else:
            f.write("✅ Training completed all planned epochs.\n")
        f.write("🔍 Post-training validation results:\n")
        f.write(str(val_results) + "\n")

    print(f"📂 Training log saved at: {log_file}")

if __name__ == "__main__":
    train_yolo()
