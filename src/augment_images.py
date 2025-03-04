import os
import cv2
import numpy as np
from tqdm import tqdm

def add_salt_and_pepper(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Add salt-and-pepper noise uniformly to an image.
    prob: fraction of pixels to be affected (e.g., 0.02 for 2%).
    """
    output = image.copy()
    probs = np.random.rand(output.shape[0], output.shape[1])
    if output.ndim == 2:  # grayscale image
        black_val = 0
        white_val = 255
    else:  # color image (BGR)
        black_val = np.array([0, 0, 0], dtype=np.uint8)
        white_val = np.array([255, 255, 255], dtype=np.uint8)
    output[probs < (prob / 2)] = black_val
    output[probs > 1 - (prob / 2)] = white_val
    return output

def add_partial_salt_and_pepper(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Apply salt-and-pepper noise to a small random region of the image.
    The affected region is kept small (approx. 5-15% of image dimensions)
    to avoid it being misinterpreted as a shape.
    """
    output = image.copy()
    h, w = output.shape[0:2]
    region_w = np.random.randint(int(0.05 * w), int(0.15 * w) + 1)
    region_h = np.random.randint(int(0.05 * h), int(0.15 * h) + 1)
    x1 = np.random.randint(0, w - region_w + 1)
    y1 = np.random.randint(0, h - region_h + 1)
    x2, y2 = x1 + region_w, y1 + region_h
    roi = output[y1:y2, x1:x2]
    roi_noised = add_salt_and_pepper(roi, prob * 0.5)
    output[y1:y2, x1:x2] = roi_noised
    return output

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a color image to full grayscale (3-channel)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_3ch

def apply_partial_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a random region of the image to grayscale, leaving the rest in color."""
    output = image.copy()
    h, w = output.shape[0:2]
    region_w = np.random.randint(int(0.2 * w), int(0.5 * w) + 1)
    region_h = np.random.randint(int(0.2 * h), int(0.5 * h) + 1)
    x1 = np.random.randint(0, w - region_w + 1)
    y1 = np.random.randint(0, h - region_h + 1)
    x2, y2 = x1 + region_w, y1 + region_h
    roi = output[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    output[y1:y2, x1:x2] = gray_roi
    return output

if __name__ == "__main__":
    # Prompt for dataset splits to augment.
    print("Would you like to augment:")
    print("(1) Only the training set")
    print("(2) Training + Validation")
    print("(3) Training + Validation + Test")
    choice = input("Enter your choice (1/2/3): ").strip()
    if choice not in {"1", "2", "3"}:
        print("Invalid choice. Exiting.")
        exit(1)
    choice = int(choice)
    splits_to_augment = ["train"]
    if choice >= 2:
        splits_to_augment.append("val")
    if choice == 3:
        splits_to_augment.append("test")
    
    # Prompt for augmentation parameters.
    perc_input = input("Enter the percentage of images to convert to grayscale (default 10%): ").strip()
    if perc_input == "":
        grayscale_percent = 0.10
    else:
        try:
            val = float(perc_input.strip('%'))
            grayscale_percent = val / 100.0 if val > 1 else val
        except:
            print("Invalid input. Using default 10%.")
            grayscale_percent = 0.10

    noise_input = input("Enter the intensity of salt-and-pepper noise (0-1, default 0.02): ").strip()
    if noise_input == "":
        noise_intensity = 0.02
    else:
        try:
            noise_intensity = float(noise_input)
        except:
            print("Invalid input. Using default 0.02.")
            noise_intensity = 0.02

    grayscale_percent = max(0.0, min(1.0, grayscale_percent))
    noise_intensity = max(0.0, min(1.0, noise_intensity))
    
    # Prompt for folder to augment (the base folder containing dataset splits).
    folder_to_augment = input("Enter the folder to augment (e.g., 'Data/images'): ").strip()
    if folder_to_augment == "":
        folder_to_augment = "Data/images"
    
    print("\nAugmentation settings:")
    sets_names = ", ".join(splits_to_augment)
    print(f" - Dataset splits to augment: {sets_names}")
    print(f" - Percentage of images to convert to grayscale: {grayscale_percent*100:.1f}% (80% full, 20% partial)")
    print(f" - Salt-and-pepper noise intensity: {noise_intensity} (80% uniform, 20% partial)")
    confirm = input("Proceed with these settings? (y/n): ").strip().lower()
    if confirm not in ["y", "yes"]:
        print("Augmentation canceled by user.")
        exit(0)
    
    # Process each selected dataset split. We will overwrite images in the same folder.
    for split in splits_to_augment:
        split_dir = os.path.join(folder_to_augment, split)
        if not os.path.isdir(split_dir):
            print(f"Folder {split_dir} not found. Skipping {split} split.")
            continue
        files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total = len(files)
        if total == 0:
            print(f"No images found in {split_dir}, skipping.")
            continue
        
        print(f"\nAugmenting {split} set: {total} images...")
        for idx in tqdm(range(total), desc=f"Processing {split}", unit="image"):
            filename = files[idx]
            img_path = os.path.join(split_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue
            in_gray = (idx < int(grayscale_percent * total))
            in_noise = (idx < int(grayscale_percent * total))  # For simplicity, same selection as grayscale.
            aug_image = image.copy()
            if in_gray and in_noise:
                if np.random.rand() < 0.2:
                    aug_image = apply_partial_grayscale(aug_image)
                else:
                    aug_image = apply_grayscale(aug_image)
                if np.random.rand() < 0.8:
                    aug_image = add_salt_and_pepper(aug_image, noise_intensity)
                else:
                    aug_image = add_partial_salt_and_pepper(aug_image, noise_intensity)
            elif in_gray:
                if np.random.rand() < 0.2:
                    aug_image = apply_partial_grayscale(aug_image)
                else:
                    aug_image = apply_grayscale(aug_image)
            elif in_noise:
                if np.random.rand() < 0.8:
                    aug_image = add_salt_and_pepper(aug_image, noise_intensity)
                else:
                    aug_image = add_partial_salt_and_pepper(aug_image, noise_intensity)
            # Overwrite the original image.
            cv2.imwrite(img_path, aug_image)
    
    print("\nAugmentation completed successfully!")
