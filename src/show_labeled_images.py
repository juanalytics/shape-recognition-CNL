import os
import cv2
import random

def main():
    # Prompt user to select dataset split
    print("Select dataset split to visualize:")
    print("(1) Training Set")
    print("(2) Validation Set")
    print("(3) Test Set")
    choice = input("Enter choice: ").strip()
    if choice not in {'1', '2', '3'}:
        print("Invalid choice. Defaulting to Training Set.")
        choice = '1'
    split_map = {'1': 'train', '2': 'val', '3': 'test'}
    split_name = split_map[choice]

    # Set directory paths for images and labels
    images_dir = os.path.join("Data", "images", split_name)
    labels_dir = os.path.join("Data", "labels", split_name)
    if not os.path.isdir(images_dir):
        print(f"Image directory not found: {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        print(f"Label directory not found: {labels_dir}")
        return

    # Collect all image file names in the selected images directory
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Ask user for viewing mode: sequential or random
    mode = input("View images sequentially or randomly? (s/r): ").strip().lower()
    if mode not in {'s', 'r'}:
        print("Invalid choice. Defaulting to sequential mode.")
        mode = 's'

    # Determine the list of images to display
    images_to_show = []
    if mode == 'r':
        # Random sampling mode
        num_str = input(f"How many images would you like to sample from the '{split_name}' set? ")
        try:
            num_samples = int(num_str)
        except ValueError:
            print("Invalid number. Defaulting to 5 images.")
            num_samples = 5
        if num_samples <= 0:
            print("Number of images must be positive. Defaulting to 5.")
            num_samples = 5
        total_images = len(image_files)
        if num_samples > total_images:
            print(f"Requested {num_samples} images, but only {total_images} available. Showing all images.")
            num_samples = total_images
        images_to_show = random.sample(image_files, num_samples)
    else:
        # Sequential mode (show all images in sorted order for consistency)
        images_to_show = sorted(image_files)

    # Define distinct colors for up to several classes (BGR format)
    colors = [
        (0, 255, 0),      # green
        (0, 0, 255),      # red
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (128, 128, 128),  # gray
        (255, 128, 0),    # orange
        (128, 0, 255),    # purple
        (0, 128, 255)     # another distinct color
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Loop through selected images and display each with drawn bounding boxes
    total = len(images_to_show)
    for idx, img_file in enumerate(images_to_show, start=1):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_file}. Skipping...")
            continue

        # Construct the corresponding label file path
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        if not os.path.isfile(label_path):
            print(f"No label file for {img_file}. Skipping...")
            continue

        # Read label file lines
        with open(label_path, 'r') as lf:
            lines = [line.strip() for line in lf if line.strip()]

        # Get image dimensions
        img_h, img_w = img.shape[0], img.shape[1]

        # Draw each bounding box from the label file
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                # Malformed line (not 5 values) – skip it
                continue
            class_id_str, x_center_str, y_center_str, w_str, h_str = parts
            try:
                class_id = int(class_id_str)
                x_center = float(x_center_str) * img_w
                y_center = float(y_center_str) * img_h
                w = float(w_str) * img_w
                h = float(h_str) * img_h
            except ValueError:
                # Skip this line if values are not parseable
                continue

            # Convert center, width, height to top-left (x1,y1) and bottom-right (x2,y2)
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            # Pick a color for this class (index into the colors list)
            color = colors[class_id % len(colors)]
            # Draw the rectangle (bounding box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

            # Prepare text label (class ID) with a background rectangle for visibility
            label_text = str(class_id)
            text_size, baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            # Position for text background and text (above the top-left corner of the box if possible)
            text_bg_x1 = x1
            text_bg_y2 = y1  # bottom of text background is the top of the box
            text_bg_y1 = y1 - text_h - 4  # top of text background (with 4px padding)
            if text_bg_y1 < 0:
                # If box is at the top edge, draw text background below the box instead
                text_bg_y1 = y1
                text_bg_y2 = y1 + text_h + 4
            text_bg_x2 = x1 + text_w + 6  # width of background (2px padding on each side)
            # Draw filled rectangle for text background
            cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, thickness=cv2.FILLED)
            # Determine text color (black or white) for contrast
            r, g, b = color[2], color[1], color[0]  # convert BGR to RGB for luminance calc
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
            # Calculate text position (slightly inside the background rectangle)
            text_x = x1 + 3
            text_y = text_bg_y2 - 3 if text_bg_y1 >= y1 else (y1 + text_h + 1)
            cv2.putText(img, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        # Show progress if in random mode or just index if sequential
        print(f"Showing image {idx}/{total}: {img_file}")
        cv2.imshow("YOLO Labels Viewer", img)
        # Wait for a key press to move to next image (0 means indefinitely)
        key = cv2.waitKey(0) & 0xFF
        # If 'q' or ESC pressed, break out early
        if key == 27 or key == ord('q'):
            print("Visualization stopped by user.")
            break
        # Close the image window (so the next image will open fresh)
        cv2.destroyAllWindows()

    # Cleanup: close any remaining windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
