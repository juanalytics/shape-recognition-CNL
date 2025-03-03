import os, random, math
from PIL import Image, ImageDraw
from tqdm import tqdm

# Constants
IMG_SIZE = 640
BASE_SIZE = 50  # minimum base size for shapes
MAX_SCALE = 4.0  # maximum scale factor (up to 4 times base size)
NO_SHAPE_PROB = 0.1  # 10% chance to generate an image with no shapes
MAX_ATTEMPTS = 10  # maximum attempts to generate a valid image

# Define shape classes according to updated .yaml:
# 0: circle, 1: square, 2: triangle, 3: rectangle, 4: ellipse, 5: pentagon, 6: polygon (n-sided, n from 6 to 20)
shape_classes = ["circle", "square", "triangle", "rectangle", "ellipse", "pentagon", "polygon"]

def get_luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b

def random_color(background_color, used_colors=[], min_diff=100):
    # Generate a random color with sufficient contrast versus background and any already-used colors.
    for _ in range(100):
        cand = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if abs(get_luminance(cand) - get_luminance(background_color)) < 100:
            continue
        if any(math.sqrt(sum((cand[i]-uc[i])**2 for i in range(3))) < min_diff for uc in used_colors):
            continue
        return cand
    return cand

def clip_bbox(x_min, y_min, x_max, y_max, img_size=IMG_SIZE):
    x_min = max(0, min(x_min, img_size - 1))
    y_min = max(0, min(y_min, img_size - 1))
    x_max = max(0, min(x_max, img_size - 1))
    y_max = max(0, min(y_max, img_size - 1))
    return x_min, y_min, x_max, y_max

def compute_yolo_label(x_min, y_min, x_max, y_max, img_size=IMG_SIZE):
    box_w = x_max - x_min
    box_h = y_max - y_min
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center / img_size, y_center / img_size, box_w / img_size, box_h / img_size

# Drawing functions for each shape type:
def draw_circle(draw, center, size, color):
    r = size / 2
    x, y = center
    bbox = [x - r, y - r, x + r, y + r]
    draw.ellipse(bbox, fill=color)
    return bbox

def draw_square(draw, center, size, color, rotation=0):
    x, y = center
    half = size / 2
    pts = [(-half, -half), (half, -half), (half, half), (-half, half)]
    sin_t = math.sin(rotation)
    cos_t = math.cos(rotation)
    rotated = [(x + dx * cos_t - dy * sin_t, y + dx * sin_t + dy * cos_t) for dx, dy in pts]
    draw.polygon(rotated, fill=color)
    xs = [pt[0] for pt in rotated]
    ys = [pt[1] for pt in rotated]
    return min(xs), min(ys), max(xs), max(ys)

def draw_triangle(draw, center, size, color, rotation=0):
    x, y = center
    r = size / math.sqrt(3)
    pts = []
    for i in range(3):
        angle = rotation + 2 * math.pi * i / 3
        pts.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    draw.polygon(pts, fill=color)
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    return min(xs), min(ys), max(xs), max(ys)

def draw_rectangle(draw, center, size, color, rotation=0):
    x, y = center
    aspect = random.uniform(0.5, 1.5)
    w, h = size, size * aspect
    half_w, half_h = w / 2, h / 2
    pts = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    sin_t = math.sin(rotation)
    cos_t = math.cos(rotation)
    rotated = [(x + dx * cos_t - dy * sin_t, y + dx * sin_t + dy * cos_t) for dx, dy in pts]
    draw.polygon(rotated, fill=color)
    xs = [pt[0] for pt in rotated]
    ys = [pt[1] for pt in rotated]
    return min(xs), min(ys), max(xs), max(ys)

def draw_ellipse(draw, center, size, color):
    x, y = center
    aspect = random.uniform(0.5, 1.5)
    w, h = size, size * aspect
    bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
    draw.ellipse(bbox, fill=color)
    return bbox

def draw_pentagon(draw, center, size, color, rotation=0):
    return draw_regular_polygon(draw, center, size, 5, color, rotation)

def draw_regular_polygon(draw, center, size, n_sides, color, rotation=0):
    x, y = center
    r = size / 2
    pts = [(x + r * math.cos(rotation + 2 * math.pi * i / n_sides),
            y + r * math.sin(rotation + 2 * math.pi * i / n_sides)) for i in range(n_sides)]
    draw.polygon(pts, fill=color)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def choose_shape_and_draw(draw, center, base_size, shape_type, background_color, used_colors):
    scale = random.uniform(1.0, MAX_SCALE)
    size = base_size * scale
    rotation = random.uniform(0, 2 * math.pi)
    color = random_color(background_color, used_colors)
    used_colors.append(color)
    if shape_type == "circle":
        bbox = draw_circle(draw, center, size, color)
    elif shape_type == "square":
        bbox = draw_square(draw, center, size, color, rotation)
    elif shape_type == "triangle":
        bbox = draw_triangle(draw, center, size, color, rotation)
    elif shape_type == "rectangle":
        bbox = draw_rectangle(draw, center, size, color, rotation)
    elif shape_type == "ellipse":
        bbox = draw_ellipse(draw, center, size, color)
    elif shape_type == "pentagon":
        bbox = draw_pentagon(draw, center, size, color, rotation)
    elif shape_type == "polygon":
        n_sides = random.randint(6, 20)
        bbox = draw_regular_polygon(draw, center, size, n_sides, color, rotation)
    else:
        bbox = draw_circle(draw, center, size, color)
    return bbox, size  # also return size for area computation

def get_random_position_inside(img_size, bbox_width, bbox_height):
    x = random.uniform(bbox_width/2, img_size - bbox_width/2)
    y = random.uniform(bbox_height/2, img_size - bbox_height/2)
    return (x, y)

def get_random_position_partial(img_size, bbox_width, bbox_height):
    edge = random.choice(["left", "right", "top", "bottom"])
    if edge == "left":
        x = random.uniform(-bbox_width/2, bbox_width/2)
        y = random.uniform(bbox_height/2, img_size - bbox_height/2)
    elif edge == "right":
        x = random.uniform(img_size - bbox_width/2, img_size + bbox_width/2)
        y = random.uniform(bbox_height/2, img_size - bbox_height/2)
    elif edge == "top":
        x = random.uniform(bbox_width/2, img_size - bbox_width/2)
        y = random.uniform(-bbox_height/2, bbox_height/2)
    else:  # bottom
        x = random.uniform(bbox_width/2, img_size - bbox_width/2)
        y = random.uniform(img_size - bbox_height/2, img_size + bbox_height/2)
    return (x, y)

def save_sample(img, labels, subset, index):
    base_dir = "Data"
    img_dir = os.path.join(base_dir, "images", subset)
    lbl_dir = os.path.join(base_dir, "labels", subset)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    img_filename = f"{index:05d}.jpg"
    lbl_filename = f"{index:05d}.txt"
    img.save(os.path.join(img_dir, img_filename), "JPEG")
    with open(os.path.join(lbl_dir, lbl_filename), 'w') as f:
        for line in labels:
            f.write(line + "\n")

def generate_image(scenario, base_size):
    """
    Generates an image and its corresponding YOLO labels.
    Scenarios:
      1 - Single Shape fully inside the image.
      2 - Single Shape partially outside (cut-off edges).
      3 - Multiple shapes (2 to 6), where 30-50% are forced to extend off the image.
      4 - Full Training Set mode (mix of scenarios chosen randomly).
      
    This function will discard images where any shape's visible (clipped) area is less than 10% 
    of its computed bounding box area.
    """
    # With some probability, return a blank image (no shapes)
    if random.random() < NO_SHAPE_PROB:
        bg_color = (255, 255, 255)
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color), []
    
    for attempt in range(MAX_ATTEMPTS):
        # Create a new background image.
        if random.random() < 0.2:
            bg_color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        else:
            bg_color = (255, 255, 255)
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color)
        draw = ImageDraw.Draw(img)
        labels = []
        used_colors = []
        valid = True  # flag to check if all shapes are sufficiently visible
        
        if scenario == 4:
            scenario = random.choice([1, 2, 3])
        
        if scenario in [1, 2]:
            # Single shape case; attempt up to MAX_ATTEMPTS
            shape_type = random.choice(shape_classes)
            scale = random.uniform(1.0, MAX_SCALE)
            size = base_size * scale
            bbox_width = size
            bbox_height = size
            if scenario == 1:
                center = get_random_position_inside(IMG_SIZE, bbox_width, bbox_height)
            else:
                center = get_random_position_partial(IMG_SIZE, bbox_width, bbox_height)
            # Draw the shape on the image and get its bounding box
            original_bbox, shape_size = choose_shape_and_draw(draw, center, base_size, shape_type, bg_color, used_colors)
            clipped_bbox = clip_bbox(original_bbox[0], original_bbox[1], original_bbox[2], original_bbox[3])
            area_original = max(0, (original_bbox[2]-original_bbox[0]) * (original_bbox[3]-original_bbox[1]))
            area_clipped = max(0, (clipped_bbox[2]-clipped_bbox[0]) * (clipped_bbox[3]-clipped_bbox[1]))
            if area_original == 0 or (area_clipped/area_original) < 0.1:
                valid = False
            else:
                class_id = shape_classes.index(shape_type)
                x_c, y_c, w_norm, h_norm = compute_yolo_label(clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3])
                labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
        elif scenario == 3:
            n_shapes = random.randint(2, 6)
            for _ in range(n_shapes):
                shape_type = random.choice(shape_classes)
                if random.random() < random.uniform(0.3, 0.5):
                    pos_func = get_random_position_partial
                else:
                    pos_func = get_random_position_inside
                scale = random.uniform(1.0, MAX_SCALE)
                size = base_size * scale
                bbox_width = size
                bbox_height = size
                center = pos_func(IMG_SIZE, bbox_width, bbox_height)
                original_bbox, shape_size = choose_shape_and_draw(draw, center, base_size, shape_type, bg_color, used_colors)
                clipped_bbox = clip_bbox(original_bbox[0], original_bbox[1], original_bbox[2], original_bbox[3])
                area_original = max(0, (original_bbox[2]-original_bbox[0]) * (original_bbox[3]-original_bbox[1]))
                area_clipped = max(0, (clipped_bbox[2]-clipped_bbox[0]) * (clipped_bbox[3]-clipped_bbox[1]))
                # If any shape is less than 10% visible, mark as invalid.
                if area_original == 0 or (area_clipped/area_original) < 0.1:
                    valid = False
                    break
                else:
                    class_id = shape_classes.index(shape_type)
                    x_c, y_c, w_norm, h_norm = compute_yolo_label(clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3])
                    labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
        # If valid image with at least one shape (or no shapes as permitted), return it.
        if valid:
            return img, labels
    # If no valid image is generated after MAX_ATTEMPTS, return a blank image.
    return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255)), []

def main():
    print("Select dataset type:")
    print("  (1) Single Shape (fully inside)")
    print("  (2) Partial Shape (cut-off edges)")
    print("  (3) Multiple Overlapping Shapes")
    print("  (4) Full Training Set (mix)")
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice in {"1", "2", "3", "4"}:
            dataset_type = int(choice)
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    while True:
        num_str = input("How many images would you like to generate? (Default: 2000): ").strip()
        if num_str == "":
            total_images = 2000
            break
        try:
            total_images = int(num_str)
            if total_images > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    # Fixed dataset split: 80% train, 10% validation, 10% test.
    train_count = int(0.8 * total_images)
    val_count = int(0.1 * total_images)
    test_count = total_images - train_count - val_count

    print(f"Generating dataset: {train_count} train, {val_count} validation, {test_count} test images.")
    
    current_index = 1
    for i in tqdm(range(train_count), desc="Generating Train Images"):
        img, labels = generate_image(dataset_type, BASE_SIZE)
        save_sample(img, labels, "train", current_index)
        current_index += 1
    current_index = 1
    for i in tqdm(range(val_count), desc="Generating Validation Images"):
        img, labels = generate_image(dataset_type, BASE_SIZE)
        save_sample(img, labels, "val", current_index)
        current_index += 1
    current_index = 1
    for i in tqdm(range(test_count), desc="Generating Test Images"):
        img, labels = generate_image(dataset_type, BASE_SIZE)
        save_sample(img, labels, "test", current_index)
        current_index += 1

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
