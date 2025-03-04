import os, random, math
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# Constants
IMG_SIZE = 640
BASE_SIZE = 50          # Minimum base size for shapes
NO_SHAPE_PROB = 0.1     # 10% chance to generate an image with no shapes (regardless of scenario)
MAX_ATTEMPTS = 10       # Maximum attempts to generate a valid image
VISIBILITY_THRESHOLD = 0.1  # At least 10% of the shape's area must be visible

# Default parameters (can be overridden by user input)
DEFAULT_MAX_ASPECT = 5.0   # Default maximum aspect ratio (e.g., 1:5)
EXTREME_MAX_ASPECT = 10.0  # Extreme maximum aspect ratio (up to 1:10) with 30% chance
QUAD_RATIO = 0.5           # Proportion of squares & rectangles vs. other shapes
MAX_SCALE = 4.0            # Maximum scale factor (up to 4 times BASE_SIZE)
MAX_SHAPES = 7             # Maximum number of shapes per image for scenario 3

# Define shape classes (do not add new ones):
# 0: circle, 1: square, 2: triangle, 3: rectangle, 4: ellipse, 5: pentagon, 6: polygon
shape_classes = ["circle", "square", "triangle", "rectangle", "ellipse", "pentagon", "polygon"]

def get_luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b

def random_color(background_color, used_colors=[], min_diff=100):
    for _ in range(100):
        cand = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if abs(get_luminance(cand) - get_luminance(background_color)) < 100:
            continue
        if any(math.sqrt(sum((cand[i] - uc[i])**2 for i in range(3))) < min_diff for uc in used_colors):
            continue
import os, random, math
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# Constants
IMG_SIZE = 640
BASE_SIZE = 50          # Minimum base size for shapes
NO_SHAPE_PROB = 0.1     # 10% chance to generate an image with no shapes (regardless of scenario)
MAX_ATTEMPTS = 10       # Maximum attempts to generate a valid image
VISIBILITY_THRESHOLD = 0.05  # At least 5% of the shape's area must be visible

# Default parameters (can be overridden by user input)
DEFAULT_MAX_ASPECT = 5.0   # Default maximum aspect ratio (e.g., 1:5)
EXTREME_MAX_ASPECT = 10.0  # Extreme maximum aspect ratio (up to 1:10) with 30% chance
QUAD_RATIO = 0.5           # Proportion of squares & rectangles vs. other shapes
MAX_SCALE = 4.0            # Maximum scale factor (up to 4 times BASE_SIZE)
MAX_SHAPES = 7             # Maximum number of shapes per image for scenario 3

# Define shape classes (do not add new ones):
# 0: circle, 1: square, 2: triangle, 3: rectangle, 4: ellipse, 5: pentagon, 6: polygon
shape_classes = ["circle", "square", "triangle", "rectangle", "ellipse", "pentagon", "polygon"]

def get_luminance(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b

def random_color(background_color, used_colors=[], min_diff=100):
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
    if random.random() < 0.3:
        aspect = random.uniform(1.0, EXTREME_MAX_ASPECT)
    else:
        aspect = random.uniform(1.0, DEFAULT_MAX_ASPECT)
    if random.random() < 0.5:
        w, h = size * aspect, size
    else:
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
    if random.random() < 0.3:
        aspect = random.uniform(1.0, EXTREME_MAX_ASPECT)
    else:
        aspect = random.uniform(1.0, DEFAULT_MAX_ASPECT)
    if random.random() < 0.5:
        w, h = size * aspect, size
    else:
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
    return bbox, size

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
    else:
        x = random.uniform(bbox_width/2, img_size - bbox_width/2)
        y = random.uniform(img_size - bbox_height/2, img_size + bbox_height/2)
    return (x, y)

def save_sample(img, labels, subset, index, dataset_version):
    base_dir = os.path.join("datasets", dataset_version)
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
      3 - Multiple shapes, where the number of shapes is uniformly chosen between 1 and MAX_SHAPES.
      4 - Full Training Set mode (mix of scenarios chosen randomly).
    
    If any generated shape fails the visibility check (less than 5% visible), it is discarded.
    """
    # With some probability, return a blank image (no shapes)
    if random.random() < NO_SHAPE_PROB:
        bg_color = (255, 255, 255)
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color), []
    
    for attempt in range(MAX_ATTEMPTS):
        if random.random() < 0.2:
            bg_color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        else:
            bg_color = (255, 255, 255)
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg_color)
        draw = ImageDraw.Draw(img)
        labels = []
        used_colors = []
        valid = True
        
        # Use customizable shape proportions:
        quad_shapes = ["square", "rectangle"]
        other_shapes = [s for s in shape_classes if s not in quad_shapes]
        def choose_shape():
            if random.random() < QUAD_RATIO:
                return random.choice(quad_shapes)
            else:
                return random.choice(other_shapes)
        
        if scenario == 4:
            scenario = random.choice([1, 2, 3])
        
        if scenario in [1, 2]:
            shape_type = choose_shape()
            scale = random.uniform(1.0, MAX_SCALE)
            size = base_size * scale
            bbox_width = size
            bbox_height = size
            if scenario == 1:
                center = get_random_position_inside(IMG_SIZE, bbox_width, bbox_height)
            else:
                center = get_random_position_partial(IMG_SIZE, bbox_width, bbox_height)
            original_bbox, shape_size = choose_shape_and_draw(draw, center, base_size, shape_type, bg_color, used_colors)
            clipped_bbox = clip_bbox(original_bbox[0], original_bbox[1], original_bbox[2], original_bbox[3])
            area_original = max(0, (original_bbox[2]-original_bbox[0]) * (original_bbox[3]-original_bbox[1]))
            area_clipped = max(0, (clipped_bbox[2]-clipped_bbox[0]) * (clipped_bbox[3]-clipped_bbox[1]))
            if area_original == 0 or (area_clipped/area_original) < VISIBILITY_THRESHOLD:
                valid = False
            else:
                class_id = shape_classes.index(shape_type)
                x_c, y_c, w_norm, h_norm = compute_yolo_label(clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3])
                labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
        elif scenario == 3:
            # Choose number of shapes uniformly between 1 and MAX_SHAPES.
            n_shapes = random.randint(1, MAX_SHAPES)
            for _ in range(n_shapes):
                shape_type = choose_shape()
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
                if area_original == 0 or (area_clipped/area_original) < VISIBILITY_THRESHOLD:
                    # Discard this shape (i.e. do not add a label) but continue with others.
                    continue
                else:
                    class_id = shape_classes.index(shape_type)
                    x_c, y_c, w_norm, h_norm = compute_yolo_label(clipped_bbox[0], clipped_bbox[1], clipped_bbox[2], clipped_bbox[3])
                    labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
        if valid:
            return img, labels
    return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255)), []

def main():
    try:
        dma_input = input("Enter default maximum aspect ratio for rectangles/ellipses (default 5.0): ").strip()
        default_max_aspect = float(dma_input) if dma_input != "" else 5.0
    except ValueError:
        default_max_aspect = 5.0

    try:
        ema_input = input("Enter extreme maximum aspect ratio for rectangles/ellipses (default 10.0): ").strip()
        extreme_max_aspect = float(ema_input) if ema_input != "" else 10.0
    except ValueError:
        extreme_max_aspect = 10.0

    try:
        quad_input = input("Enter proportion of squares & rectangles (0 to 1, default 0.5): ").strip()
        quad_ratio = float(quad_input) if quad_input != "" else 0.5
    except ValueError:
        quad_ratio = 0.5

    try:
        ms_input = input("Enter maximum scale factor for shapes (default 4.0): ").strip()
        max_scale_input = float(ms_input) if ms_input != "" else 4.0
    except ValueError:
        max_scale_input = 4.0

    try:
        max_shapes_input = input("Enter maximum number of shapes per image (default 7): ").strip()
        max_shapes = int(max_shapes_input) if max_shapes_input != "" else 7
    except ValueError:
        max_shapes = 7

    try:
        version_input = input("Enter dataset version name (default 'version_1'): ").strip()
        dataset_version = version_input if version_input != "" else "version_1"
    except Exception:
        dataset_version = "version_1"

    global DEFAULT_MAX_ASPECT, EXTREME_MAX_ASPECT, QUAD_RATIO, MAX_SCALE, MAX_SHAPES
    DEFAULT_MAX_ASPECT = default_max_aspect
    EXTREME_MAX_ASPECT = extreme_max_aspect
    QUAD_RATIO = quad_ratio
    MAX_SCALE = max_scale_input
    MAX_SHAPES = max_shapes

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
    
    train_count = int(0.8 * total_images)
    val_count = int(0.1 * total_images)
    test_count = total_images - train_count - val_count

    print(f"Generating dataset version '{dataset_version}': {train_count} train, {val_count} validation, {test_count} test images.")
    
    current_index = 1
    for i in tqdm(range(train_count), desc="Generating Train Images"):
        img, labels = generate_image(dataset_type, BASE_SIZE)
        save_sample(img, labels, "train", current_index, dataset_version)
        current_index += 1
    current_index = 1
    for i in tqdm(range(val_count), desc="Generating Validation Images"):
        img, labels = generate_image(dataset_type, BASE_SIZE)
        save_sample(img, labels, "val", current_index, dataset_version)
        current_index += 1
    current_index = 1
    for i in tqdm(range(test_count), desc="Generating Test Images"):
        img, labels = generate_image(dataset_type, BASE_SIZE)
        save_sample(img, labels, "test", current_index, dataset_version)
        current_index += 1

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
