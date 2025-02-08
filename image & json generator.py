import random
import math
import json
import os
import json
from PIL import Image, ImageDraw

def random_color(alpha=255):
    """Generate a random RGBA color."""
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            alpha)

def bboxes_intersect(b1, b2):
    """Return True if two bounding boxes b1 and b2 intersect.
       Each bbox is a tuple (xmin, ymin, xmax, ymax)."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

def create_random_image_and_json(image_id):
    # Random image dimensions.
    width = random.randint(400, 800)
    height = random.randint(400, 800)
    
    # Create a random background color and image (RGBA for compositing).
    bg_color = random_color(alpha=255)
    image = Image.new("RGBA", (width, height), bg_color)
    
    # Lists to hold JSON objects and computed final bounding boxes.
    objects = []
    shape_bboxes = []  # Each element is a tuple: (shape_id, final_bbox)
    
    # Determine number of shapes (1 to 10).
    num_shapes = random.randint(1, 10)
    
    # Available shape types.
    shape_types = ["circle", "ellipse", "rectangle", "square", "triangle"]
    
    # We'll assign each shape an incremental id starting at 1.
    current_id = 1

    for _ in range(num_shapes):
        # Create a transparent layer to draw the shape.
        shape_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape_layer)
        
        # Choose a random shape type and color.
        shape_type = random.choice(shape_types)
        shape_color = random_color(alpha=255)
        # Random rotation in degrees.
        rotation = random.randint(0, 360)
        
        # Define a random bounding box for the shape.
        shape_width = random.randint(20, width // 4)
        shape_height = random.randint(20, height // 4)
        x0 = random.randint(0, max(0, width - shape_width))
        y0 = random.randint(0, max(0, height - shape_height))
        x1 = x0 + shape_width
        y1 = y0 + shape_height
        
        # Prepare the JSON data for this shape.
        shape_json = {
            "id": current_id,
            "type": shape_type,
            "color": {
                "red": shape_color[0],
                "green": shape_color[1],
                "blue": shape_color[2]
            }
        }
        
        if shape_type == "circle":
            # For circle, force a square bounding box and determine center and radius.
            diameter = min(shape_width, shape_height)
            x1 = x0 + diameter
            y1 = y0 + diameter
            draw.ellipse([x0, y0, x1, y1], fill=shape_color)
            shape_json["points"] = {
                "center": {"x": x0 + diameter // 2, "y": y0 + diameter // 2},
                "radius": diameter // 2,
                "rotation": rotation
            }
        elif shape_type == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=shape_color)
            shape_json["points"] = {
                "center": {"x": x0 + shape_width // 2, "y": y0 + shape_height // 2},
                "radii": {"width": shape_width // 2, "height": shape_height // 2},
                "rotation": rotation
            }
        elif shape_type == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=shape_color)
            shape_json["points"] = {
                "topleft": {"x": x0, "y": y0},
                "sides": {"width": shape_width, "height": shape_height},
                "rotation": rotation
            }
        elif shape_type == "square":
            # For square, use the smaller side as the side length.
            side = min(shape_width, shape_height)
            x1 = x0 + side
            y1 = y0 + side
            draw.rectangle([x0, y0, x1, y1], fill=shape_color)
            shape_json["points"] = {
                "topleft": {"x": x0, "y": y0},
                "side": side,
                "rotation": rotation
            }
        elif shape_type == "triangle":
            # Generate three random points within the bounding box.
            p1 = {"x": random.randint(x0, x1), "y": random.randint(y0, y1)}
            p2 = {"x": random.randint(x0, x1), "y": random.randint(y0, y1)}
            p3 = {"x": random.randint(x0, x1), "y": random.randint(y0, y1)}
            draw.polygon([(p1["x"], p1["y"]), (p2["x"], p2["y"]), (p3["x"], p3["y"])], fill=shape_color)
            # For triangle, modify the JSON structure to include vertices and rotation.
            shape_json["points"] = {
                "vertices": [p1, p2, p3],
                "rotation": rotation
            }
        
        # Rotate the shape layer; using expand=True so the full rotated shape is captured.
        rotated_layer = shape_layer.rotate(rotation, expand=True)
        # Compute the bounding box (in rotated_layer coordinates) of non-transparent pixels.
        bbox_local = rotated_layer.getbbox()
        # Compute the offset used when pasting the rotated layer on the final image.
        offset_x = (width - rotated_layer.size[0]) // 2
        offset_y = (height - rotated_layer.size[1]) // 2
        # The final bounding box in the final image.
        final_bbox = (bbox_local[0] + offset_x,
                      bbox_local[1] + offset_y,
                      bbox_local[2] + offset_x,
                      bbox_local[3] + offset_y)
        
        # Paste the rotated layer onto a temporary canvas and composite onto the final image.
        temp_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        temp_layer.paste(rotated_layer, (offset_x, offset_y), rotated_layer)
        image = Image.alpha_composite(image, temp_layer)
        
        # Append JSON object and store the final bounding box for relation testing.
        objects.append(shape_json)
        shape_bboxes.append((current_id, final_bbox))
        current_id += 1

    # Generate relations only for overlapping shapes based on their final rotated bounding boxes.
    relations = []
    for i in range(len(shape_bboxes)):
        id_i, bbox_i = shape_bboxes[i]
        for j in range(i + 1, len(shape_bboxes)):
            id_j, bbox_j = shape_bboxes[j]
            if bboxes_intersect(bbox_i, bbox_j):
                # The shape with the lower id is assumed to be drawn earlier (and hence behind).
                relations.append({
                    "obj1": id_i,
                    "obj2": id_j
                })

    # Build the overall scene JSON following the template.
    scene = {
        "size": {"width": width, "height": height},
        "background": {
            "red": bg_color[0],
            "green": bg_color[1],
            "blue": bg_color[2]
        },
        "objects": objects,
        "relations": relations
    }
    
    # Save the image (convert to RGB to remove alpha) and JSON file.

    # Ensure output directories exist
    os.makedirs("generated images", exist_ok=True)
    os.makedirs("generated JSON", exist_ok=True)

    # Convert image to RGB and save as .jpg in the "generated images" folder.
    final_image = image.convert("RGB")
    final_image.save(f"generated images/randimage{image_id}.jpg")
    print(f"Image saved as generated images/randimage{image_id}.jpg")

    # Save the JSON file in the "generated JSON" folder.
    with open(f"generated JSON/randjson{image_id}.json", "w") as json_file:
        json.dump(scene, json_file, indent=4)
    print(f"JSON file saved as generated JSON/randjson{image_id}.json")

        

if __name__ == "__main__":
    images_amount = int(input('Amount of images to generate: '))
    for image_id in range(1,images_amount):
        create_random_image_and_json(image_id)
