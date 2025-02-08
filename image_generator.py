from PIL import Image, ImageDraw, ImageColor
import random
import math

def random_color(alpha=255):
    """Generate a random RGBA color."""
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            alpha)

def create_random_image():
    # Generate random image dimensions
    width = random.randint(50, 1000)
    height = random.randint(50, 1000)
    
    # Create a background color (opaque)
    bg_color = random_color(alpha=255)
    # Create an RGBA image for alpha compositing later
    image = Image.new("RGBA", (width, height), bg_color)
    
    # Determine random number of shapes (at most 10)
    num_shapes = random.randint(1, 10)
    
    # shape_types = ["circle", "ellipse", "rectangle", "square", "triangle", "polygon"]
    shape_types = ["circle", "ellipse", "rectangle", "square", "triangle"]
    
    for _ in range(num_shapes):
        # Create a transparent layer for the shape
        shape_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape_layer)
        
        shape_type = random.choice(shape_types)
        shape_color = random_color(alpha=255)
        rotation = random.randint(0, 360)
        
        # Define a random bounding box for the shape
        shape_width = random.randint(20, width // 4)
        shape_height = random.randint(20, height // 4)
        x0 = random.randint(0, max(0, width - shape_width))
        y0 = random.randint(0, max(0, height - shape_height))
        x1 = x0 + shape_width
        y1 = y0 + shape_height
        
        if shape_type == "circle":
            # For a circle, use a square bounding box (diameter)
            diameter = min(shape_width, shape_height)
            x1 = x0 + diameter
            y1 = y0 + diameter
            draw.ellipse([x0, y0, x1, y1], fill=shape_color)
        elif shape_type == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=shape_color)
        elif shape_type == "rectangle":
            draw.rectangle([x0, y0, x1, y1], fill=shape_color)
        elif shape_type == "square":
            side = min(shape_width, shape_height)
            x1 = x0 + side
            y1 = y0 + side
            draw.rectangle([x0, y0, x1, y1], fill=shape_color)
        elif shape_type == "triangle":
            # Generate three random points within the bounding box
            p1 = (random.randint(x0, x1), random.randint(y0, y1))
            p2 = (random.randint(x0, x1), random.randint(y0, y1))
            p3 = (random.randint(x0, x1), random.randint(y0, y1))
            draw.polygon([p1, p2, p3], fill=shape_color)
        # elif shape_type == "polygon":
        #     # Generate a polygon with a random number of sides (between 5 and 8)
        #     num_sides = random.randint(5, 8)
        #     points = [(random.randint(x0, x1), random.randint(y0, y1))
        #               for _ in range(num_sides)]
        #     draw.polygon(points, fill=shape_color)
        
        # Rotate the shape layer (expand=True to accommodate the whole rotated image)
        rotated_layer = shape_layer.rotate(rotation, expand=False)
        
        
        # Composite the rotated shape onto the main image
        image = Image.alpha_composite(image, rotated_layer)

    
    # Convert to RGB and save the image
    final_image = image.convert("RGB")
    final_image.save("random_image.png")
    print("Image saved as random_image.png")

if __name__ == "__main__":
    create_random_image()
