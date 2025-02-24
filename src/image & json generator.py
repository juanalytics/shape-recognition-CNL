import random
import math
import json
import os
from PIL import Image, ImageDraw

def random_color(alpha=255):
    """
    Genera un color aleatorio (RGBA) con las siguientes condiciones:
      - Con 2% de probabilidad devuelve negro puro (0,0,0)
      - Con 2% de probabilidad devuelve blanco puro (255,255,255)
      - En el resto, genera colores en los que la diferencia (max - min) entre 
        los canales R, G y B sea lo suficientemente grande.
    """
    r = random.random()
    if r < 0.02:
        return (0, 0, 0, alpha)
    elif r < 0.04:
        return (255, 255, 255, alpha)
    while True:
        r_val = random.randint(0, 255)
        g_val = random.randint(0, 255)
        b_val = random.randint(0, 255)
        # Evitar pure black y pure white.
        if (r_val, g_val, b_val) == (0, 0, 0) or (r_val, g_val, b_val) == (255, 255, 255):
            continue
        diff = max(r_val, g_val, b_val) - min(r_val, g_val, b_val)
        if diff >= 100:
            return (r_val, g_val, b_val, alpha)

def bboxes_intersect(b1, b2):
    """Retorna True si dos bounding boxes se intersectan.
       Cada bbox es una tupla (xmin, ymin, xmax, ymax)."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

def create_random_image_and_json(image_id):
    # Dimensiones aleatorias de la imagen.
    width = random.randint(100, 500)
    height = random.randint(100, 500)
    
    # Crear un color de fondo aleatorio y la imagen (RGBA para composición).
    r = random.random()
    if r <= 0.3:
        bg_color = (255, 255, 255, 255)
    else:
        bg_color = random_color(alpha=255)
    image = Image.new("RGBA", (width, height), bg_color)
    
    # Listas para almacenar los objetos JSON y los bounding boxes finales de las formas.
    objects = []
    shape_bboxes = []  # Cada elemento es una tupla: (shape_id, final_bbox)
    
    # Número de formas (entre 1 y 7).
    num_shapes = random.randint(1, 7)
    
    # Tipos de formas disponibles.
    shape_types = ["circle", "ellipse", "rectangle", "square", "triangle", "polygon"]
    
    # Asignaremos a cada forma un id incremental a partir de 1.
    current_id = 1

    for _ in range(num_shapes):
        # Crear una capa transparente para dibujar la forma.
        shape_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape_layer)
        
        # Elegir un tipo de forma aleatorio.
        shape_type = random.choice(shape_types)
        
        # Definir un bounding box aleatorio para la forma.
        shape_width = random.randint(20, width // 2)
        shape_height = random.randint(20, height // 2)
        x0 = random.randint(0, max(0, width - shape_width))
        y0 = random.randint(0, max(0, height - shape_height))
        x1 = x0 + shape_width
        y1 = y0 + shape_height

        # Antes de elegir el color de la forma, obtener el color subyacente en la imagen actual.
        center_point = ((x0 + x1) // 2, (y0 + y1) // 2)
        underlying_color = image.getpixel(center_point)[:3]  # Solo R, G, B
        
        # Escoger un color para la forma que sea suficientemente distinto del subyacente.
        attempts = 0
        while True:
            shape_color = random_color(alpha=255)
            diff = max(abs(shape_color[i] - underlying_color[i]) for i in range(3))
            if diff >= 50:
                break
            attempts += 1
            if attempts > 10:
                break
        
        # Rotación aleatoria en grados.
        rotation = random.randint(0, 360)
        
        # Preparar los datos JSON para la forma.
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
            # Para círculo, forzar un bounding box cuadrado y determinar centro y radio.
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
            # Para cuadrado, usar el lado menor.
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
            # Generar tres puntos aleatorios dentro del bounding box, asegurando ángulos internos entre 10° y 170°.
            def dist(a, b):
                return math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)
            def compute_angle(a, b, c):
                return math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))
            max_attempts = 100
            for attempt in range(max_attempts):
                p1 = {"x": random.randint(x0, x1), "y": random.randint(y0, y1)}
                p2 = {"x": random.randint(x0, x1), "y": random.randint(y0, y1)}
                p3 = {"x": random.randint(x0, x1), "y": random.randint(y0, y1)}
                d12 = dist(p1, p2)
                d23 = dist(p2, p3)
                d31 = dist(p3, p1)
                if d12 == 0 or d23 == 0 or d31 == 0:
                    continue
                try:
                    angle1 = compute_angle(d12, d31, d23)
                    angle2 = compute_angle(d12, d23, d31)
                    angle3 = compute_angle(d31, d23, d12)
                except ValueError:
                    continue
                if 10 <= angle1 <= 170 and 10 <= angle2 <= 170 and 10 <= angle3 <= 170:
                    break
            draw.polygon([(p1["x"], p1["y"]), (p2["x"], p2["y"]), (p3["x"], p3["y"])], fill=shape_color)
            shape_json["points"] = {
                "vertices": [p1, p2, p3],
                "rotation": rotation
            }
        elif shape_type == "polygon":
            # Generar un polígono regular.
            # Elegir número de lados entre 5 y 8.
            num_sides = random.randint(5, 8)
            # Usar el centro del bounding box y un radio basado en la dimensión menor.
            center_x = x0 + shape_width // 2
            center_y = y0 + shape_height // 2
            radius = min(shape_width, shape_height) // 2
            offset_angle = random.uniform(0, 2 * math.pi)
            vertices = []
            for i in range(num_sides):
                angle = offset_angle + i * (2 * math.pi / num_sides)
                vx = int(center_x + radius * math.cos(angle))
                vy = int(center_y + radius * math.sin(angle))
                vertices.append({"x": vx, "y": vy})
            draw.polygon([(v["x"], v["y"]) for v in vertices], fill=shape_color)
            # Registrar los vértices en el JSON; se incluye la rotación para mantener consistencia.
            shape_json["points"] = {
                "vertices": vertices,
                "rotation": rotation
            }
        
        # Rotar la capa de la forma (expand=True para capturar la forma completa).
        rotated_layer = shape_layer.rotate(rotation, expand=True)
        # Obtener el bounding box (en coordenadas de la capa rotada) de los píxeles no transparentes.
        bbox_local = rotated_layer.getbbox()
        # Calcular el offset al pegar la capa rotada sobre la imagen final.
        offset_x_layer = (width - rotated_layer.size[0]) // 2
        offset_y_layer = (height - rotated_layer.size[1]) // 2
        # Bounding box final en la imagen.
        final_bbox = (bbox_local[0] + offset_x_layer,
                      bbox_local[1] + offset_y_layer,
                      bbox_local[2] + offset_x_layer,
                      bbox_local[3] + offset_y_layer)
        
        # Pegar la capa rotada en un lienzo temporal y componer sobre la imagen final.
        temp_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        temp_layer.paste(rotated_layer, (offset_x_layer, offset_y_layer), rotated_layer)
        image = Image.alpha_composite(image, temp_layer)
        
        # Agregar el objeto JSON y almacenar el bounding box final para las relaciones.
        objects.append(shape_json)
        shape_bboxes.append((current_id, final_bbox))
        current_id += 1

    # Generar relaciones solo para formas que se superponen, omitiendo aquellas en que una está eclipsada en más del 95%.
    relations = []
    for i in range(len(shape_bboxes)):
        id_i, bbox_i = shape_bboxes[i]
        area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
        for j in range(i + 1, len(shape_bboxes)):
            id_j, bbox_j = shape_bboxes[j]
            inter_left = max(bbox_i[0], bbox_j[0])
            inter_top = max(bbox_i[1], bbox_j[1])
            inter_right = min(bbox_i[2], bbox_j[2])
            inter_bottom = min(bbox_i[3], bbox_j[3])
            inter_width = max(0, inter_right - inter_left)
            inter_height = max(0, inter_bottom - inter_top)
            inter_area = inter_width * inter_height
            if inter_area == 0:
                continue
            area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
            small_area = min(area_i, area_j)
            if small_area > 0 and (inter_area / small_area) > 0.95:
                continue
            if bboxes_intersect(bbox_i, bbox_j):
                relations.append({
                    "obj1": id_j,
                    "obj2": id_i
                })

    # Construir el JSON de la escena según la plantilla.
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
    
    # Carpetas para las imagenes
    img_location= "./data/raw"
    json_location= "./data/raw"


    # Asegurarse de que existan los directorios de salida.
    os.makedirs(img_location, exist_ok=True)
    os.makedirs(json_location, exist_ok=True)

    # Guardar la imagen (convertir a RGB para eliminar el canal alfa) y el archivo JSON.
    final_image = image.convert("RGB")
    final_image.save(f"{img_location}/{image_id}.jpg")
    print(f"Image saved as {img_location}/{image_id}.jpg")
    with open(f"{json_location}/{image_id}.json", "w") as json_file:
        json.dump(scene, json_file, indent=4)
    print(f"JSON file saved as {json_location}/{image_id}.json")

if __name__ == "__main__":
    images_amount = int(input('Amount of images to generate: '))
    for image_id in range(1, images_amount + 1):
        create_random_image_and_json(image_id)
