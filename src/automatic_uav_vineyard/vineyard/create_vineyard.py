#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import xml.etree.ElementTree as ET
from config.load_vineyard_params import load_vineyard_params

# Carica la configurazione dei parametri del vigneto
params = load_vineyard_params()

ground_size_x = params["ground_size_x"]
ground_size_y = params["ground_size_y"]
first_row_x = params["first_row_x"]
first_row_y = params["first_row_y"]
row_length = params["row_length"]
row_spacing = params["row_spacing"]
num_rows = params["num_rows"]
z_poles = params["z_poles"]
pole_radius = params["pole_radius"]
plant_spacing = params["plant_spacing"]
plant_wood_height = params["plant_wood_height"]
home_x = params["home_x"]
home_y = params["home_y"]
home_z = params["home_z"]

#ground_size_x = 200  # meters
#ground_size_y = 200  # meters
#first_row_x = -10.0
#first_row_y = 20.0
#row_length = 20.0
#row_spacing = 2.5
#num_rows = 10
#z_poles = 2.0
#pole_radius = 0.2
#plant_spacing = 0.5
#plant_wood_height = 0.8

# -----------------------------
# SDF ROOT + WORLD
# -----------------------------
sdf = ET.Element("sdf", version="1.9")
world = ET.SubElement(sdf, "world", name="vineyard_world")

# -----------------------------
# Physics and Environment
# -----------------------------
physics = ET.SubElement(world, "physics", type="ode")
ET.SubElement(physics, "max_step_size").text = "0.004"
ET.SubElement(physics, "real_time_factor").text = "1.0"
ET.SubElement(physics, "real_time_update_rate").text = "250"

ET.SubElement(world, "gravity").text = "0 0 -9.81"
ET.SubElement(world, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"
ET.SubElement(world, "atmosphere", type="adiabatic")

# -----------------------------
# Scene and Sun
# -----------------------------
scene = ET.SubElement(world, "scene")
ET.SubElement(scene, "ambient").text = "0.8 0.5 1 1"
ET.SubElement(scene, "grid").text = "false"
sky = ET.SubElement(scene, "sky")
ET.SubElement(sky, "clouds").text = "true"
ET.SubElement(scene, "shadows").text = "true"

light = ET.SubElement(world, "light", name="sun", type="directional")
ET.SubElement(light, "pose").text = "0 0 500 0 0 0"
ET.SubElement(light, "cast_shadows").text = "true"
ET.SubElement(light, "diffuse").text = "1 1 1 1"
ET.SubElement(light, "specular").text = "0.2 0.2 0.2 1"
ET.SubElement(light, "direction").text = "-0.2 0.4 -1"
ET.SubElement(light, "intensity").text = "1.0"

# -----------------------------
# Vineyard Model (inline)
# -----------------------------
model = ET.SubElement(world, "model", name="vineyard")
ET.SubElement(model, "static").text = "true"

link = ET.SubElement(model, "link", name="vineyard_link")

# Ground plane
collision_ground = ET.SubElement(link, "collision", name="ground_collision")
geometry_ground = ET.SubElement(collision_ground, "geometry")
plane_ground = ET.SubElement(geometry_ground, "plane")
ET.SubElement(plane_ground, "normal").text = "0 0 1"
ET.SubElement(plane_ground, "size").text = f"{ground_size_x} {ground_size_y}"

visual_ground = ET.SubElement(link, "visual", name="ground_visual")
geometry_visual = ET.SubElement(visual_ground, "geometry")
plane_visual = ET.SubElement(geometry_visual, "plane")
ET.SubElement(plane_visual, "normal").text = "0 0 1"
ET.SubElement(plane_visual, "size").text = f"{ground_size_x} {ground_size_y}"

material_ground = ET.SubElement(visual_ground, "material")
ET.SubElement(material_ground, "ambient").text = "0.5 1.0 0.5 1"
ET.SubElement(material_ground, "diffuse").text = "0.6 1.0 0.6 1"

# Vineyard poles
for row in range(num_rows):
    x_pos = first_row_x + row * row_spacing
    y_start = first_row_y
    y_end = first_row_y + row_length

    for y_pos in [y_start, y_end]:
        pole_name = f"pole_{row}_{int(y_pos)}"

        # Collision
        collision_pole = ET.SubElement(link, "collision", name=f"{pole_name}_collision")
        geometry_pole = ET.SubElement(collision_pole, "geometry")
        cylinder_pole = ET.SubElement(geometry_pole, "cylinder")
        ET.SubElement(cylinder_pole, "radius").text = str(pole_radius)
        ET.SubElement(cylinder_pole, "length").text = str(z_poles)
        ET.SubElement(collision_pole, "pose").text = f"{x_pos} {y_pos} {z_poles/2} 0 0 0"

        # Visual
        visual_pole = ET.SubElement(link, "visual", name=f"{pole_name}_visual")
        geometry_visual_pole = ET.SubElement(visual_pole, "geometry")
        cylinder_visual_pole = ET.SubElement(geometry_visual_pole, "cylinder")
        ET.SubElement(cylinder_visual_pole, "radius").text = str(pole_radius)
        ET.SubElement(cylinder_visual_pole, "length").text = str(z_poles)
        ET.SubElement(visual_pole, "pose").text = f"{x_pos} {y_pos} {z_poles/2} 0 0 0"

        material_pole = ET.SubElement(visual_pole, "material")
        ET.SubElement(material_pole, "ambient").text = "0.55 0.27 0.07 1"
        ET.SubElement(material_pole, "diffuse").text = "0.65 0.32 0.17 1"

# Vineyard plants + crowns
for row in range(num_rows):
    x_pos = first_row_x + row * row_spacing
    y_start = first_row_y
    y_end = first_row_y + row_length
    num_plants = int(row_length / plant_spacing)

    # Trunks
    for i in range(1, num_plants):
        y_pos = y_start + i * plant_spacing
        plant_name = f"plant_{row}_{i}"

        collision_trunk = ET.SubElement(link, "collision", name=f"{plant_name}_trunk_collision")
        geometry_trunk = ET.SubElement(collision_trunk, "geometry")
        cylinder_trunk = ET.SubElement(geometry_trunk, "cylinder")
        ET.SubElement(cylinder_trunk, "radius").text = "0.05"
        ET.SubElement(cylinder_trunk, "length").text = str(plant_wood_height)
        ET.SubElement(collision_trunk, "pose").text = f"{x_pos} {y_pos} {plant_wood_height/2} 0 0 0"

        visual_trunk = ET.SubElement(link, "visual", name=f"{plant_name}_trunk_visual")
        geometry_visual_trunk = ET.SubElement(visual_trunk, "geometry")
        cylinder_visual_trunk = ET.SubElement(geometry_visual_trunk, "cylinder")
        ET.SubElement(cylinder_visual_trunk, "radius").text = "0.05"
        ET.SubElement(cylinder_visual_trunk, "length").text = str(plant_wood_height)
        ET.SubElement(visual_trunk, "pose").text = f"{x_pos} {y_pos} {plant_wood_height/2} 0 0 0"

        material_trunk = ET.SubElement(visual_trunk, "material")
        ET.SubElement(material_trunk, "ambient").text = "0.4 0.25 0.1 1"
        ET.SubElement(material_trunk, "diffuse").text = "0.5 0.3 0.15 1"

    # Crowns
    crown_height = z_poles - plant_wood_height
    crown_z_center = plant_wood_height + crown_height / 2
    crown_length = row_length - 2 * pole_radius
    crown_y_center = y_start + row_length / 2
    crown_name = f"crown_row_{row}"

    collision_crown = ET.SubElement(link, "collision", name=f"{crown_name}_collision")
    geometry_crown = ET.SubElement(collision_crown, "geometry")
    box_crown = ET.SubElement(geometry_crown, "box")
    ET.SubElement(box_crown, "size").text = f"0.6 {crown_length} {crown_height}"
    ET.SubElement(collision_crown, "pose").text = f"{x_pos} {crown_y_center} {crown_z_center} 0 0 0"

    visual_crown = ET.SubElement(link, "visual", name=f"{crown_name}_visual")
    geometry_visual_crown = ET.SubElement(visual_crown, "geometry")
    box_visual_crown = ET.SubElement(geometry_visual_crown, "box")
    ET.SubElement(box_visual_crown, "size").text = f"0.6 {crown_length} {crown_height}"
    ET.SubElement(visual_crown, "pose").text = f"{x_pos} {crown_y_center} {crown_z_center} 0 0 0"

    material_crown = ET.SubElement(visual_crown, "material")
    ET.SubElement(material_crown, "ambient").text = "0.0 0.25 0.0 1"
    ET.SubElement(material_crown, "diffuse").text = "0.0 0.45 0.0 1"

# -----------------------------
# Aruco tag
# -----------------------------
include = ET.SubElement(world, "include")
ET.SubElement(include, "uri").text = "model://arucotag"
ET.SubElement(include, "pose").text = f"{float(home_x)} {float(home_y)} {float(home_z)} 0 0 0"

# -----------------------------
# Geo Reference
# -----------------------------
spherical_coordinates = ET.SubElement(world, "spherical_coordinates")
ET.SubElement(spherical_coordinates, "surface_model").text = "EARTH_WGS84"
ET.SubElement(spherical_coordinates, "world_frame_orientation").text = "ENU"
ET.SubElement(spherical_coordinates, "latitude_deg").text = "37.412173071650805"
ET.SubElement(spherical_coordinates, "longitude_deg").text = "-121.998878727967"
ET.SubElement(spherical_coordinates, "elevation").text = "38"

# -----------------------------
# Save SDF file
# -----------------------------
tree = ET.ElementTree(sdf)
ET.indent(tree, space="  ", level=0)
tree.write("vineyard_world.sdf", encoding="utf-8", xml_declaration=True)

print("File vineyard_world.sdf generato con successo!")
