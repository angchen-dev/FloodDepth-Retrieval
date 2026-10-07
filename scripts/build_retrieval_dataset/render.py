#!/usr/bin/env python3
"""Render flooded 3D vehicles from the paper's retrieval viewpoints."""

import argparse
import gc
import math
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def color_car_to_red():
    """ Finds the object named 'car' and assigns or creates a red material. """
    
    # Find the object named 'car'
    car_obj = bpy.data.objects.get("car")
    
    if car_obj is None or car_obj.type != 'MESH':
        print("⚠ No valid mesh object named 'car' found.")
        return

    mat_name = "RedMaterial"

    # Check if material already exists
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        # Create new red material
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)  # Red
            bsdf.inputs["Roughness"].default_value = 0.5  # Adjust roughness
            bsdf.inputs["Metallic"].default_value = 0.1  # Slight metallic look
        print(f"🎨 Created new red material: {mat_name}")

    # Remove all existing materials from the object
    car_obj.data.materials.clear()
    
    # Assign the new material
    car_obj.data.materials.append(mat)
    print(f"✅ Assigned {mat_name} to {car_obj.name}")

def prepare_context_for_join():
    # Ensure the context is in Object Mode and clean up the selection.
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    # Select all mesh objects and set one as the active object
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    for obj in mesh_objects:
        obj.select_set(True)
    if mesh_objects:
        bpy.context.view_layer.objects.active = mesh_objects[0]

    # Return True if there are at least two mesh objects to join
    return len(mesh_objects) > 1

def create_flood_cube(obj, z_ratio):
    if obj is None:
        return

    # Get world coordinates of bounding box corners (axis-aligned bounding box).
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x, max_x = min(c.x for c in bbox_corners), max(c.x for c in bbox_corners)
    min_y, max_y = min(c.y for c in bbox_corners), max(c.y for c in bbox_corners)
    min_z, max_z = min(c.z for c in bbox_corners), max(c.z for c in bbox_corners)

    
    # Compute bounding box dimensions
    bbox_width = max_x - min_x
    bbox_depth = max_y - min_y
    bbox_height = max_z - min_z
    

    # Compute center (X, Y remain unchanged, Z stays the same)
    bbox_center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    
    bbox_info = [bbox_center, bbox_width, bbox_depth, bbox_height]

    # Create a cube that represents the full bounding box
    xy_factor = 3
    
    
    # create bbox cube 
    bpy.ops.mesh.primitive_cube_add(size=1, location=bbox_center)
    bbox_cube = bpy.context.object
    bbox_cube.scale = (bbox_width, bbox_depth, bbox_height)
    bbox_cube.location.z = min_z + (bbox_height / 2)
    bbox_cube.hide_viewport = True
    bbox_cube.hide_render = True
    bbox_cube.hide_select = True
    
    # create flood cube
    bpy.ops.mesh.primitive_cube_add(size=1, location=bbox_center)
    flood_cube = bpy.context.object
    flood_cube.scale = (bbox_width*xy_factor, bbox_depth*xy_factor, bbox_height*z_ratio)
    flood_cube.location.z = min_z + (bbox_height*z_ratio / 2)
    
    mat = bpy.data.materials.new(name="FloodCubeMaterial")
    mat.use_nodes = True  # Use shader nodes for better shading
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0, 1, 0, 1)  # Opaque Green
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Alpha"].default_value = 1.0

    # Assign the material to the cube
    if flood_cube.data.materials:
        flood_cube.data.materials[0] = mat
        bbox_cube.data.materials[0] = mat
    else:
        flood_cube.data.materials.append(mat)
        bbox_cube.data.materials.append(mat)
    
    bbox_cube.display_type = 'WIRE'
    flood_cube.display_type = 'WIRE'
    flood_depth = min_z + bbox_height*z_ratio
    
    return bbox_cube, flood_cube, flood_depth, bbox_info

def create_upper_hemisphere_cameras(target_obj, bbox_info, L=5, tau=0.8, 
                        radius_factor=3, cam_num_layer=4):
                            
    """ Creates cameras following the upper hemisphere sampling method with inversed Fibonacci sequence. """

    if target_obj is None or bbox_info is None:
        print("⚠ No target object or bounding box info provided for camera placement.")
        return []

    # Unpack bbox_info
    bbox_center, bbox_width, bbox_depth, bbox_height = bbox_info

    # Compute hemisphere radius based on object size and distance factor
    radius = math.sqrt(bbox_width ** 2 + bbox_depth ** 2) / 2 * radius_factor

    cam_num_sequence = [cam_num_layer] * L
    print(f"✅ Camera numbers of each layer: {cam_num_sequence}")

    cameras = []

    for l in range(1, L + 1):
        num_cameras = cam_num_sequence[l - 1]  # Inverted Fibonacci sequence
        elevation_angle = (tau * math.pi / (2 * L)) * (l-1)  # φ(k,l)

        for k in range(1, num_cameras + 1):
            azimuth_angle_unit = 2 * math.pi / num_cameras
            azimuth_angle = azimuth_angle_unit * (k - 1) + (azimuth_angle_unit/L)*(l-1)  # θ(k,l)


            # Convert spherical to Cartesian coordinates
            x = bbox_center.x + radius * math.cos(elevation_angle) * math.cos(azimuth_angle)
            y = bbox_center.y + radius * math.cos(elevation_angle) * math.sin(azimuth_angle)
            z = bbox_center.z + radius * math.sin(elevation_angle)

            # Add camera at computed position
            bpy.ops.object.camera_add(location=(x, y, z))
            camera = bpy.context.object

            # Add a "Track To" constraint so the camera faces the object center
            constraint = camera.constraints.new(type='TRACK_TO')
            constraint.target = target_obj
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            constraint.up_axis = 'UP_Y'

            cameras.append(camera)

    print(f"✅ {len(cameras)} cameras added successfully using upper hemisphere sampling!")
    return cameras

def filter_cameras_by_flood_depth(cameras, flood_depth):
    """ Removes cameras whose Z position is below the given flood depth. """
    
    filtered_cameras = []
    
    for camera in cameras:
        if camera.location.z >= flood_depth:
            filtered_cameras.append(camera)
        else:
            # Remove the camera from the scene if it's below flood depth
            bpy.data.objects.remove(camera, do_unlink=True)

    print(f"✅ {len(filtered_cameras)} cameras remain after filtering (above flood depth: {flood_depth}).")
    
    return filtered_cameras

def enable_gpu_rendering():
    # Detect the active render engine
    engine = bpy.context.scene.render.engine

    # Enable GPU rendering for Cycles
    if engine == 'CYCLES':
        preferences = bpy.context.preferences.addons['cycles'].preferences

        # Select GPU compute type based on system
        if bpy.app.build_options.cuda:
            preferences.compute_device_type = 'CUDA'  # NVIDIA CUDA
        elif bpy.app.build_options.optix:
            preferences.compute_device_type = 'OPTIX'  # NVIDIA OptiX (RTX GPUs)
        elif bpy.app.build_options.opencl:
            preferences.compute_device_type = 'OPENCL'  # AMD OpenCL
        elif bpy.app.build_options.metal:
            preferences.compute_device_type = 'METAL'  # Apple Metal (MacOS)

        bpy.context.scene.cycles.device = 'GPU'

        # Enable all available GPUs
        preferences.get_devices()
        for device in preferences.devices:
            device.use = True  # Enable all detected GPUs

        print(f"✅ GPU Rendering enabled for Cycles ({preferences.compute_device_type})")

    # Enable GPU for Eevee (if supported)
    elif engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.use_gtao = True  # Enable Ambient Occlusion
        bpy.context.scene.eevee.use_bloom = True  # Enable Bloom
        bpy.context.scene.eevee.use_ssr = True  # Enable Screen Space Reflections
        bpy.context.scene.eevee.taa_samples = 16  # Reduce samples for speed
        print("✅ Eevee rendering optimized for performance")

    # No GPU settings needed for Workbench
    elif engine == 'BLENDER_WORKBENCH':
        print("⚠ Workbench engine does not use GPU acceleration.")

    else:
        print(f"⚠ Unknown render engine: {engine}")
        
def add_lighting(bbox_info, hdri_path, addHDRLight):
    if addHDRLight:
        # Switch to 'WORLD' context
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world

        # Use nodes for the world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)

        # Create nodes
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        output_node.location = (300, 0)

        background_node = nodes.new(type='ShaderNodeBackground')
        background_node.location = (0, 0)

        env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
        env_tex_node.image = bpy.data.images.load(hdri_path)
        env_tex_node.location = (-300, 0)

        # Link nodes
        links.new(env_tex_node.outputs['Color'], background_node.inputs['Color'])
        links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

        # Optional: Adjust HDRI strength
        background_node.inputs['Strength'].default_value = 1.0

        print("HDRI environment set to:", hdri_path)
    else:
    
        if bbox_info is None:
            print("⚠ No bounding box info provided for lighting.")
            return
        
        # Unpack bbox_info
        bbox_center, bbox_width, bbox_depth, bbox_height = bbox_info

        # Compute light positions (slightly above the object)
        bbox_top = bbox_center.z + bbox_height   # Slightly above the top

        # Delete existing lights
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)

        # Add a Sun Light directly above the object
        # bpy.ops.object.light_add(type='SUN', location=(bbox_center.x+500, bbox_center.y+500, 0))
        # sun = bpy.context.object
        # sun.data.energy = 1  # Increase brightness
        # sun.rotation_euler = (0, 0, 0)  # Directly downward

        # Add an Area Light slightly above for soft shadows
        bpy.ops.object.light_add(type='AREA', location=(bbox_center.x, bbox_center.y, bbox_top+5))
        area_light = bpy.context.object
        area_light.data.energy = 300  # Increase power
        area_light.data.size = 300  # Large area for softer shadows
        area_light.rotation_euler = (0, 0, 0)  # Directly downward

        # Increase world background brightness (HDRI effect)
        bpy.context.scene.world.use_nodes = True
        bg = bpy.context.scene.world.node_tree.nodes["Background"]
        bg.inputs[1].default_value = 5 # Increase ambient light strength

        print("✅ Lights added successfully above the object!")

def remove_all_lights():
    """ Removes all light sources from the Blender scene. """
    
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Collect all light objects
    lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
    
    if not lights:
        print("⚠ No lights found in the scene.")
        return

    # Remove all light objects
    for light in lights:
        bpy.data.objects.remove(light, do_unlink=True)

    print(f"✅ Removed {len(lights)} lights from the scene.")

def render_snapshot(filepath, mode, setTrans=False):
    """ Renders a snapshot using specified rendering engine with green background. """

    # General settings for both Cycles and Eevee
    bpy.context.scene.render.resolution_x = 480  # Width
    bpy.context.scene.render.resolution_y = 480  # Height
    bpy.context.scene.render.resolution_percentage = 100  # Render at full resolution
    bpy.context.scene.render.filepath = filepath
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = setTrans

    if mode == 'CYCLES':
        bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles render engine
        bpy.context.scene.cycles.device = 'GPU'  # Ensure GPU is used

        # Optimize rendering settings
        bpy.context.scene.cycles.samples = 64  # Reduce samples for speed
        bpy.context.scene.cycles.use_adaptive_sampling = True  # Enable adaptive sampling
        bpy.context.scene.cycles.use_denoising = True  # Enable AI denoising
        bpy.ops.render.render(write_still=True)

    elif mode == 'BLENDER_EEVEE':
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # Use Eevee (fastest)
        bpy.ops.render.render(write_still=True)

def load_model(model_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_scene.gltf(filepath=model_path)
    
    if prepare_context_for_join():
        bpy.ops.object.join()
        print("Objects joined successfully.")
    else:
        print("Not enough mesh objects to join or only one object imported.")

    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':  # Ensure we rename only mesh objects
            obj.name = "car"
        
    car = bpy.data.objects.get("car")

    return car

def color_trans_area(render_path):
    image = bpy.data.images.load(render_path)
    image.colorspace_settings.name = 'sRGB'
    image.alpha_mode = 'STRAIGHT'
    width, height = image.size
    pixels = np.array(image.pixels[:])  # Blender stores pixels in a 1D array (R, G, B, A)
    pixels = pixels.reshape((height, width, 4))
    transparent_mask = pixels[:, :, 3] == 0
    pixels[transparent_mask] = [0, 1, 0, 1]  # Green in normalized format
    image.pixels[:] = pixels.flatten()
    image.file_format = 'PNG'
    image.filepath_raw = render_path
    image.save()

# ----- main -----
def main(model_path, save_path, z_ratio, layer_num, cam_num_layer, hdri_path, addHDRLight):
    car_model = load_model(model_path)
    car_model = bpy.context.active_object


    if car_model:
        # car_model.scale *= 100  # scale up 100× (adjust as needed)
        bpy.context.view_layer.update()  # refresh the scene after transform
        car_model.display_type = 'SOLID'
        car_model.show_bounds = True 
        
        # Create a bounding box cube
        bbox_cube, flood_cube, flood_depth, bbox_info = create_flood_cube(car_model, z_ratio)
        cameras = create_upper_hemisphere_cameras(
            bbox_cube, bbox_info, L=layer_num, tau=0.8, radius_factor=3, cam_num_layer=cam_num_layer)
        cameras = filter_cameras_by_flood_depth(cameras, flood_depth)
        
        
        # Init rendering
        enable_gpu_rendering()

        # Render images without flood cube
        add_lighting(bbox_info, hdri_path, addHDRLight)
        for i, camera in enumerate(cameras):  
            camera.data.clip_start = 0.01  # Near clipping (adjust if needed)
            camera.data.clip_end = 1000
            bpy.context.scene.camera = camera  # Set camera for rendering
            render_path = f"{save_path}/wo_cube/{str(i).zfill(6)}.png"
            flood_cube.hide_render = True
            render_snapshot(render_path, mode='BLENDER_EEVEE', setTrans=True)

        # Render images with flood cube
        remove_all_lights()
        color_car_to_red()
        for i, camera in enumerate(cameras):
            camera.data.clip_start = 0.01  # Near clipping (adjust if needed)
            camera.data.clip_end = 1000
            bpy.context.scene.camera = camera  # Set camera for rendering
            flood_cube.hide_render = False
            render_path = f"{save_path}/w_cube/{str(i).zfill(6)}.png"
            render_snapshot(render_path, mode='BLENDER_EEVEE', setTrans=True)

            """ color the transparent area """
            color_trans_area(render_path)

  
                    
DEFAULT_MODELS = (
    "2022_peugeot_308.glb",
)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Render flooded vehicles for the retrieval dataset."
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--hdri-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--cameras-per-layer", type=int, default=72)
    parser.add_argument("--depth-levels", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--without-hdr-light", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_rendering(args):
    if args.layers <= 0 or args.cameras_per_layer <= 0 or args.depth_levels <= 0:
        raise ValueError("layers, cameras-per-layer, and depth-levels must be positive")
    models_dir = args.models_dir.expanduser().resolve()
    hdri_path = args.hdri_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(f"3D model directory not found: {models_dir}")
    if not args.without_hdr_light and not hdri_path.is_file():
        raise FileNotFoundError(f"HDR environment not found: {hdri_path}")

    missing = [name for name in args.models if not (models_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing 3D models: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        model_path = models_dir / model_name
        model_stem = model_path.stem
        print(f"--- {model_stem} ---")
        for depth_index in range(args.depth_levels):
            depth_ratio = round(depth_index / args.depth_levels, 6)
            save_path = output_dir / model_stem / str(depth_ratio)
            if save_path.exists():
                if not args.overwrite:
                    print(f"Skipping existing render directory: {save_path}")
                    continue
                shutil.rmtree(save_path)
            (save_path / "w_cube").mkdir(parents=True)
            (save_path / "wo_cube").mkdir(parents=True)
            main(
                str(model_path),
                str(save_path),
                depth_ratio,
                args.layers,
                args.cameras_per_layer,
                str(hdri_path),
                not args.without_hdr_light,
            )
            gc.collect()


if __name__ == "__main__":
    run_rendering(parse_args())
