import bpy
from mathutils import Vector
import math
import shutil
import os
import bmesh
import gc
import bpy
import bmesh
import numpy as np
from mathutils import Vector
import glob
import os
import numpy as np
import shutil


# ----- functions -----

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

def render_snapshot(filepath, setTrans=False):
    """
    用Eevee Next渲染引擎渲染并保存图片。默认GPU，速度快。
    """
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = 480
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = filepath
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = setTrans

    eevee = bpy.context.scene.eevee

    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = True
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = True
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = True
    if hasattr(eevee, "taa_samples"):
        eevee.taa_samples = 16


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


def set_background_white():
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (1, 1, 1, 1)  # RGBA纯白
        bg.inputs[1].default_value = 1.0           # 强度
    bpy.context.scene.render.film_transparent = False


def get_paper_view_angles():
    # 36标准视角
    base_angles = [i*10 for i in range(36)]
    # 前方补充（-20到20，每5°，以0°为中心，结果mod 360，去除与base_angles重合的角度）
    front_angles = [(a % 360) for a in range(-20, 25, 5) if (a % 360) not in base_angles]
    # 后方补充（160到200，每5°，以180°为中心，结果mod 360，去除与base_angles重合的角度）
    rear_angles = [(a % 360) for a in range(160, 205, 5) if (a % 360) not in base_angles]
    # 合并、去重
    all_angles = sorted(set(base_angles + front_angles + rear_angles))
    return all_angles

def create_horizontal_circular_cameras(target_obj, distance_factor=1.5):
    """
    严格按论文采样：36标准视角+8前补+8后补（总共44）
    返回cameras, bbox_info
    """
    cameras = []

    # Get world coordinates of bounding box corners
    bbox_corners = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
    min_x, max_x = min(c.x for c in bbox_corners), max(c.x for c in bbox_corners)
    min_y, max_y = min(c.y for c in bbox_corners), max(c.y for c in bbox_corners)
    min_z, max_z = min(c.z for c in bbox_corners), max(c.z for c in bbox_corners)
    # Compute bounding box dimensions
    bbox_width = max_x - min_x
    bbox_depth = max_y - min_y
    bbox_height = max_z - min_z
    bbox_center = Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2))
    bbox_info = [bbox_center, bbox_width, bbox_depth, bbox_height]

    radius = max(bbox_width, bbox_depth) * distance_factor

    # 清理已有Empty
    if "CarLookAt" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["CarLookAt"], do_unlink=True)
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=bbox_center)
    empty = bpy.context.object
    empty.name = "CarLookAt"

    # 使用论文44角度
    angles = get_paper_view_angles()
    for idx, deg in enumerate(angles):
        rad = math.radians(deg)
        x = bbox_center.x + radius * math.cos(rad)
        y = bbox_center.y + radius * math.sin(rad)
        z = bbox_center.z 

        bpy.ops.object.camera_add(location=(x, y, z))
        camera = bpy.context.object

        # Track To "CarLookAt"
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = empty
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        camera["view_angle_deg"] = deg
        cameras.append(camera)

    print(f"✅ {len(cameras)} cameras added according to paper angles: {angles}")
    return cameras, bbox_info



def add_lighting(bbox_info):
    """ Adds a Sun and Area Light above the object using bbox_info. """
    
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
    bpy.ops.object.light_add(type='SUN', location=(bbox_center.x, bbox_center.y, bbox_top + 5))
    sun = bpy.context.object
    sun.data.energy = 1  # Increase brightness
    sun.rotation_euler = (0, 0, 0)  # Directly downward

    # Add an Area Light slightly above for soft shadows
    bpy.ops.object.light_add(type='AREA', location=(bbox_center.x, bbox_center.y, bbox_top))
    area_light = bpy.context.object
    area_light.data.energy = 100  # Increase power
    area_light.data.size = 50  # Large area for softer shadows

    # Increase world background brightness (HDRI effect)
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[1].default_value = 5 # Increase ambient light strength

    print("✅ Lights added successfully above the object!")

# def crop_rendered_images(image_w_cube_path, image_wo_cube_path):



# ----- main -----
def main(model_path, save_path):
    car_model = load_model(model_path)

    car_model = bpy.context.active_object

    if car_model:
        bpy.context.view_layer.update()
        car_model.display_type = 'SOLID'
        car_model.show_bounds = True 
        
        cameras, bbox_info = create_horizontal_circular_cameras(car_model, distance_factor=1.5)

        for i, camera in enumerate(cameras): 
            camera.data.clip_start = 0.01  # Near clipping (adjust if needed)
            camera.data.clip_end = 1000
            bpy.context.scene.camera = camera  # Set camera for rendering
            render_path = f"{save_path}/wo_cube/{str(i).zfill(6)}.png"
            render_snapshot(render_path, setTrans=True)

        # add_lighting(bbox_info)
        color_car_to_red()
        for i, camera in enumerate(cameras):
            camera.data.clip_start = 0.01  # Near clipping (adjust if needed)
            camera.data.clip_end = 1000
            bpy.context.scene.camera = camera  # Set camera for rendering
            render_path = f"{save_path}/w_cube/{str(i).zfill(6)}.png"
            # set_background_white()
            render_snapshot(render_path, setTrans=True)
            
            """ color the transparent area """
            color_trans_area(render_path)
                    
        
                    

if __name__ == "__main__":

    
    car_model_path = "./3d_car_models/2022_peugeot_308.glb"
    save_dir = "./baselines/Park/target_images"


    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    main(car_model_path, save_dir)
        
        
