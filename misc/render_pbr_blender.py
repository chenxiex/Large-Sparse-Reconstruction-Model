# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import math
import os
import sys

# pyre-fixme[21]: Could not find module `bpy`.
import bpy
import numpy as np

# pyre-fixme[21]: Could not find module `mathutils`.
from mathutils import Matrix, Vector


def parse_vector(value):
    return ast.literal_eval(value)


def clean_blender_scene():
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Select all objects in the scene
    bpy.ops.object.select_all(action="SELECT")

    # Delete selected objects
    bpy.ops.object.delete()

    # Remove all collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    # Remove all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # Remove all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    # Remove all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

    # Remove all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

    # Remove all curves
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)

    # Remove all cameras
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)

    # Remove all lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Remove all armatures
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)

    # Remove all actions (animations)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)

    # Remove all node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)


def get_blender_version():
    """Returns the major version of Blender as an integer."""
    return bpy.app.version[0]


def import_obj(obj_path, rotate_y_to_z=False):
    # Import the .obj file into the scene with Z-up orientation
    if get_blender_version() >= 4:
        # Blender 4.x uses wm.obj_import with different parameter names
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    else:
        # Blender 3.x uses import_scene.obj
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="Y", axis_up="Z")
    # Get the imported object (assuming it's the active object after import)
    objs = bpy.context.selected_objects
    if rotate_y_to_z:
        for obj in objs:
            obj.rotation_euler[0] = math.radians(90)
    return objs


def create_material(name="NewMaterial"):
    """Creates a new material with a Principled BSDF shader."""
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    return material


def add_image_texture_node(nodes, image_path, location=(0, 0), colorspace="COLOR"):
    """Adds an image texture node with a specific image and color space setting."""
    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.location = location
    texture_node.image = bpy.data.images.load(image_path)
    texture_node.image.colorspace_settings.name = colorspace
    return texture_node


def add_texture_coordinate_and_mapping_nodes(nodes, links):
    """Adds Texture Coordinate and Mapping nodes, links UV to Mapping, and returns Mapping node."""
    # Texture Coordinate node
    tex_coord_node = nodes.new(type="ShaderNodeTexCoord")
    tex_coord_node.location = (-800, 0)

    # Mapping node
    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-600, 0)

    # Link Texture Coordinate UV output to Mapping input
    links.new(tex_coord_node.outputs["UV"], mapping_node.inputs["Vector"])

    return mapping_node


def link_texture_to_material(links, texture_node, shader_input):
    """Links an image texture node to a specific input on the Principled BSDF shader."""
    links.new(texture_node.outputs["Color"], shader_input)


def get_node(material, type="BSDF_PRINDIPLED"):
    # Check if the material and its node tree exist
    if material and material.use_nodes:
        # Get the node tree of the material
        nodes = material.node_tree.nodes

        # Find the Principled BSDF node
        result_node = None
        for node in nodes:
            if node.type == type:
                result_node = node
                return result_node
    return None


def setup_material(
    objs, albedo_path, roughness_path, metallic_path, srgb_specularity=False
):
    for obj in objs:
        """Sets up a Principled BSDF material with roughness, and metallic textures."""
        # Get material from the object, or create one if missing
        material = obj.active_material
        if material is None:
            material = bpy.data.materials.new(name=obj.name + "_Material")
            obj.data.materials.append(material)
        if not material.use_nodes:
            material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        # Create Principled BSDF node and Material Output node
        bsdf_node = get_node(material, "BSDF_PRINCIPLED")
        # Add Texture Coordinate and Mapping nodes
        mapping_node = add_texture_coordinate_and_mapping_nodes(nodes, links)

        # link Base Color texture to uv mapping
        basecolor_node = add_image_texture_node(
            nodes, albedo_path, location=(-300, -800), colorspace="sRGB"
        )
        link_texture_to_material(links, basecolor_node, bsdf_node.inputs["Base Color"])
        links.new(mapping_node.outputs["Vector"], basecolor_node.inputs["Vector"])

        # Add and link Roughness texture
        spec_colorspace = "sRGB" if srgb_specularity else "Non-Color"
        roughness_node = add_image_texture_node(
            nodes, roughness_path, location=(-300, -400), colorspace=spec_colorspace
        )
        link_texture_to_material(links, roughness_node, bsdf_node.inputs["Roughness"])
        links.new(mapping_node.outputs["Vector"], roughness_node.inputs["Vector"])

        # Add and link Metallic texture
        metallic_node = add_image_texture_node(
            nodes, metallic_path, location=(-300, 0), colorspace=spec_colorspace
        )
        link_texture_to_material(links, metallic_node, bsdf_node.inputs["Metallic"])
        links.new(mapping_node.outputs["Vector"], metallic_node.inputs["Vector"])


# set up environment map
def setup_envmap(envmap_path, envmap_rotation=(0.0, 0.0, 0.0), envmap_intensity=1.0):
    scene = bpy.context.scene

    # Get the environment node tree of the current scene
    if scene.world is None:
        # create a new world
        new_world = bpy.data.worlds.new("new_world")
        new_world.use_nodes = True
        scene.world = new_world

    node_tree = scene.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add nodes
    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_texturecoordinate = tree_nodes.new(type="ShaderNodeTexCoord")
    node_mapping = tree_nodes.new(type="ShaderNodeMapping")
    node_environment = tree_nodes.new("ShaderNodeTexEnvironment")
    node_output = tree_nodes.new(type="ShaderNodeOutputWorld")

    # Position nodes
    node_texturecoordinate.location = -700, 0
    node_mapping.location = -500, 0
    node_environment.location = -300, 0
    node_output.location = 200, 0

    # Load HDRI/EXR image
    node_environment.image = bpy.data.images.load(envmap_path)

    # Set map rotation
    node_mapping.inputs["Rotation"].default_value = envmap_rotation

    # Set light intensity
    node_background.inputs[1].default_value = envmap_intensity

    # Link all nodes
    links = node_tree.links
    links.new(
        node_texturecoordinate.outputs["Generated"], node_mapping.inputs["Vector"]
    )
    links.new(node_mapping.outputs["Vector"], node_environment.inputs["Vector"])
    links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    links.new(node_background.outputs["Background"], node_output.inputs["Surface"])


def set_fov(cam, fov_radians, sensor_width=32):
    cam.data.sensor_width = sensor_width
    cam.data.lens = (sensor_width / 2) / math.tan(fov_radians / 2)

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    return cam, cam_constraint


def set_camera(camera_pos_path, fov_path):
    fov_radians = 0.0
    with open(fov_path, "r") as f:
        first_line = f.readline()
        fov_radians = float(first_line.strip())

    # Load the NumPy matrix from a file
    numpy_matrix = np.load(camera_pos_path)

    if numpy_matrix.ndim == 2:
        # 2-dimensional array, access with 2 indices
        rotation_matrix = numpy_matrix[:, :3]
        location = numpy_matrix[:, 3]

        # Convert NumPy rotation matrix to a Blender Matrix
        blender_rotation_matrix = Matrix(rotation_matrix.tolist())

        # Convert rotation matrix to Euler (can also use to_quaternion() if preferred)
        euler_rotation = blender_rotation_matrix.to_euler()

        # Create the translation vector (location)
        location_vector = Vector(location)

        # Create a new camera object
        camera_data = bpy.data.cameras.new(name="cam")
        camera = bpy.data.objects.new("cam", camera_data)
        bpy.context.collection.objects.link(camera)

        camera.location = location_vector
        camera.rotation_euler = euler_rotation

        set_fov(camera, fov_radians)
        bpy.context.scene.camera = camera

    else:
        # Loop through each element in the rotation matrix and location arrays
        for i in range(numpy_matrix.shape[0]):
            # Separate rotation (n x 3 x 3) and location (n x 3 x 1) from the matrix
            rotation_matrix = numpy_matrix[i, :, :3]
            location = numpy_matrix[i, :, 3]

            # Convert NumPy rotation matrix to a Blender Matrix
            blender_rotation_matrix = Matrix(rotation_matrix.tolist())

            # Convert rotation matrix to Euler (can also use to_quaternion() if preferred)
            euler_rotation = blender_rotation_matrix.to_euler()

            # Create the translation vector (location)
            location_vector = Vector(location)

            # Create a new camera object
            camera_data = bpy.data.cameras.new(name="cam")
            camera = bpy.data.objects.new("cam", camera_data)
            bpy.context.collection.objects.link(camera)

            camera.location = location_vector
            camera.rotation_euler = euler_rotation

            set_fov(camera, fov_radians)
            bpy.context.scene.camera = camera


def set_backgroud_comp(envmap_show_image):
    # set white background only if user wants
    if not envmap_show_image:
        # Enable nodes for compositing
        bpy.context.scene.use_nodes = True
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        # Clear all nodes
        for node in nodes:
            nodes.remove(node)

        # Add Render Layers node
        render_layer_node = nodes.new(type="CompositorNodeRLayers")
        render_layer_node.location = (-300, 0)

        # Add Alpha Over node
        alpha_over_node = nodes.new(type="CompositorNodeAlphaOver")
        alpha_over_node.location = (0, 0)
        alpha_over_node.use_premultiply = True  # Premultiplied alpha for proper overlay

        # Set the background color to white
        alpha_over_node.inputs[1].default_value = (1, 1, 1, 1)  # RGBA for pure white

        # Add Composite output node
        composite_node = nodes.new(type="CompositorNodeComposite")
        composite_node.location = (300, 0)

        # Link nodes
        links.new(
            render_layer_node.outputs["Image"], alpha_over_node.inputs[2]
        )  # Rendered image
        links.new(
            alpha_over_node.outputs[0], composite_node.inputs[0]
        )  # Output to Composite


def set_render(envmap_show_image, resolution, is_hdr=False):
    scene = bpy.context.scene
    render = bpy.context.scene.render
    # Set the render engine to Cycles
    render.engine = "CYCLES"

    # Set the render resolution to 512x512
    render.resolution_x = resolution
    render.resolution_y = resolution

    if is_hdr:
        render.image_settings.file_format = "OPEN_EXR"
        render.image_settings.color_depth = "32"
        render.image_settings.exr_codec = "ZIP"
    else:
        render.image_settings.file_format = "PNG"
        render.image_settings.color_depth = "16"
    render.image_settings.color_mode = "RGBA"
    render.resolution_percentage = 100
    render.film_transparent = not envmap_show_image

    scene.cycles.device = "CPU"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.samples = 512
    scene.cycles.use_denoising = False
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"


def execute_render(output_dir, is_hdr=False):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene
    # Get a list of all camera objects in the scene
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    # Loop through each camera object
    for i, camera in enumerate(cameras):
        # Set the active camera to the current camera
        bpy.context.scene.camera = camera

        # Save the rendered image to a file with the camera name and a 3-digit number
        if is_hdr:
            filename = f"{str(i).zfill(3)}_relight.exr"
        else:
            filename = f"{str(i).zfill(3)}_relight.png"

        scene.render.filepath = os.path.join(output_dir, filename)
        bpy.ops.render.render(write_still=True)


def setup_scene(
    obj_path,
    albedo_path,
    roughness_path,
    metallic_path,
    envmap_path,
    envmap_rotation,
    envmap_intensity,
    envmap_show_image,
    camera_pos_path,
    camera_fov_path,
    output_dir,
    resolution,
    srgb_specularity=False,
    rotate_y_to_z=False,
):
    objs = import_obj(obj_path, rotate_y_to_z=rotate_y_to_z)
    setup_material(objs, albedo_path, roughness_path, metallic_path, srgb_specularity)
    setup_envmap(envmap_path, envmap_rotation, envmap_intensity)
    set_camera(camera_pos_path, camera_fov_path)
    set_backgroud_comp(envmap_show_image)

    set_render(envmap_show_image, resolution, is_hdr=True)
    execute_render(output_dir, is_hdr=True)


def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments

    # Import .obj and textures
    parser.add_argument("--obj_path", type=str, default="mesh_uv.obj")
    parser.add_argument("--albedo_image_path", type=str, default="mesh_uv_albedo.png")
    parser.add_argument(
        "--roughness_image_path", type=str, default="mesh_uv_roughness.png"
    )
    parser.add_argument(
        "--metallic_image_path", type=str, default="mesh_uv_metallic.png"
    )

    # Import environment map and its config
    parser.add_argument("--environment_map_path", type=str, default="env.hdr")
    parser.add_argument(
        "--environment_rotation", type=parse_vector, default="(0.0, 0.0, 0.0)"
    )
    parser.add_argument("--environment_light_strength", type=float, default=1.0)
    parser.add_argument("--environment_show_image", action="store_true")

    # Import camera information
    parser.add_argument("--cam_pos_path", type=str, default="camera_info_corret.npy")
    parser.add_argument("--cam_fov_path", type=str, default="fov.txt")

    parser.add_argument("--output_dir", type=str, default="image")

    parser.add_argument("--image_resolution", type=int, default=512)

    parser.add_argument("--srgb_specularity", action="store_true")
    parser.add_argument("--rotate_y_to_z", action="store_true")

    # Parse known args (ignore Blender's internal args)
    args, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    return args


args = parse_args()
mesh_path = args.obj_path
albedo_path = args.albedo_image_path
roughness_path = args.roughness_image_path
metallic_path = args.metallic_image_path
envmap_path = args.environment_map_path
envmap_rotation = tuple(math.radians(angle) for angle in args.environment_rotation)
envmap_intensity = args.environment_light_strength
camera_pos_path = args.cam_pos_path
camera_fov_path = args.cam_fov_path
output_path = args.output_dir
resolution = args.image_resolution

clean_blender_scene()
setup_scene(
    mesh_path,
    albedo_path,
    roughness_path,
    metallic_path,
    envmap_path,
    envmap_rotation,
    envmap_intensity,
    args.environment_show_image,
    camera_pos_path,
    camera_fov_path,
    output_path,
    resolution,
    srgb_specularity=args.srgb_specularity,
    rotate_y_to_z=args.rotate_y_to_z,
)
