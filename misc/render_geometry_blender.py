# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import sys

# pyre-fixme[21]: Could not find module `bpy`.
import bpy
import numpy as np

# pyre-fixme[21]: Could not find module `mathutils`.
from mathutils import Matrix, Vector


def clean_blender_scene():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)
    for image in bpy.data.images:
        bpy.data.images.remove(image)
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)


def get_blender_version():
    return bpy.app.version[0]


def import_obj(obj_path, rotate_y_to_z=False):
    if get_blender_version() >= 4:
        bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward="Y", axis_up="Z")
    objs = bpy.context.selected_objects
    if rotate_y_to_z:
        for obj in objs:
            obj.rotation_euler[0] = math.radians(90)
    return objs


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

    numpy_matrix = np.load(camera_pos_path)

    if numpy_matrix.ndim == 2:
        rotation_matrix = numpy_matrix[:, :3]
        location = numpy_matrix[:, 3]

        blender_rotation_matrix = Matrix(rotation_matrix.tolist())
        euler_rotation = blender_rotation_matrix.to_euler()
        location_vector = Vector(location)

        camera_data = bpy.data.cameras.new(name="cam")
        camera = bpy.data.objects.new("cam", camera_data)
        bpy.context.collection.objects.link(camera)

        camera.location = location_vector
        camera.rotation_euler = euler_rotation

        set_fov(camera, fov_radians)
        bpy.context.scene.camera = camera
    else:
        for i in range(numpy_matrix.shape[0]):
            rotation_matrix = numpy_matrix[i, :, :3]
            location = numpy_matrix[i, :, 3]

            blender_rotation_matrix = Matrix(rotation_matrix.tolist())
            euler_rotation = blender_rotation_matrix.to_euler()
            location_vector = Vector(location)

            camera_data = bpy.data.cameras.new(name="cam")
            camera = bpy.data.objects.new("cam", camera_data)
            bpy.context.collection.objects.link(camera)

            camera.location = location_vector
            camera.rotation_euler = euler_rotation

            set_fov(camera, fov_radians)
            bpy.context.scene.camera = camera


def set_geometry_material(objs):
    """Set a simple diffuse material for geometry rendering."""
    for obj in objs:
        material = bpy.data.materials.new(name="GeometryMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
        bsdf_node.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf_node.location = (0, 0)

        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        material_output.location = (300, 0)

        links.new(bsdf_node.outputs["BSDF"], material_output.inputs["Surface"])

        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def setup_white_world():
    """Set the world background to white."""
    scene = bpy.context.scene
    if scene.world is None:
        new_world = bpy.data.worlds.new("new_world")
        new_world.use_nodes = True
        scene.world = new_world

    node_tree = scene.world.node_tree
    tree_nodes = node_tree.nodes
    tree_nodes.clear()

    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    node_background.inputs["Strength"].default_value = 1.0

    node_output = tree_nodes.new(type="ShaderNodeOutputWorld")
    node_tree.links.new(
        node_background.outputs["Background"], node_output.inputs["Surface"]
    )


def set_render_geometry(resolution, white_bg=False):
    """Configure Cycles for geometry pass rendering (normal + depth) in EXR."""
    scene = bpy.context.scene
    render = scene.render

    render.engine = "CYCLES"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.film_transparent = not white_bg

    render.image_settings.file_format = "OPEN_EXR"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = "32"
    render.image_settings.exr_codec = "ZIP"

    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.transmission_bounces = 0

    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"

    view_layer = scene.view_layers[0]
    view_layer.use_pass_normal = True
    view_layer.use_pass_z = True


def setup_geometry_compositor(output_dir):
    """Set up compositor nodes to output normal and depth as separate EXR files."""
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    for node in nodes:
        nodes.remove(node)

    render_layer_node = nodes.new(type="CompositorNodeRLayers")
    render_layer_node.location = (-300, 0)

    normal_output = nodes.new(type="CompositorNodeOutputFile")
    normal_output.location = (300, 200)
    normal_output.base_path = output_dir
    normal_output.format.file_format = "OPEN_EXR"
    normal_output.format.color_mode = "RGB"
    normal_output.format.color_depth = "32"
    normal_output.format.exr_codec = "ZIP"
    normal_output.file_slots[0].path = "normal_"

    depth_output = nodes.new(type="CompositorNodeOutputFile")
    depth_output.location = (300, -200)
    depth_output.base_path = output_dir
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_mode = "RGB"
    depth_output.format.color_depth = "32"
    depth_output.format.exr_codec = "ZIP"
    depth_output.file_slots[0].path = "depth_"

    links.new(render_layer_node.outputs["Normal"], normal_output.inputs[0])
    links.new(render_layer_node.outputs["Depth"], depth_output.inputs[0])


def execute_render_geometry(output_dir):
    """Render geometry passes for each camera, then rename outputs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene
    cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]

    for i, camera in enumerate(cameras):
        scene.camera = camera
        scene.frame_set(i + 1)
        setup_geometry_compositor(output_dir)
        bpy.ops.render.render()

        prefix = str(i).zfill(3)
        for pass_name in ["normal", "depth"]:
            src_pattern = "%s_%04d.exr" % (pass_name, i + 1)
            src_path = os.path.join(output_dir, src_pattern)
            dst_path = os.path.join(output_dir, "%s_%s.exr" % (prefix, pass_name))
            if os.path.exists(src_path):
                os.rename(src_path, dst_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cam_pos_path", type=str, required=True)
    parser.add_argument("--cam_fov_path", type=str, required=True)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--rotate_y_to_z", action="store_true")
    parser.add_argument("--white_bg", action="store_true")

    args, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    return args


args = parse_args()

clean_blender_scene()

objs = import_obj(args.obj_path, rotate_y_to_z=args.rotate_y_to_z)
set_geometry_material(objs)
set_camera(args.cam_pos_path, args.cam_fov_path)
set_render_geometry(args.image_resolution, white_bg=args.white_bg)
if args.white_bg:
    setup_white_world()
execute_render_geometry(args.output_dir)
