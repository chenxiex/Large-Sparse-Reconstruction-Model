# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# pyre-fixme[21]: Could not find module `bpy`.
import bpy


def get_blender_version():
    """Returns the major version of Blender as an integer."""
    return bpy.app.version[0]


def import_obj(obj_path):
    if os.path.exists(obj_path):
        if get_blender_version() >= 4:
            # Blender 4.x uses wm.obj_import
            bpy.ops.wm.obj_import(filepath=obj_path)
        else:
            # Blender 3.x uses import_scene.obj
            bpy.ops.import_scene.obj(filepath=obj_path)
    else:
        print(obj_path + " does not exists!")


def export_obj(obj_path):
    if get_blender_version() >= 4:
        # Blender 4.x uses wm.obj_export
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_normals=False,
            export_selected_objects=True,
            export_materials=True,
        )
    else:
        # Blender 3.x uses export_scene.obj
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_normals=False,
            use_selection=True,
            use_materials=True,
        )


obj_in_fn = sys.argv[-2]
obj_out_fn = sys.argv[-1]

bpy.ops.object.select_all(action="DESELECT")
# Select the default cube (or any object named "Cube")
if "Cube" in bpy.data.objects:
    bpy.data.objects["Cube"].select_set(True)
# Delete the selected objects
bpy.ops.object.delete()

import_obj(obj_in_fn)

obj = bpy.data.objects[-1]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
# Smoothing
bpy.ops.object.modifier_add(type="SMOOTH")
bpy.context.object.modifiers["Smooth"].factor = 1.0
bpy.context.object.modifiers["Smooth"].iterations = 10
bpy.ops.object.modifier_apply(modifier="Smooth")

# UV Unwrap
obj = bpy.context.active_object
# Ensure the object is a mesh
if obj.type == "MESH":
    # Switch to Edit mode
    bpy.ops.object.mode_set(mode="EDIT")

    # Select all faces
    bpy.ops.mesh.select_all(action="SELECT")

    # Unwrap the mesh using the Smart UV Project method
    bpy.ops.uv.smart_project(island_margin=0.001)

    # Switch back to Object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    print(f"UV map created for {obj.name}")
else:
    print("The selected object is not a mesh.")
export_obj(obj_out_fn)
