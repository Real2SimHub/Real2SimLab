"""A minimal PhysCodeGym-style twin of physics_iq_0001_ball_and_block_fall.

A ball and a block are released above a table and fall under a rigid-body simulation; nothing
is keyframed. Deliberately crude (no pillows, no grabbers, rough camera), to show the
``blender_scene`` convention and how far a simple scene gets on PhysCodeGym's pixel metrics.
"""

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y, scene.render.fps = 640, 360, 30
scene.frame_start, scene.frame_end = 1, 240

bpy.ops.rigidbody.world_add()
scene.rigidbody_world.point_cache.frame_end = 240

bpy.ops.mesh.primitive_plane_add(size=3, location=(0, 0, 0))
bpy.ops.rigidbody.object_add()
bpy.context.object.rigid_body.type = "PASSIVE"

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=(-0.15, 0, 0.6))
bpy.ops.rigidbody.object_add()
bpy.context.object.rigid_body.collision_shape = "SPHERE"
bpy.context.object.rigid_body.restitution = 0.6

bpy.ops.mesh.primitive_cube_add(size=0.08, location=(0.15, 0, 0.6))
bpy.ops.rigidbody.object_add()

bpy.ops.object.camera_add(location=(0, -1.6, 0.45), rotation=(1.39, 0, 0))
scene.camera = bpy.context.object
bpy.ops.object.light_add(type="SUN", location=(0, 0, 3))

bpy.ops.ptcache.bake_all(bake=True)
scene.render.filepath = "/app/outputs/frames/frame_"
bpy.ops.wm.save_as_mainfile(filepath="/app/outputs/model.blend")
bpy.ops.render.render(animation=True)
