from math import pi
from trimesh import Trimesh
import numpy as np
import os
import trimesh
import trimesh.transformations as T

DEBUG = bool(os.environ.get("IFIELD_DEBUG", ""))

__doc__ = """
Here are some helper functions for processing data.
"""

def rotate_to_closest_axis_aligned_bounds(
        mesh       : Trimesh,
        order_axes : bool    = True,
        fail_ok    : bool    = True,
        ) -> np.ndarray:
    to_origin_mat4, extents = trimesh.bounds.oriented_bounds(mesh, ordered=not order_axes)
    to_aabb_rot_mat4 = T.euler_matrix(*T.decompose_matrix(to_origin_mat4)[3])

    if not order_axes:
        return to_aabb_rot_mat4

    v  = pi / 4 * 1.01 # tolerance
    v2 = pi / 2

    faces = (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (0,-1),
    )
    orientations = [ # 6 faces x 4 rotations per face
        (f[0] * v2, f[1] * v2, i * v2)
        for i in range(4)
        for f in faces]

    for x, y, z in orientations:
        mat4 = T.euler_matrix(x, y, z) @ to_aabb_rot_mat4
        ai, aj, ak = T.euler_from_matrix(mat4)
        if abs(ai) <= v and abs(aj) <= v and abs(ak) <= v:
            return mat4

    if fail_ok: return to_aabb_rot_mat4
    raise Exception("Unable to orient mesh")
