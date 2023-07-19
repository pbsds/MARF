from __future__ import annotations
from ...utils.helpers import compose
from functools import reduce, lru_cache
from math import ceil
from typing import Iterable
import numpy as np
import operator

__doc__ = """
Here are some helper functions for processing data.
"""


def img2col(img: np.ndarray, psize: int) -> np.ndarray:
    # based of ycb_generate_point_cloud.py provided by YCB

    n_channels = 1 if len(img.shape) == 2 else img.shape[0]
    n_channels, rows, cols = (1,) * (3 - len(img.shape)) + img.shape

    # pad the image
    img_pad = np.zeros((
        n_channels,
        int(ceil(1.0 * rows / psize) * psize),
        int(ceil(1.0 * cols / psize) * psize),
    ))
    img_pad[:, 0:rows, 0:cols] = img

    # allocate output buffer
    final = np.zeros((
        img_pad.shape[1],
        img_pad.shape[2],
        n_channels,
        psize,
        psize,
    ))

    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                img_shift = np.vstack((
                    img_pad[c, x:],
                    img_pad[c, :x]))
                img_shift = np.column_stack((
                    img_shift[:, y:],
                    img_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    img_shift.reshape(
                        int(img_pad.shape[1] / psize), psize,
                        int(img_pad.shape[2] / psize), psize),
                    1,
                    2)

    # crop output and unwrap axes with size==1
    return np.squeeze(final[
        0:rows - psize + 1,
        0:cols - psize + 1])

def filter_depth_discontinuities(depth_map: np.ndarray, filt_size = 7, thresh = 1000) -> np.ndarray:
    """
    Removes data close to discontinuities, with size filt_size.
    """
    # based of ycb_generate_point_cloud.py provided by YCB

    # Ensure that filter sizes are okay
    assert filt_size % 2, "Can only use odd filter sizes."

    # Compute discontinuities
    offset  = int(filt_size - 1) // 2
    patches = 1.0 * img2col(depth_map, filt_size)
    mids    = patches[:, :, offset, offset]
    mins    = np.min(patches, axis=(2, 3))
    maxes   = np.max(patches, axis=(2, 3))

    discont = np.maximum(
        np.abs(mins  - mids),
        np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros(depth_map.shape, dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depth_map * (1 - final_mark)

def reorient_depth_map(
        depth_map      : np.ndarray,
        rgb_map        : np.ndarray,
        depth_mat3     : np.ndarray,          # 3x3 intrinsic camera matrix
        depth_vec5     : np.ndarray,          # 5 distortion parameters (k1, k2, p1, p2, k3)
        rgb_mat3       : np.ndarray,          # 3x3 intrinsic camera matrix
        rgb_vec5       : np.ndarray,          # 5 distortion parameters (k1, k2, p1, p2, k3)
        ir_to_rgb_mat4 : np.ndarray,          # extrinsic transformation matrix from depth to rgb camera viewpoint
        rgb_mask_map   : np.ndarray = None,
        _output_points              = False,  # retval (H, W) if false else (N, XYZRGB)
        _output_hits_uvs            = False,  # retval[1] is dtype=bool of hits shaped like depth_map
        ) -> np.ndarray:

    """
    Corrects depth_map to be from the same view as the rgb_map, with the same dimensions.
    If _output_points is True, the points returned are in the rgb camera space.
    """
    # based of ycb_generate_point_cloud.py provided by YCB
    # now faster AND more easy on the GIL

    height_old, width_old, *_ = depth_map.shape
    height,     width,     *_ = rgb_map.shape


    d_cx, r_cx = depth_mat3[0, 2], rgb_mat3[0, 2] # optical center
    d_cy, r_cy = depth_mat3[1, 2], rgb_mat3[1, 2]
    d_fx, r_fx = depth_mat3[0, 0], rgb_mat3[0, 0] # focal length
    d_fy, r_fy = depth_mat3[1, 1], rgb_mat3[1, 1]
    d_k1, d_k2, d_p1, d_p2, d_k3 = depth_vec5
    c_k1, c_k2, c_p1, c_p2, c_k3 = rgb_vec5

    # make a UV grid over depth_map
    u, v = np.meshgrid(
        np.arange(width_old),
        np.arange(height_old),
    )

    # compute xyz coordinates for all depths
    xyz_depth = np.stack((
        (u - d_cx) / d_fx,
        (v - d_cy) / d_fy,
        depth_map,
        np.ones(depth_map.shape)
    )).reshape((4, -1))
    xyz_depth = xyz_depth[:, xyz_depth[2] != 0]

    # undistort depth coordinates
    d_x, d_y = xyz_depth[:2]
    r = np.linalg.norm(xyz_depth[:2], axis=0)
    xyz_depth[0, :] \
        = d_x / (1 + d_k1*r**2 + d_k2*r**4 + d_k3*r**6) \
        - (2*d_p1*d_x*d_y + d_p2*(r**2 + 2*d_x**2))
    xyz_depth[1, :] \
        = d_y / (1 + d_k1*r**2 + d_k2*r**4 + d_k3*r**6) \
        - (d_p1*(r**2 + 2*d_y**2) + 2*d_p2*d_x*d_y)

    # unproject x and y
    xyz_depth[0, :] *= xyz_depth[2, :]
    xyz_depth[1, :] *= xyz_depth[2, :]

    # convert depths to RGB camera viewpoint
    xyz_rgb = ir_to_rgb_mat4 @ xyz_depth

    # project depths to RGB canvas
    rgb_z_inv = 1 / xyz_rgb[2] # perspective correction
    rgb_uv = np.stack((
        xyz_rgb[0] * rgb_z_inv * r_fx + r_cx + 0.5,
        xyz_rgb[1] * rgb_z_inv * r_fy + r_cy + 0.5,
    )).astype(np.int)

    # mask of the rgb_xyz values within view of rgb_map
    mask = reduce(operator.and_, [
        rgb_uv[0] >= 0,
        rgb_uv[1] >= 0,
        rgb_uv[0] < width,
        rgb_uv[1] < height,
    ])
    if rgb_mask_map is not None:
        mask[mask] &= rgb_mask_map[
            rgb_uv[1, mask],
            rgb_uv[0, mask]]

    if not _output_points: # output image
        output = np.zeros((height, width), dtype=depth_map.dtype)
        output[
            rgb_uv[1, mask],
            rgb_uv[0, mask],
            ] = xyz_rgb[2, mask]

    else: # output pointcloud
        rgbs = rgb_map[ # lookup rgb values using rgb_uv
            rgb_uv[1, mask],
            rgb_uv[0, mask]]
        output = np.stack((
            xyz_rgb[0, mask], # x
            xyz_rgb[1, mask], # y
            xyz_rgb[2, mask], # z
            rgbs[:, 0],      # r
            rgbs[:, 1],      # g
            rgbs[:, 2],      # b
        )).T

    # output for realsies
    if not _output_hits_uvs: #raw
        return output
    else: # with hit mask
        uv = np.zeros((height, width), dtype=bool)
        # filter points overlapping in the depth map
        uv_indices = (
            rgb_uv[1, mask],
            rgb_uv[0, mask],
        )
        _, chosen = np.unique( uv_indices[0] << 32 | uv_indices[1], return_index=True )
        output = output[chosen, :]
        uv[uv_indices] = True
        return output, uv

def join_rgb_and_depth_to_points(*a, **kw) -> np.ndarray:
    return reorient_depth_map(*a, _output_points=True, **kw)

@compose(np.array) # block lru cache mutation
@lru_cache(maxsize=1)
@compose(list)
def generate_equidistant_sphere_points(
        n                          : int,
        centroid                   : np.ndarray = (0, 0, 0),
        radius                     : float      = 1,
        compute_sphere_coordinates : bool       = False,
        compute_normals            : bool       = False,
        shift_theta                : bool       = False,
        ) -> Iterable[tuple[float, ...]]:
    # Deserno M. How to generate equidistributed points on the surface of a sphere
    # https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

    if compute_sphere_coordinates and compute_normals:
        raise ValueError(
            "'compute_sphere_coordinates' and 'compute_normals' are mutually exclusive"
        )

    n_count = 0
    a       = 4 * np.pi / n
    d       = np.sqrt(a)
    n_theta = round(np.pi / d)
    d_theta = np.pi / n_theta
    d_phi   = a     / d_theta

    for i in range(0, n_theta):
        theta = np.pi * (i + 0.5) / n_theta
        n_phi = round(2 * np.pi * np.sin(theta) / d_phi)

        for j in range(0, n_phi):
            phi = 2 * np.pi * j / n_phi

            if compute_sphere_coordinates: # (theta, phi)
                yield (
                    theta if shift_theta else theta - 0.5*np.pi,
                    phi,
                )
            elif compute_normals: # (x, y, z, nx, ny, nz)
                yield (
                    centroid[0] + radius * np.sin(theta) * np.cos(phi),
                    centroid[1] + radius * np.sin(theta) * np.sin(phi),
                    centroid[2] + radius * np.cos(theta),
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                )
            else: # (x, y, z)
                yield (
                    centroid[0] + radius * np.sin(theta) * np.cos(phi),
                    centroid[1] + radius * np.sin(theta) * np.sin(phi),
                    centroid[2] + radius * np.cos(theta),
                )
            n_count += 1


def generate_random_sphere_points(
        n                          : int,
        centroid                   : np.ndarray = (0, 0, 0),
        radius                     : float      = 1,
        compute_sphere_coordinates : bool       = False,
        compute_normals            : bool       = False,
        shift_theta                : bool       = False, # depends on convention
        ) -> np.ndarray:
    if compute_sphere_coordinates and compute_normals:
        raise ValueError(
            "'compute_sphere_coordinates' and 'compute_normals' are mutually exclusive"
        )

    theta = np.arcsin(np.random.uniform(-1, 1, n)) # inverse transform sampling
    phi   = np.random.uniform(0, 2*np.pi, n)

    if compute_sphere_coordinates: # (theta, phi)
        return np.stack((
            theta if not shift_theta else 0.5*np.pi + theta,
            phi,
        ), axis=1)
    elif compute_normals: # (x, y, z, nx, ny, nz)
        return np.stack((
            centroid[0] + radius * np.cos(theta) * np.cos(phi),
            centroid[1] + radius * np.cos(theta) * np.sin(phi),
            centroid[2] + radius * np.sin(theta),
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            np.sin(theta),
        ), axis=1)
    else: # (x, y, z)
        return np.stack((
            centroid[0] + radius * np.cos(theta) * np.cos(phi),
            centroid[1] + radius * np.cos(theta) * np.sin(phi),
            centroid[2] + radius * np.sin(theta),
        ), axis=1)
