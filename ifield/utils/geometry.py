from torch import Tensor
from torch.nn import functional as F
from typing import Optional, Literal
import torch
from .helpers import compose


def get_ray_origins(cam2world: Tensor):
    return cam2world[..., :3, 3]

def camera_uv_to_rays(
        cam2world  : Tensor,
        uv         : Tensor,
        intrinsics : Tensor,
        ) -> tuple[Tensor, Tensor]:
    """
    Computes rays and origins from batched cam2world & intrinsics matrices, as well as pixel coordinates
    cam2world:  (..., 4, 4)
    intrinsics: (..., 3, 3)
    uv:         (..., n, 2)
    """
    ray_dirs    = get_ray_directions(uv, cam2world=cam2world, intrinsics=intrinsics)
    ray_origins = get_ray_origins(cam2world)
    ray_origins = ray_origins[..., None, :].expand([*uv.shape[:-1], 3])
    return ray_origins, ray_dirs

RayEmbedding = Literal[
    "plucker",   # LFN
    "perp_foot", # PRIF
    "both",
]

@compose(torch.cat, dim=-1)
@compose(tuple)
def ray_input_embedding(ray_origins: Tensor, ray_dirs: Tensor, mode: RayEmbedding = "plucker", normalize_dirs=False, is_training=False):
    """
    Computes the plucker coordinates / perpendicular foot from ray origins and directions, appending it to direction
    """
    assert ray_origins.shape[-1] == ray_dirs.shape[-1] == 3, \
        f"{ray_dirs.shape = }, {ray_origins.shape = }"

    if normalize_dirs:
        ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)

    yield ray_dirs

    do_moment    = mode in ("plucker",   "both")
    do_perp_feet = mode in ("perp_foot", "both")
    assert do_moment or do_perp_feet

    moment = torch.cross(ray_origins, ray_dirs, dim=-1)
    if do_moment:
        yield moment

    if do_perp_feet:
        perp_feet = torch.cross(ray_dirs, moment, dim=-1)
        yield perp_feet

def ray_input_embedding_length(mode: RayEmbedding = "plucker") -> int:
    do_moment    = mode in ("plucker",   "both")
    do_perp_feet = mode in ("perp_foot", "both")
    assert do_moment or do_perp_feet

    out = 3 # ray_dirs
    if do_moment:
        out += 3 # moment
    if do_perp_feet:
        out += 3 # perp foot
    return out

def parse_intrinsics(intrinsics, return_dict=False):
    fx = intrinsics[..., 0, 0:1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    if return_dict:
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    else:
        return fx, fy, cx, cy

def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x

def lift(x, y, z, intrinsics, homogeneous=False):
    """

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)

def project(x, y, z, intrinsics):
    """

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    return torch.stack((x_proj, y_proj, z), dim=-1)

def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)
    world_coords = torch.einsum("b...ij,b...kj->b...ki", cam2world, pixel_points_cam)[..., :3]

    return world_coords

def project_point_on_ray(projection_point, ray_origin, ray_dir):
    dot = torch.einsum("...j,...j", projection_point-ray_origin, ray_dir)
    return ray_origin + dot[..., None] * ray_dir

def get_ray_directions(
        xy         : Tensor, # (..., N, 2)
        cam2world  : Tensor, # (..., 4, 4)
        intrinsics : Tensor, # (..., 3, 3)
        ):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs

def ray_sphere_intersect(
        ray_origins    : Tensor,                  # (..., 3)
        ray_dirs       : Tensor,                  # (..., 3)
        sphere_centers : Optional[Tensor] = None, # (..., 3)
        sphere_radii   : Optional[Tensor] = 1,    # (...)
        *,
        return_parts   : bool             = False,
        allow_nans     : bool             = True,
        improve_miss_grads : bool         = False,
        ) -> tuple[Tensor, ...]:
    if improve_miss_grads: assert not allow_nans, "improve_miss_grads does not work with allow_nans"
    if sphere_centers is None:
        ray_origins_centered = ray_origins #- torch.zeros_like(ray_origins)
    else:
        ray_origins_centered = ray_origins - sphere_centers

    ray_dir_dot_origins = (ray_dirs * ray_origins_centered).sum(dim=-1, keepdim=True)
    discriminants2 = ray_dir_dot_origins**2 - ((ray_origins_centered * ray_origins_centered).sum(dim=-1) - sphere_radii**2)[..., None]
    if not allow_nans or return_parts:
        is_intersecting = discriminants2 > 0
    if allow_nans:
        discriminants = torch.sqrt(discriminants2)
    else:
        discriminants = torch.sqrt(torch.where(is_intersecting, discriminants2,
            discriminants2 - discriminants2.detach() + 0.001 
            if improve_miss_grads else
            torch.zeros_like(discriminants2)
        ))
        assert not discriminants.detach().isnan().any() # slow, use optimizations!

    if not return_parts:
        return (
            ray_origins + ray_dirs * (- ray_dir_dot_origins - discriminants),
            ray_origins + ray_dirs * (- ray_dir_dot_origins + discriminants),
        )
    else:
        return (
            ray_origins + ray_dirs * (- ray_dir_dot_origins),
            ray_origins + ray_dirs * (- ray_dir_dot_origins - discriminants),
            ray_origins + ray_dirs * (- ray_dir_dot_origins + discriminants),
            is_intersecting.squeeze(-1),
        )
