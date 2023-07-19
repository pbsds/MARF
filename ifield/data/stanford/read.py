from . import config
from ..common import points
from ..common import processing
from ..common.scan import SingleViewScan, SingleViewUVScan
from ..common.types import MalformedMesh
from functools import lru_cache, wraps
from typing import Optional, Iterable
from pathlib import Path
import gzip
import numpy as np
import trimesh
import trimesh.transformations as T

__doc__ = """
Here are functions for reading and preprocessing shapenet benchmark data

There are essentially a few sets per object:
    "img" - meaning the RGBD images (none found in stanford)
    "mesh_scans" - meaning synthetic scans of a mesh
"""

MESH_TRANSFORM_SKYWARD   = T.rotation_matrix(np.pi/2, (1, 0, 0))
MESH_TRANSFORM_CANONICAL = { # to gain a shared canonical orientation
    "armadillo"    : T.rotation_matrix(np.pi,    (0, 0, 1)) @ MESH_TRANSFORM_SKYWARD,
    "asian_dragon" : T.rotation_matrix(-np.pi/2, (0, 0, 1)) @ MESH_TRANSFORM_SKYWARD,
    "bunny"        : MESH_TRANSFORM_SKYWARD,
    "dragon"       : MESH_TRANSFORM_SKYWARD,
    "drill_bit"    : MESH_TRANSFORM_SKYWARD,
    "happy_buddha" : MESH_TRANSFORM_SKYWARD,
    "lucy"         : T.rotation_matrix(np.pi, (0, 0, 1)),
    "thai_statue"  : MESH_TRANSFORM_SKYWARD,
}

def list_object_names() -> list[str]:
    # downloaded only:
    return [
        i for i, v in config.MODELS.items()
        if (config.DATA_PATH / "meshes" / v.mesh_fname).is_file()
    ]

@lru_cache(maxsize=1)
def list_mesh_scan_sphere_coords(n_poses: int = 50) -> list[tuple[float, float]]: # (theta, phi)
    return points.generate_equidistant_sphere_points(n_poses, compute_sphere_coordinates=True)#, shift_theta=True

def mesh_scan_identifier(*, phi: float, theta: float) -> str:
    return (
        f"{'np'[theta>=0]}{abs(theta):.2f}"
        f"{'np'[phi  >=0]}{abs(phi)  :.2f}"
    ).replace(".", "d")

@lru_cache(maxsize=1)
def list_mesh_scan_identifiers(n_poses: int = 50) -> list[str]:
    out = [
        mesh_scan_identifier(phi=phi, theta=theta)
        for theta, phi in list_mesh_scan_sphere_coords(n_poses)
    ]
    assert len(out) == len(set(out))
    return out

# ===

@lru_cache(maxsize=1)
def read_mesh(obj_name: str) -> trimesh.Trimesh:
    path = config.DATA_PATH / "meshes" / config.MODELS[obj_name].mesh_fname
    if not path.exists():
        raise FileNotFoundError(f"{obj_name = } -> {str(path) = }")
    try:
        if path.suffixes[-1] == ".gz":
            with gzip.open(path, "r") as f:
                mesh = trimesh.load(f, file_type="".join(path.suffixes[:-1])[1:])
        else:
            mesh = trimesh.load(path)
    except Exception as e:
        raise MalformedMesh(f"Trimesh raised: {e.__class__.__name__}: {e}") from e

    # rotate to be upright in pyrender
    mesh.apply_transform(MESH_TRANSFORM_CANONICAL.get(obj_name, MESH_TRANSFORM_SKYWARD))

    return mesh

# === single-view scan clouds

def compute_mesh_scan_point_cloud(
        obj_name : str,
        *,
        phi      : float,
        theta    : float,
        compute_miss_distances : bool = False,
        compute_normals        : bool = True,
        convert_ok             : bool = False, # this does not respect the other hparams
        **kw,
        ) -> SingleViewScan:

    if convert_ok:
        try:
            return read_mesh_scan_uv(obj_name, phi=phi, theta=theta).to_scan()
        except FileNotFoundError:
            pass

    mesh = read_mesh(obj_name)
    return SingleViewScan.from_mesh_single_view(mesh,
        phi                    = phi,
        theta                  = theta,
        compute_normals        = compute_normals,
        compute_miss_distances = compute_miss_distances,
        **kw,
    )

def precompute_mesh_scan_point_clouds(obj_names, *, page: tuple[int, int] = (0, 1), force: bool = False, debug: bool = False, n_poses: int = 50, **kw):
    "precomputes all single-view scan clouds and stores them as HDF5 datasets"
    cam_poses        = list_mesh_scan_sphere_coords(n_poses)
    pose_identifiers = list_mesh_scan_identifiers  (n_poses)
    assert len(cam_poses) == len(pose_identifiers)
    paths = list_mesh_scan_point_cloud_h5_fnames(obj_names, pose_identifiers)
    mlen = max(map(len, config.MODELS.keys()))
    pretty_identifiers = [
        f"{obj_name.ljust(mlen)} @ {i:>5} @ ({itentifier}: {theta:.2f}, {phi:.2f})"
        for obj_name in obj_names
        for i, (itentifier, (theta, phi)) in enumerate(zip(pose_identifiers, cam_poses))
    ]
    mesh_cache = []
    @wraps(compute_mesh_scan_point_cloud)
    def computer(pretty_identifier: str) -> SingleViewScan:
        obj_name, index, _ = map(str.strip, pretty_identifier.split("@"))
        theta, phi = cam_poses[int(index)]
        return compute_mesh_scan_point_cloud(obj_name, phi=phi, theta=theta, _mesh_cache=mesh_cache, **kw)
    return processing.precompute_data(computer, pretty_identifiers, paths, page=page, force=force, debug=debug)

def read_mesh_scan_point_cloud(obj_name, *, identifier: str = None, phi: float = None, theta: float = None) -> SingleViewScan:
    if identifier is None:
        if phi is None or theta is None:
            raise ValueError("Provide either phi+theta or an identifier!")
        identifier = mesh_scan_identifier(phi=phi, theta=theta)
    file = config.DATA_PATH / "clouds" / obj_name / f"mesh_scan_{identifier}_clouds.h5"
    if not file.exists(): raise FileNotFoundError(str(file))
    return SingleViewScan.from_h5_file(file)

def list_mesh_scan_point_cloud_h5_fnames(obj_names: Iterable[str], identifiers: Optional[Iterable[str]] = None, **kw) -> list[Path]:
    if identifiers is None:
        identifiers = list_mesh_scan_identifiers(**kw)
    return [
        config.DATA_PATH / "clouds" / obj_name / f"mesh_scan_{identifier}_clouds.h5"
        for obj_name   in obj_names
        for identifier in identifiers
    ]

# === single-view UV scan clouds

def compute_mesh_scan_uv(
        obj_name : str,
        *,
        phi      : float,
        theta    : float,
        compute_miss_distances : bool = False,
        fill_missing_points    : bool = False,
        compute_normals        : bool = True,
        convert_ok             : bool = False,
        **kw,
        ) -> SingleViewUVScan:

    if convert_ok:
        try:
            return read_mesh_scan_point_cloud(obj_name, phi=phi, theta=theta).to_uv_scan()
        except FileNotFoundError:
            pass

    mesh = read_mesh(obj_name)
    scan = SingleViewUVScan.from_mesh_single_view(mesh,
        phi                = phi,
        theta              = theta,
        compute_normals    = compute_normals,
        **kw,
    )
    if compute_miss_distances:
        scan.compute_miss_distances()
    if fill_missing_points:
        scan.fill_missing_points()

    return scan

def precompute_mesh_scan_uvs(obj_names, *, page: tuple[int, int] = (0, 1), force: bool = False, debug: bool = False, n_poses: int = 50, **kw):
    "precomputes all single-view scan clouds and stores them as HDF5 datasets"
    cam_poses        = list_mesh_scan_sphere_coords(n_poses)
    pose_identifiers = list_mesh_scan_identifiers  (n_poses)
    assert len(cam_poses) == len(pose_identifiers)
    paths = list_mesh_scan_uv_h5_fnames(obj_names, pose_identifiers)
    mlen = max(map(len, config.MODELS.keys()))
    pretty_identifiers = [
        f"{obj_name.ljust(mlen)} @ {i:>5} @ ({itentifier}: {theta:.2f}, {phi:.2f})"
        for obj_name in obj_names
        for i, (itentifier, (theta, phi)) in enumerate(zip(pose_identifiers, cam_poses))
    ]
    mesh_cache = []
    @wraps(compute_mesh_scan_uv)
    def computer(pretty_identifier: str) -> SingleViewScan:
        obj_name, index, _ = map(str.strip, pretty_identifier.split("@"))
        theta, phi = cam_poses[int(index)]
        return compute_mesh_scan_uv(obj_name, phi=phi, theta=theta, _mesh_cache=mesh_cache, **kw)
    return processing.precompute_data(computer, pretty_identifiers, paths, page=page, force=force, debug=debug)

def read_mesh_scan_uv(obj_name, *, identifier: str = None, phi: float = None, theta: float = None) -> SingleViewUVScan:
    if identifier is None:
        if phi is None or theta is None:
            raise ValueError("Provide either phi+theta or an identifier!")
        identifier = mesh_scan_identifier(phi=phi, theta=theta)
    file = config.DATA_PATH / "clouds" / obj_name / f"mesh_scan_{identifier}_uv.h5"
    if not file.exists(): raise FileNotFoundError(str(file))
    return SingleViewUVScan.from_h5_file(file)

def list_mesh_scan_uv_h5_fnames(obj_names: Iterable[str], identifiers: Optional[Iterable[str]] = None, **kw) -> list[Path]:
    if identifiers is None:
        identifiers = list_mesh_scan_identifiers(**kw)
    return [
        config.DATA_PATH / "clouds" / obj_name / f"mesh_scan_{identifier}_uv.h5"
        for obj_name   in obj_names
        for identifier in identifiers
    ]

# === sphere-view (UV) scan clouds

def compute_mesh_sphere_scan(
        obj_name : str,
        *,
        compute_normals : bool = True,
        **kw,
        ) -> SingleViewUVScan:
    mesh = read_mesh(obj_name)
    scan = SingleViewUVScan.from_mesh_sphere_view(mesh,
        compute_normals    = compute_normals,
        **kw,
    )
    return scan

def precompute_mesh_sphere_scan(obj_names, *, page: tuple[int, int] = (0, 1), force: bool = False, debug: bool = False, n_points: int = 4000, **kw):
    "precomputes all single-view scan clouds and stores them as HDF5 datasets"
    paths = list_mesh_sphere_scan_h5_fnames(obj_names)
    @wraps(compute_mesh_sphere_scan)
    def computer(obj_name: str) -> SingleViewScan:
        return compute_mesh_sphere_scan(obj_name, **kw)
    return processing.precompute_data(computer, obj_names, paths, page=page, force=force, debug=debug)

def read_mesh_mesh_sphere_scan(obj_name) -> SingleViewUVScan:
    file = config.DATA_PATH / "clouds" / obj_name / "mesh_sphere_scan.h5"
    if not file.exists(): raise FileNotFoundError(str(file))
    return SingleViewUVScan.from_h5_file(file)

def list_mesh_sphere_scan_h5_fnames(obj_names: Iterable[str]) -> list[Path]:
    return [
        config.DATA_PATH / "clouds" / obj_name / "mesh_sphere_scan.h5"
        for obj_name   in obj_names
    ]
