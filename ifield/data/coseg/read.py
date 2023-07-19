from . import config
from ..common import points
from ..common import processing
from ..common.scan import SingleViewScan, SingleViewUVScan
from ..common.types import MalformedMesh
from functools import lru_cache
from typing import Optional, Iterable
import numpy as np
import trimesh
import trimesh.transformations as T

__doc__ = """
Here are functions for reading and preprocessing coseg benchmark data

There are essentially a few sets per object:
    "img" - meaning the RGBD images (none found in coseg)
    "mesh_scans" - meaning synthetic scans of a mesh
"""

MESH_TRANSFORM_SKYWARD = T.rotation_matrix(np.pi/2, (1, 0, 0)) # rotate to be upright in pyrender
MESH_POSE_CORRECTIONS = { # to gain a shared canonical orientation
    ("four-legged", 381): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 382): T.rotation_matrix(   1*np.pi/2, (0, 0, 1)),
    ("four-legged", 383): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 384): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 385): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 386): T.rotation_matrix(   1*np.pi/2, (0, 0, 1)),
    ("four-legged", 387): T.rotation_matrix(-0.2*np.pi/2, (0, 1, 0))@T.rotation_matrix(1*np.pi/2, (0, 0, 1)),
    ("four-legged", 388): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 389): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 390): T.rotation_matrix(   0*np.pi/2, (0, 0, 1)),
    ("four-legged", 391): T.rotation_matrix(   0*np.pi/2, (0, 0, 1)),
    ("four-legged", 392): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 393): T.rotation_matrix(   0*np.pi/2, (0, 0, 1)),
    ("four-legged", 394): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 395): T.rotation_matrix(-0.2*np.pi/2, (0, 1, 0))@T.rotation_matrix(1*np.pi/2, (0, 0, 1)),
    ("four-legged", 396): T.rotation_matrix(   1*np.pi/2, (0, 0, 1)),
    ("four-legged", 397): T.rotation_matrix(   0*np.pi/2, (0, 0, 1)),
    ("four-legged", 398): T.rotation_matrix(  -1*np.pi/2, (0, 0, 1)),
    ("four-legged", 399): T.rotation_matrix(   0*np.pi/2, (0, 0, 1)),
    ("four-legged", 400): T.rotation_matrix(   0*np.pi/2, (0, 0, 1)),
}


ModelUid = tuple[str, int]

@lru_cache(maxsize=1)
def list_object_sets() -> list[str]:
    return sorted(
        object_set.name
        for object_set in config.DATA_PATH.iterdir()
        if (object_set / "shapes").is_dir() and object_set.name != "archive"
    )

@lru_cache(maxsize=1)
def list_model_ids(object_sets: Optional[tuple[str]] = None) -> list[ModelUid]:
    return sorted(
        (object_set.name, int(model.stem))
        for object_set in config.DATA_PATH.iterdir()
        if (object_set / "shapes").is_dir() and object_set.name != "archive" and (object_sets is None or object_set.name in object_sets)
        for model in (object_set / "shapes").iterdir()
        if model.is_file() and model.suffix == ".off"
    )

def list_model_id_strings(object_sets: Optional[tuple[str]] = None) -> list[str]:
    return [model_uid_to_string(object_set_id, model_id) for object_set_id, model_id in list_model_ids(object_sets)]

def model_uid_to_string(object_set_id: str, model_id: int) -> str:
    return f"{object_set_id}-{model_id}"

def model_id_string_to_uid(model_string_uid: str) -> ModelUid:
    object_set, split, model = model_string_uid.rpartition("-")
    assert split == "-"
    return (object_set, int(model))

@lru_cache(maxsize=1)
def list_mesh_scan_sphere_coords(n_poses: int = 50) -> list[tuple[float, float]]: # (theta, phi)
    return points.generate_equidistant_sphere_points(n_poses, compute_sphere_coordinates=True)

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

def read_mesh(object_set_id: str, model_id: int) -> trimesh.Trimesh:
    path = config.DATA_PATH / object_set_id / "shapes" / f"{model_id}.off"
    if not path.is_file():
        raise FileNotFoundError(f"{path = }")
    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception as e:
        raise MalformedMesh(f"Trimesh raised: {e.__class__.__name__}: {e}") from e

    pose = MESH_POSE_CORRECTIONS.get((object_set_id, int(model_id)))
    mesh.apply_transform(pose @ MESH_TRANSFORM_SKYWARD if pose is not None else MESH_TRANSFORM_SKYWARD)
    return mesh

# === single-view scan clouds

def compute_mesh_scan_point_cloud(
        object_set_id : str,
        model_id      : int,
        phi           : float,
        theta         : float,
        *,
        compute_miss_distances : bool = False,
        fill_missing_points    : bool = False,
        compute_normals        : bool = True,
        convert_ok             : bool = False,
        **kw,
        ) -> SingleViewScan:

    if convert_ok:
        try:
            return read_mesh_scan_uv(object_set_id, model_id, phi=phi, theta=theta).to_scan()
        except FileNotFoundError:
            pass

    mesh = read_mesh(object_set_id, model_id)
    scan = SingleViewScan.from_mesh_single_view(mesh,
        phi                    = phi,
        theta                  = theta,
        compute_normals        = compute_normals,
        **kw,
    )
    if compute_miss_distances:
        scan.compute_miss_distances()
    if fill_missing_points:
        scan.fill_missing_points()

    return scan

def precompute_mesh_scan_point_clouds(models: Iterable[ModelUid], *, n_poses: int = 50, page: tuple[int, int] = (0, 1), force = False, debug = False, **kw):
    "precomputes all single-view scan clouds and stores them as HDF5 datasets"
    cam_poses        = list_mesh_scan_sphere_coords(n_poses=n_poses)
    pose_identifiers = list_mesh_scan_identifiers  (n_poses=n_poses)
    assert len(cam_poses) == len(pose_identifiers)
    paths = list_mesh_scan_point_cloud_h5_fnames(models, pose_identifiers, n_poses=n_poses)
    mlen_syn = max(len(object_set_id) for object_set_id, model_id in models)
    mlen_mod = max(len(str(model_id)) for object_set_id, model_id in models)
    pretty_identifiers = [
        f"{object_set_id.ljust(mlen_syn)} @ {str(model_id).ljust(mlen_mod)} @ {i:>5} @ ({itentifier}: {theta:.2f}, {phi:.2f})"
        for object_set_id, model_id in models
        for i, (itentifier, (theta, phi)) in enumerate(zip(pose_identifiers, cam_poses))
    ]
    mesh_cache = []
    def computer(pretty_identifier: str) -> SingleViewScan:
        object_set_id, model_id, index, _ = map(str.strip, pretty_identifier.split("@"))
        theta, phi = cam_poses[int(index)]
        return compute_mesh_scan_point_cloud(object_set_id, int(model_id), phi=phi, theta=theta, _mesh_cache=mesh_cache, **kw)
    return processing.precompute_data(computer, pretty_identifiers, paths, page=page, force=force, debug=debug)

def read_mesh_scan_point_cloud(object_set_id: str, model_id: int, *, identifier: str = None, phi: float = None, theta: float = None) -> SingleViewScan:
    if identifier is None:
        if phi is None or theta is None:
            raise ValueError("Provide either phi+theta or an identifier!")
        identifier = mesh_scan_identifier(phi=phi, theta=theta)
    file = config.DATA_PATH / object_set_id / "uv_scan_clouds" / f"{model_id}_normalized_{identifier}.h5"
    return SingleViewScan.from_h5_file(file)

def list_mesh_scan_point_cloud_h5_fnames(models: Iterable[ModelUid], identifiers: Optional[Iterable[str]] = None, **kw):
    if identifiers is None:
        identifiers = list_mesh_scan_identifiers(**kw)
    return [
        config.DATA_PATH / object_set_id / "uv_scan_clouds" / f"{model_id}_normalized_{identifier}.h5"
        for object_set_id, model_id in models
        for identifier in identifiers
    ]


# === single-view UV scan clouds

def compute_mesh_scan_uv(
        object_set_id : str,
        model_id      : int,
        phi           : float,
        theta         : float,
        *,
        compute_miss_distances : bool = False,
        fill_missing_points    : bool = False,
        compute_normals        : bool = True,
        convert_ok             : bool = False,
        **kw,
        ) -> SingleViewUVScan:

    if convert_ok:
        try:
            return read_mesh_scan_point_cloud(object_set_id, model_id, phi=phi, theta=theta).to_uv_scan()
        except FileNotFoundError:
            pass

    mesh = read_mesh(object_set_id, model_id)
    scan = SingleViewUVScan.from_mesh_single_view(mesh,
        phi                    = phi,
        theta                  = theta,
        compute_normals        = compute_normals,
        **kw,
    )
    if compute_miss_distances:
        scan.compute_miss_distances()
    if fill_missing_points:
        scan.fill_missing_points()

    return scan

def precompute_mesh_scan_uvs(models: Iterable[ModelUid], *, n_poses: int = 50, page: tuple[int, int] = (0, 1), force = False, debug = False, **kw):
    "precomputes all single-view scan clouds and stores them as HDF5 datasets"
    cam_poses        = list_mesh_scan_sphere_coords(n_poses=n_poses)
    pose_identifiers = list_mesh_scan_identifiers  (n_poses=n_poses)
    assert len(cam_poses) == len(pose_identifiers)
    paths = list_mesh_scan_uv_h5_fnames(models, pose_identifiers, n_poses=n_poses)
    mlen_syn = max(len(object_set_id) for object_set_id, model_id in models)
    mlen_mod = max(len(str(model_id)) for object_set_id, model_id in models)
    pretty_identifiers = [
        f"{object_set_id.ljust(mlen_syn)} @ {str(model_id).ljust(mlen_mod)} @ {i:>5} @ ({itentifier}: {theta:.2f}, {phi:.2f})"
        for object_set_id, model_id in models
        for i, (itentifier, (theta, phi)) in enumerate(zip(pose_identifiers, cam_poses))
    ]
    mesh_cache = []
    def computer(pretty_identifier: str) -> SingleViewUVScan:
        object_set_id, model_id, index, _ = map(str.strip, pretty_identifier.split("@"))
        theta, phi = cam_poses[int(index)]
        return compute_mesh_scan_uv(object_set_id, int(model_id), phi=phi, theta=theta, _mesh_cache=mesh_cache, **kw)
    return processing.precompute_data(computer, pretty_identifiers, paths, page=page, force=force, debug=debug)

def read_mesh_scan_uv(object_set_id: str, model_id: int, *, identifier: str = None, phi: float = None, theta: float = None) -> SingleViewUVScan:
    if identifier is None:
        if phi is None or theta is None:
            raise ValueError("Provide either phi+theta or an identifier!")
        identifier = mesh_scan_identifier(phi=phi, theta=theta)
    file = config.DATA_PATH / object_set_id / "uv_scan_clouds" / f"{model_id}_normalized_{identifier}.h5"

    return SingleViewUVScan.from_h5_file(file)

def list_mesh_scan_uv_h5_fnames(models: Iterable[ModelUid], identifiers: Optional[Iterable[str]] = None, **kw):
    if identifiers is None:
        identifiers = list_mesh_scan_identifiers(**kw)
    return [
        config.DATA_PATH / object_set_id / "uv_scan_clouds" / f"{model_id}_normalized_{identifier}.h5"
        for object_set_id, model_id in models
        for identifier in identifiers
    ]


# === sphere-view (UV) scan clouds

def compute_mesh_sphere_scan(
        object_set_id : str,
        model_id      : int,
        *,
        compute_normals : bool = True,
        **kw,
        ) -> SingleViewUVScan:
    mesh = read_mesh(object_set_id, model_id)
    scan = SingleViewUVScan.from_mesh_sphere_view(mesh,
        compute_normals = compute_normals,
        **kw,
    )
    return scan

def precompute_mesh_sphere_scan(models: Iterable[ModelUid], *, page: tuple[int, int] = (0, 1), force: bool = False, debug: bool = False, n_points: int = 4000, **kw):
    "precomputes all sphere scan clouds and stores them as HDF5 datasets"
    paths = list_mesh_sphere_scan_h5_fnames(models)
    identifiers = [model_uid_to_string(*i) for i in models]
    def computer(identifier: str) -> SingleViewScan:
        object_set_id, model_id = model_id_string_to_uid(identifier)
        return compute_mesh_sphere_scan(object_set_id, model_id, **kw)
    return processing.precompute_data(computer, identifiers, paths, page=page, force=force, debug=debug)

def read_mesh_mesh_sphere_scan(object_set_id: str, model_id: int) -> SingleViewUVScan:
    file = config.DATA_PATH / object_set_id / "sphere_scan_clouds" / f"{model_id}_normalized.h5"
    return SingleViewUVScan.from_h5_file(file)

def list_mesh_sphere_scan_h5_fnames(models: Iterable[ModelUid]) -> list[str]:
    return [
        config.DATA_PATH / object_set_id / "sphere_scan_clouds" / f"{model_id}_normalized.h5"
        for object_set_id, model_id in models
    ]
