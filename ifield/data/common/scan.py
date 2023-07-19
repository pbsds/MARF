from ...utils.helpers import compose
from . import points
from .h5_dataclasses import H5Dataclass, H5Array, H5ArrayNoSlice, TransformableDataclassMixin
from methodtools import lru_cache
from sklearn.neighbors import BallTree
import faiss
from trimesh import Trimesh
from typing import Iterable
from typing import Optional, TypeVar
import mesh_to_sdf
import mesh_to_sdf.scan as sdf_scan
import numpy as np
import trimesh
import trimesh.transformations as T
import warnings

__doc__ = """
Here are some helper types for data.
"""

_T = TypeVar("T")

class InvalidateLRUOnWriteMixin:
    def __setattr__(self, key, value):
        if not key.startswith("__wire|"):
            for attr in dir(self):
                if attr.startswith("__wire|"):
                    getattr(self, attr).cache_clear()
        return super().__setattr__(key, value)
def lru_property(func):
    return lru_cache(maxsize=1)(property(func))

class SingleViewScan(H5Dataclass, TransformableDataclassMixin, InvalidateLRUOnWriteMixin, require_all=True):
    points_hit     : H5ArrayNoSlice           # (N, 3)
    normals_hit    : Optional[H5ArrayNoSlice] # (N, 3)
    points_miss    : H5ArrayNoSlice           # (M, 3)
    distances_miss : Optional[H5ArrayNoSlice] # (M)
    colors_hit     : Optional[H5ArrayNoSlice] # (N, 3)
    colors_miss    : Optional[H5ArrayNoSlice] # (M, 3)
    uv_hits        : Optional[H5ArrayNoSlice] # (H, W) dtype=bool
    uv_miss        : Optional[H5ArrayNoSlice] # (H, W) dtype=bool (the reason we store both is due to missing data depth sensor data or filtered backfaces)
    cam_pos        : H5ArrayNoSlice           # (3)
    cam_mat4       : Optional[H5ArrayNoSlice] # (4, 4)
    proj_mat4      : Optional[H5ArrayNoSlice] # (4, 4)
    transforms     : dict[str, H5ArrayNoSlice] # a map of 4x4 transformation matrices

    def transform(self: _T, mat4: np.ndarray, inplace=False) -> _T:
        scale_xyz = mat4[:3, :3].sum(axis=0) # https://math.stackexchange.com/a/1463487
        assert all(scale_xyz - scale_xyz[0] < 1e-8), f"differenty scaled axes: {scale_xyz}"

        out = self if inplace else self.copy(deep=False)
        out.points_hit     = T.transform_points(self.points_hit,  mat4)
        out.normals_hit    = T.transform_points(self.normals_hit, mat4) if self.normals_hit is not None else None
        out.points_miss    = T.transform_points(self.points_miss, mat4)
        out.distances_miss = self.distances_miss * scale_xyz
        out.cam_pos        = T.transform_points(self.points_cam,  mat4)[-1]
        out.cam_mat4       = (mat4 @ self.cam_mat4)  if self.cam_mat4  is not None else None
        out.proj_mat4      = (mat4 @ self.proj_mat4) if self.proj_mat4 is not None else None
        return out

    def compute_miss_distances(self: _T, *, copy: bool = False, deep: bool = False) -> _T:
        assert not self.has_miss_distances
        if not self.is_hitting:
            raise ValueError("No hits to compute the ray distance towards")

        out = self.copy(deep=deep) if copy else self
        out.distances_miss \
            = distance_from_rays_to_point_cloud(
                ray_origins = out.points_cam,
                ray_dirs    = out.ray_dirs_miss,
                points      = out.points_hit,
            ).astype(out.points_cam.dtype)

        return out

    @lru_property
    def points(self) -> np.ndarray: # (N+M+1, 3)
        return np.concatenate((
            self.points_hit,
            self.points_miss,
            self.points_cam,
        ))

    @lru_property
    def uv_points(self) -> np.ndarray: # (N+M+1, 3)
        if not self.has_uv: raise ValueError
        out = np.full((*self.uv_hits.shape, 3), np.nan, dtype=self.points_hit.dtype)
        out[self.uv_hits, :] = self.points_hit
        out[self.uv_miss, :] = self.points_miss
        return out

    @lru_property
    def uv_normals(self) -> np.ndarray: # (N+M+1, 3)
        if not self.has_uv: raise ValueError
        out = np.full((*self.uv_hits.shape, 3), np.nan, dtype=self.normals_hit.dtype)
        out[self.uv_hits, :] = self.normals_hit
        return out

    @lru_property
    def points_cam(self) -> Optional[np.ndarray]: # (1, 3)
        if self.cam_pos is None: return None
        return self.cam_pos[None, :]

    @lru_property
    def points_hit_centroid(self) -> np.ndarray:
        return self.points_hit.mean(axis=0)

    @lru_property
    def points_hit_std(self) -> np.ndarray:
        return self.points_hit.std(axis=0)

    @lru_property
    def is_hitting(self) -> bool:
        return len(self.points_hit) > 0

    @lru_property
    def is_empty(self) -> bool:
        return not (len(self.points_hit) or len(self.points_miss))

    @lru_property
    def has_colors(self) -> bool:
        return self.colors_hit is not None or self.colors_miss is not None

    @lru_property
    def has_normals(self) -> bool:
        return self.normals_hit is not None

    @lru_property
    def has_uv(self) -> bool:
        return self.uv_hits is not None

    @lru_property
    def has_miss_distances(self) -> bool:
        return self.distances_miss is not None

    @lru_property
    def xyzrgb_hit(self) -> np.ndarray: # (N, 6)
        if self.colors_hit is None: raise ValueError
        return np.concatenate([self.points_hit, self.colors_hit], axis=1)

    @lru_property
    def xyzrgb_miss(self) -> np.ndarray: # (M, 6)
        if self.colors_miss is None: raise ValueError
        return np.concatenate([self.points_miss, self.colors_miss], axis=1)

    @lru_property
    def ray_dirs_hit(self) -> np.ndarray: # (N, 3)
        out = self.points_hit - self.points_cam
        out /= np.linalg.norm(out, axis=-1)[:, None] # normalize
        return out

    @lru_property
    def ray_dirs_miss(self) -> np.ndarray: # (N, 3)
        out = self.points_miss - self.points_cam
        out /= np.linalg.norm(out, axis=-1)[:, None] # normalize
        return out

    @classmethod
    def from_mesh_single_view(cls, mesh: Trimesh, *, compute_miss_distances: bool = False, **kw) -> "SingleViewScan":
        if "phi" not in kw and not "theta" in kw:
            kw["theta"], kw["phi"] = points.generate_random_sphere_points(1, compute_sphere_coordinates=True)[0]
        scan = sample_single_view_scan_from_mesh(mesh, **kw)
        if compute_miss_distances and scan.is_hitting:
            scan.compute_miss_distances()
        return scan

    def to_uv_scan(self) -> "SingleViewUVScan":
        return SingleViewUVScan.from_scan(self)

    @classmethod
    def from_uv_scan(self, uvscan: "SingleViewUVScan") -> "SingleViewUVScan":
        return uvscan.to_scan()

# The same, but with support for pagination (should have been this way since the start...)
class SingleViewUVScan(H5Dataclass, TransformableDataclassMixin, InvalidateLRUOnWriteMixin, require_all=True):
    # B may be (N) or (H, W), the latter may be flattened
    hits      : H5Array                  # (*B) dtype=bool
    miss      : H5Array                  # (*B) dtype=bool (the reason we store both is due to missing data depth sensor data or filtered backface hits)
    points    : H5Array                  # (*B, 3) on far plane if miss, NaN if neither hit or miss
    normals   : Optional[H5Array]        # (*B, 3) NaN if not hit
    colors    : Optional[H5Array]        # (*B, 3)
    distances : Optional[H5Array]        # (*B) NaN if not miss
    cam_pos   : Optional[H5ArrayNoSlice] # (3) or (*B, 3)
    cam_mat4  : Optional[H5ArrayNoSlice] # (4, 4)
    proj_mat4 : Optional[H5ArrayNoSlice] # (4, 4)
    transforms : dict[str, H5ArrayNoSlice] # a map of 4x4 transformation matrices

    @classmethod
    def from_scan(cls, scan: SingleViewScan):
        if not scan.has_uv:
            raise ValueError("Scan cloud has no UV data")
        hits, miss = scan.uv_hits, scan.uv_miss
        dtype = scan.points_hit.dtype
        assert hits.ndim in (1, 2), hits.ndim
        assert hits.shape == miss.shape, (hits.shape, miss.shape)

        points = np.full((*hits.shape, 3), np.nan, dtype=dtype)
        points[hits, :] = scan.points_hit
        points[miss, :] = scan.points_miss

        normals = None
        if scan.has_normals:
            normals = np.full((*hits.shape, 3), np.nan, dtype=dtype)
            normals[hits, :] = scan.normals_hit

        distances = None
        if scan.has_miss_distances:
            distances = np.full(hits.shape, np.nan, dtype=dtype)
            distances[miss] = scan.distances_miss

        colors = None
        if scan.has_colors:
            colors = np.full((*hits.shape, 3), np.nan, dtype=dtype)
            if scan.colors_hit  is not None:
                colors[hits, :] = scan.colors_hit
            if scan.colors_miss is not None:
                colors[miss, :] = scan.colors_miss

        return cls(
            hits      = hits,
            miss      = miss,
            points    = points,
            normals   = normals,
            colors    = colors,
            distances = distances,
            cam_pos   = scan.cam_pos,
            cam_mat4  = scan.cam_mat4,
            proj_mat4 = scan.proj_mat4,
            transforms = scan.transforms,
        )

    def to_scan(self) -> "SingleViewScan":
        if not self.is_single_view: raise ValueError
        return SingleViewScan(
            points_hit     = self.points   [self.hits, :],
            points_miss    = self.points   [self.miss, :],
            normals_hit    = self.normals  [self.hits, :] if self.has_normals        else None,
            distances_miss = self.distances[self.miss]    if self.has_miss_distances else None,
            colors_hit     = self.colors   [self.hits, :] if self.has_colors         else None,
            colors_miss    = self.colors   [self.miss, :] if self.has_colors         else None,
            uv_hits        = self.hits,
            uv_miss        = self.miss,
            cam_pos        = self.cam_pos,
            cam_mat4       = self.cam_mat4,
            proj_mat4      = self.proj_mat4,
            transforms     = self.transforms,
        )

    def to_mesh(self) -> trimesh.Trimesh:
        faces: list[(tuple[int, int],)*3] = []
        for x in range(self.hits.shape[0]-1):
            for y in range(self.hits.shape[1]-1):
                c11 = x,   y
                c12 = x,   y+1
                c22 = x+1, y+1
                c21 = x+1, y

                n = sum(map(self.hits.__getitem__, (c11, c12, c22, c21)))
                if n == 3:
                    faces.append((*filter(self.hits.__getitem__, (c11, c12, c22, c21)),))
                elif n == 4:
                    faces.append((c11, c12, c22))
                    faces.append((c11, c22, c21))
        xy2idx = {c:i for i, c in enumerate(set(k for j in faces for k in j))}
        assert self.colors is not None
        return trimesh.Trimesh(
            vertices      = [self.points[i] for i in xy2idx.keys()],
            vertex_colors = [self.colors[i] for i in xy2idx.keys()] if self.colors is not None else None,
            faces         = [tuple(xy2idx[i] for i in face) for face in faces],
        )

    def transform(self: _T, mat4: np.ndarray, inplace=False) -> _T:
        scale_xyz = mat4[:3, :3].sum(axis=0) # https://math.stackexchange.com/a/1463487
        assert all(scale_xyz - scale_xyz[0] < 1e-8), f"differenty scaled axes: {scale_xyz}"

        unflat = self.hits.shape
        flat   = np.product(unflat)

        out = self if inplace else self.copy(deep=False)
        out.points    = T.transform_points(self.points .reshape((*flat, 3)), mat4).reshape((*unflat, 3))
        out.normals   = T.transform_points(self.normals.reshape((*flat, 3)), mat4).reshape((*unflat, 3)) if self.normals_hit is not None else None
        out.distances = self.distances_miss * scale_xyz
        out.cam_pos   = T.transform_points(self.cam_pos[None, ...],  mat4)[0]
        out.cam_mat4  = (mat4 @ self.cam_mat4)  if self.cam_mat4  is not None else None
        out.proj_mat4 = (mat4 @ self.proj_mat4) if self.proj_mat4 is not None else None
        return out

    def compute_miss_distances(self: _T, *, copy: bool = False, deep: bool = False, surface_points: Optional[np.ndarray] = None) -> _T:
        assert not self.has_miss_distances

        shape = self.hits.shape

        out = self.copy(deep=deep) if copy else self
        out.distances = np.zeros(shape, dtype=self.points.dtype)
        if self.is_hitting:
            out.distances[self.miss] \
                = distance_from_rays_to_point_cloud(
                    ray_origins = self.cam_pos_unsqueezed_miss,
                    ray_dirs    = self.ray_dirs_miss,
                    points      = surface_points if surface_points is not None else self.points[self.hits],
                )

        return out

    def fill_missing_points(self: _T, *, copy: bool = False, deep: bool = False) -> _T:
        """
        Fill in missing points as hitting the far plane.
        """
        if not self.is_2d:
            raise ValueError("Cannot fill missing points for non-2d scan!")
        if not self.is_single_view:
            raise ValueError("Cannot fill missing points for non-single-view scans!")
        if self.cam_mat4 is None:
            raise ValueError("cam_mat4 is None")
        if self.proj_mat4 is None:
            raise ValueError("proj_mat4 is None")

        uv = np.argwhere(self.missing).astype(self.points.dtype)
        uv[:, 0] /= (self.missing.shape[1] - 1) / 2
        uv[:, 1] /= (self.missing.shape[0] - 1) / 2
        uv -= 1
        uv = np.stack((
            uv[:, 1],
            -uv[:, 0],
            np.ones(uv.shape[0]), # far clipping plane
            np.ones(uv.shape[0]), # homogeneous coordinate
        ), axis=-1)
        uv = uv @ (self.cam_mat4 @ np.linalg.inv(self.proj_mat4)).T

        out = self.copy(deep=deep) if copy else self
        out.points[self.missing, :] = uv[:, :3] / uv[:, 3][:, None]
        return out

    @lru_property
    def is_hitting(self) -> bool:
        return np.any(self.hits)

    @lru_property
    def has_colors(self) -> bool:
        return not self.colors is None

    @lru_property
    def has_normals(self) -> bool:
        return not self.normals is None

    @lru_property
    def has_miss_distances(self) -> bool:
        return not self.distances is None

    @lru_property
    def any_missing(self) -> bool:
        return np.any(self.missing)

    @lru_property
    def has_missing(self) -> bool:
        return self.any_missing and not np.any(np.isnan(self.points[self.missing]))

    @lru_property
    def cam_pos_unsqueezed(self) -> H5Array:
        if self.cam_pos.ndim != 1:
            return self.cam_pos
        else:
            cam_pos = self.cam_pos
            for _ in range(self.hits.ndim):
                cam_pos = cam_pos[None, ...]
            return cam_pos

    @lru_property
    def cam_pos_unsqueezed_hit(self) -> H5Array:
        if self.cam_pos.ndim != 1:
            return self.cam_pos[self.hits, :]
        else:
            return self.cam_pos[None, :]

    @lru_property
    def cam_pos_unsqueezed_miss(self) -> H5Array:
        if self.cam_pos.ndim != 1:
            return self.cam_pos[self.miss, :]
        else:
            return self.cam_pos[None, :]

    @lru_property
    def ray_dirs(self) -> H5Array:
        return (self.points - self.cam_pos_unsqueezed) * (1 / self.depths[..., None])

    @lru_property
    def ray_dirs_hit(self) -> H5Array:
        out = self.points[self.hits, :] - self.cam_pos_unsqueezed_hit
        out /= np.linalg.norm(out, axis=-1)[..., None] # normalize
        return out

    @lru_property
    def ray_dirs_miss(self) -> H5Array:
        out = self.points[self.miss, :] - self.cam_pos_unsqueezed_miss
        out /= np.linalg.norm(out, axis=-1)[..., None] # normalize
        return out

    @lru_property
    def depths(self) -> H5Array:
        return np.linalg.norm(self.points - self.cam_pos_unsqueezed, axis=-1)

    @lru_property
    def missing(self) -> H5Array:
        return ~(self.hits | self.miss)

    @classmethod
    def from_mesh_single_view(cls, mesh: Trimesh, *, compute_miss_distances: bool = False, **kw) -> "SingleViewUVScan":
        if "phi" not in kw and not "theta" in kw:
            kw["theta"], kw["phi"] = points.generate_random_sphere_points(1, compute_sphere_coordinates=True)[0]
        scan = sample_single_view_scan_from_mesh(mesh, **kw).to_uv_scan()
        if compute_miss_distances:
            scan.compute_miss_distances()
        assert scan.is_2d
        return scan

    @classmethod
    def from_mesh_sphere_view(cls, mesh: Trimesh, *, compute_miss_distances: bool = False, **kw) -> "SingleViewUVScan":
        scan = sample_sphere_view_scan_from_mesh(mesh, **kw)
        if compute_miss_distances:
            surface_points = None
            if scan.hits.sum() > mesh.vertices.shape[0]:
                surface_points = mesh.vertices.astype(scan.points.dtype)
                if not kw.get("no_unit_sphere", False):
                    translation, scale = compute_unit_sphere_transform(mesh, dtype=scan.points.dtype)
                    surface_points = (surface_points + translation) * scale
            scan.compute_miss_distances(surface_points=surface_points)
        assert scan.is_flat
        return scan

    def flatten_and_permute_(self: _T, copy=False) -> _T: # inplace by default
        n_items     = np.product(self.hits.shape)
        permutation = np.random.permutation(n_items)

        out = self.copy(deep=False) if copy else self
        out.hits      = out.hits     .reshape((n_items,  ))[permutation]
        out.miss      = out.miss     .reshape((n_items,  ))[permutation]
        out.points    = out.points   .reshape((n_items, 3))[permutation, :]
        out.normals   = out.normals  .reshape((n_items, 3))[permutation, :] if out.has_normals        else None
        out.colors    = out.colors   .reshape((n_items, 3))[permutation, :] if out.has_colors         else None
        out.distances = out.distances.reshape((n_items,  ))[permutation]    if out.has_miss_distances else None
        return out

    @property
    def is_single_view(self) -> bool:
        return np.product(self.cam_pos.shape[:-1]) == 1 if not self.cam_pos is None else True

    @property
    def is_flat(self) -> bool:
        return len(self.hits.shape) == 1

    @property
    def is_2d(self) -> bool:
        return len(self.hits.shape) == 2


# transforms can be found in pytorch3d.transforms and in open3d
# and in trimesh.transformations

def sample_single_view_scans_from_mesh(
        mesh                : Trimesh,
        *,
        n_batches           : int,
        scan_resolution     : int     = 400,
        compute_normals     : bool    = False,
        fov                 : float   = 1.0472,  # 60 degrees in radians, vertical field of view.
        camera_distance     : float   = 2,
        no_filter_backhits  : bool    = False,
        ) -> Iterable[SingleViewScan]:

    normalized_mesh_cache = []

    for _ in range(n_batches):
        theta, phi = points.generate_random_sphere_points(1, compute_sphere_coordinates=True)[0]

        yield sample_single_view_scan_from_mesh(
            mesh                = mesh,
            phi                 = phi,
            theta               = theta,
            _mesh_is_normalized = False,
            scan_resolution     = scan_resolution,
            compute_normals     = compute_normals,
            fov                 = fov,
            camera_distance     = camera_distance,
            no_filter_backhits  = no_filter_backhits,
            _mesh_cache = normalized_mesh_cache,
        )

def sample_single_view_scan_from_mesh(
        mesh                   : Trimesh,
        *,
        phi                    : float,
        theta                  : float,
        scan_resolution        : int            = 200,
        compute_normals        : bool           = False,
        fov                    : float          = 1.0472,  # 60 degrees in radians, vertical field of view.
        camera_distance        : float          = 2,
        no_filter_backhits     : bool           = False,
        no_unit_sphere         : bool           = False,
        dtype                  : type           = np.float32,
        _mesh_cache            : Optional[list] = None, # provide a list if mesh is reused
        ) -> SingleViewScan:

    # scale and center to unit sphere
    is_cache = isinstance(_mesh_cache, list)
    if is_cache and _mesh_cache and _mesh_cache[0] is mesh:
        _, mesh, translation, scale = _mesh_cache
    else:
        if is_cache:
            if _mesh_cache:
                _mesh_cache.clear()
            _mesh_cache.append(mesh)
        translation, scale = compute_unit_sphere_transform(mesh)
        mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
        if is_cache:
            _mesh_cache.extend((mesh, translation, scale))

    z_near = 1
    z_far  = 3
    cam_mat4 = sdf_scan.get_camera_transform_looking_at_origin(phi, theta, camera_distance=camera_distance)
    cam_pos = cam_mat4 @ np.array([0, 0, 0, 1])

    scan = sdf_scan.Scan(mesh,
        camera_transform  = cam_mat4,
        resolution        = scan_resolution,
        calculate_normals = compute_normals,
        fov               = fov,
        z_near            = z_near,
        z_far             = z_far,
        no_flip_backfaced_normals = True
    )

    # all the scan rays that hit the far plane, based on sdf_scan.Scan.__init__
    misses = np.argwhere(scan.depth_buffer == 0)
    points_miss = np.ones((misses.shape[0], 4))
    points_miss[:, [1, 0]] = misses.astype(float) / (scan_resolution -1) * 2 - 1
    points_miss[:, 1] *= -1
    points_miss[:, 2] = 1 # far plane in clipping space
    points_miss = points_miss @ (cam_mat4 @ np.linalg.inv(scan.projection_matrix)).T
    points_miss /= points_miss[:, 3][:, np.newaxis]
    points_miss = points_miss[:, :3]

    uv_hits = scan.depth_buffer != 0
    uv_miss = ~uv_hits

    if not no_filter_backhits:
        if not compute_normals:
            raise ValueError("not `no_filter_backhits` requires `compute_normals`")
        # inner product
        mask = np.einsum('ij,ij->i', scan.points - cam_pos[:3][None, :], scan.normals) < 0
        scan.points  = scan.points [mask, :]
        scan.normals = scan.normals[mask, :]
        uv_hits[uv_hits] = mask

    transforms = {}

    # undo unit-sphere transform
    if no_unit_sphere:
        scan.points = scan.points * (1 / scale) - translation
        points_miss = points_miss * (1 / scale) - translation
        cam_pos[:3] = cam_pos[:3] * (1 / scale) - translation
        cam_mat4[:3, :] *= 1 / scale
        cam_mat4[:3, 3] -= translation

        transforms["unit_sphere"] = T.scale_and_translate(scale=scale, translate=translation)
        transforms["model"] = np.eye(4)
    else:
        transforms["model"] = np.linalg.inv(T.scale_and_translate(scale=scale, translate=translation))
        transforms["unit_sphere"] = np.eye(4)

    return SingleViewScan(
        normals_hit    = scan.normals           .astype(dtype),
        points_hit     = scan.points            .astype(dtype),
        points_miss    = points_miss            .astype(dtype),
        distances_miss = None,
        colors_hit     = None,
        colors_miss    = None,
        uv_hits        = uv_hits                .astype(bool),
        uv_miss        = uv_miss                .astype(bool),
        cam_pos        = cam_pos[:3]            .astype(dtype),
        cam_mat4       = cam_mat4               .astype(dtype),
        proj_mat4      = scan.projection_matrix .astype(dtype),
        transforms     = {k:v.astype(dtype) for k, v in transforms.items()},
    )

def sample_sphere_view_scan_from_mesh(
        mesh                   : Trimesh,
        *,
        sphere_points          : int            = 4000, # resulting rays are n*(n-1)
        compute_normals        : bool           = False,
        no_filter_backhits     : bool           = False,
        no_unit_sphere         : bool           = False,
        no_permute             : bool           = False,
        dtype                  : type           = np.float32,
        **kw,
        ) -> SingleViewUVScan:
    translation, scale = compute_unit_sphere_transform(mesh, dtype=dtype)

    # get unit-sphere points, then transform to model space
    two_sphere = generate_equidistant_sphere_rays(sphere_points, **kw).astype(dtype) # (n*(n-1), 2, 3)
    two_sphere = two_sphere / scale - translation # we transform after cache lookup

    if mesh.ray.__class__.__module__.split(".")[-1] != "ray_pyembree":
        warnings.warn("Pyembree not found, the ray-tracing will be SLOW!")

    (
        locations,
        index_ray,
        index_tri,
    ) = mesh.ray.intersects_location(
        two_sphere[:, 0, :],
        two_sphere[:, 1, :] - two_sphere[:, 0, :], # direction, not target coordinate
        multiple_hits=False,
    )


    if compute_normals:
        location_normals = mesh.face_normals[index_tri]

    batch = two_sphere.shape[:1]
    hits          = np.zeros((*batch,), dtype=np.bool)
    miss          = np.ones((*batch,), dtype=np.bool)
    cam_pos       = two_sphere[:, 0, :]
    intersections = two_sphere[:, 1, :] # far-plane, effectively
    normals       = np.zeros((*batch, 3), dtype=dtype)

    index_ray_front = index_ray
    if not no_filter_backhits:
        if not compute_normals:
            raise ValueError("not `no_filter_backhits` requires `compute_normals`")
        mask = ((intersections[index_ray] - cam_pos[index_ray]) * location_normals).sum(axis=-1) <= 0
        index_ray_front = index_ray[mask]


    hits[index_ray_front]    = True
    miss[index_ray]          = False
    intersections[index_ray] = locations
    normals[index_ray]       = location_normals


    if not no_permute:
        assert len(batch) == 1, batch
        permutation = np.random.permutation(*batch)
        hits          = hits         [permutation]
        miss          = miss         [permutation]
        intersections = intersections[permutation, :]
        normals       = normals      [permutation, :]
        cam_pos       = cam_pos      [permutation, :]

    # apply unit sphere transform
    if not no_unit_sphere:
        intersections = (intersections + translation) * scale
        cam_pos       = (cam_pos       + translation) * scale

    return SingleViewUVScan(
        hits       = hits,
        miss       = miss,
        points     = intersections,
        normals    = normals,
        colors     = None, # colors
        distances  = None,
        cam_pos    = cam_pos,
        cam_mat4   = None,
        proj_mat4  = None,
        transforms = {},
    )

def distance_from_rays_to_point_cloud(
        ray_origins     : np.ndarray, # (*A, 3)
        ray_dirs        : np.ndarray, # (*A, 3)
        points          : np.ndarray, # (*B, 3)
        dirs_normalized : bool = False,
        n_steps         : int  = 40,
        ) -> np.ndarray: # (A)

    # anything outside of this volume will never constribute to the result
    max_norm = max(
        np.linalg.norm(ray_origins, axis=-1).max(),
        np.linalg.norm(points,      axis=-1).max(),
    ) * 1.02

    if not dirs_normalized:
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1)[..., None]


    # deal with single-view clouds
    if ray_origins.shape != ray_dirs.shape:
        ray_origins = np.broadcast_to(ray_origins, ray_dirs.shape)

    n_points = np.product(points.shape[:-1])
    use_faiss = n_points > 160000*4
    if not use_faiss:
        index = BallTree(points)
    else:
        # http://ann-benchmarks.com/index.html
        assert np.issubdtype(points.dtype,      np.float32)
        assert np.issubdtype(ray_origins.dtype, np.float32)
        assert np.issubdtype(ray_dirs.dtype,    np.float32)
        index = faiss.index_factory(points.shape[-1], "NSG32,Flat") # https://github.com/facebookresearch/faiss/wiki/The-index-factory

        index.nprobe = 5 # 10 # default is 1
        index.train(points)
        index.add(points)

    if not use_faiss:
        min_d, min_n = index.query(ray_origins, k=1, return_distance=True)
    else:
        min_d, min_n = index.search(ray_origins, k=1)
        min_d = np.sqrt(min_d)
    acc_d = min_d.copy()

    for step in range(1, n_steps+1):
        query_points = ray_origins + acc_d * ray_dirs
        if max_norm is not None:
            qmask = np.linalg.norm(query_points, axis=-1) < max_norm
            if not qmask.any(): break
            query_points = query_points[qmask]
        else:
            qmask = slice(None)
        if not use_faiss:
            current_d, current_n = index.query(query_points, k=1, return_distance=True)
        else:
            current_d, current_n = index.search(query_points, k=1)
            current_d = np.sqrt(current_d)
        if max_norm is not None:
            min_d[qmask] = np.minimum(current_d, min_d[qmask])
            new_min_mask = min_d[qmask] == current_d
            qmask2 = qmask.copy()
            qmask2[qmask2] = new_min_mask[..., 0]
            min_n[qmask2] = current_n[new_min_mask[..., 0]]
            acc_d[qmask] += current_d * 0.25
        else:
            np.minimum(current_d, min_d, out=min_d)
            new_min_mask = min_d == current_d
            min_n[new_min_mask] = current_n[new_min_mask]
            acc_d += current_d * 0.25

    closest_points = points[min_n[:, 0], :] # k=1
    distances = np.linalg.norm(np.cross(closest_points - ray_origins, ray_dirs, axis=-1), axis=-1)
    return distances

# helpers

@compose(np.array) # make copy to avoid lru cache mutation
@lru_cache(maxsize=1)
def generate_equidistant_sphere_rays(n : int, **kw) -> np.ndarray: # output (n*n(-1)) rays, n may be off
    sphere_points = points.generate_equidistant_sphere_points(n=n, **kw)

    indices = np.indices((len(sphere_points),))[0] # (N)
    # cartesian product
    cprod = np.transpose([np.tile(indices, len(indices)), np.repeat(indices, len(indices))]) # (N**2, 2)
    # filter repeated combinations
    permutations = cprod[cprod[:, 0] != cprod[:, 1], :] # (N*(N-1), 2)
    # lookup sphere points
    two_sphere = sphere_points[permutations, :] # (N*(N-1), 2, 3)

    return two_sphere

def compute_unit_sphere_transform(mesh: Trimesh, *, dtype=type) -> tuple[np.ndarray, float]:
    """
    returns translation and scale which mesh_to_sdf applies to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    if dtype is not None:
        translation = translation.astype(dtype)
        scale       = scale      .astype(dtype)
    return translation, scale
