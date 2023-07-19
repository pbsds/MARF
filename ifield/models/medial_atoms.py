from .. import param
from ..modules import fc
from ..data.common import points
from ..utils import geometry
from ..utils.helpers import compose
from textwrap import indent, dedent
from torch import nn, Tensor
from typing import Optional
import torch
import warnings

# generalize this into a HypoHyperConcat net? ConditionedNet?
class MedialAtomNet(nn.Module):
    def __init__(self,
            in_features     : int,
            latent_features : int,
            hidden_features : int,
            hidden_layers   : int,
            n_atoms         : int   = 1,
            final_init_wrr : tuple[float, float] | None = (0.05, 0.6, 0.1),
            **kw,
            ):
        super().__init__()
        assert n_atoms >= 1, n_atoms
        self.n_atoms = n_atoms

        self.fc = fc.FCBlock(
            in_features      = in_features,
            hidden_layers    = hidden_layers,
            hidden_features  = hidden_features,
            out_features     = n_atoms * 4, # n_atoms * (x, y, z, r)
            outermost_linear = True,
            latent_features  = latent_features,
            **kw,
        )

        if final_init_wrr is not None:
            with torch.no_grad():
                w, r1, r2 = final_init_wrr
                if w != 1: self.fc[-1].linear.weight *= w
                dtype = self.fc[-1].linear.bias.dtype
                self.fc[-1].linear.bias[..., [4*n+i for n in range(n_atoms) for i in range(3)]] = torch.tensor(points.generate_random_sphere_points(n_atoms, radius=r1), dtype=dtype).flatten()
                self.fc[-1].linear.bias[..., 3::4] = r2

    @property
    def is_conditioned(self):
        return self.fc.is_conditioned

    @classmethod
    @compose("\n".join)
    def make_jinja_template(cls, *, exclude_list: set[str] = {}, top_level: bool = True, **kw) -> str:
        yield param.make_jinja_template(cls, top_level=top_level, exclude_list=exclude_list, **kw)
        yield fc.FCBlock.make_jinja_template(top_level=False, exclude_list={
            "in_features",
            "hidden_layers",
            "hidden_features",
            "out_features",
            "outermost_linear",
            "latent_features",
        })

    def forward(self, x: Tensor, z: Optional[Tensor] = None):
        if __debug__ and self.is_conditioned and z is None:
            warnings.warn(f"{self.__class__.__qualname__} is conditioned, but the forward pass was not supplied with a conditioning tensor.")
        return self.fc(x, z)

    def compute_intersections(self,
            ray_origins        : Tensor, # (..., 3)
            ray_dirs           : Tensor, # (..., 3)
            medial_atoms       : Tensor, # (..., 4*self.n_atoms)
            *,
            intersections_only : bool = True,
            return_all_atoms   : bool = False, # only applies if intersections_only=False
            allow_nans         : bool = True,
            improve_miss_grads : bool = False,
            ) -> tuple[(Tensor,)*5]:
        assert ray_origins.shape[:-1] == ray_dirs.shape[:-1] == medial_atoms.shape[:-1], \
            (ray_origins.shape, ray_dirs.shape, medial_atoms.shape)
        assert medial_atoms.shape[-1] % 4 == 0, \
            medial_atoms.shape
        assert ray_origins.shape[-1] == ray_dirs.shape[-1] == 3, \
            (ray_origins.shape, ray_dirs.shape)

        #n_atoms = medial_atoms.shape[-1] // 4
        n_atoms = medial_atoms.shape[-1] >> 2

        # reshape (..., n_atoms * d) to (..., n_atoms, d)
        medial_atoms = medial_atoms.view(*medial_atoms.shape[:-1], n_atoms, 4)
        ray_origins  = ray_origins.unsqueeze(-2).broadcast_to([*ray_origins.shape[:-1], n_atoms, 3])
        ray_dirs     = ray_dirs   .unsqueeze(-2).broadcast_to([*ray_dirs   .shape[:-1], n_atoms, 3])

        # unpack atoms
        sphere_centers = medial_atoms[..., :3]
        sphere_radii   = medial_atoms[..., 3].abs()

        assert not ray_origins   .detach().isnan().any()
        assert not ray_dirs      .detach().isnan().any()
        assert not sphere_centers.detach().isnan().any()
        assert not sphere_radii  .detach().isnan().any()

        # compute intersections
        (
            sphere_center_projs, # (..., 3)
            intersections_near,  # (..., 3)
            intersections_far,   # (..., 3)
            is_intersecting,     # (...) bool
        ) = geometry.ray_sphere_intersect(
            ray_origins,
            ray_dirs,
            sphere_centers,
            sphere_radii,
            return_parts = True,
            allow_nans = allow_nans,
            improve_miss_grads = improve_miss_grads,
        )

        # early return
        if intersections_only and n_atoms == 1:
            return intersections_near.squeeze(-2), is_intersecting.squeeze(-1)

        # compute how close each hit and miss are
        depths      = ((intersections_near - ray_origins) * ray_dirs).sum(-1)
        silhouettes = torch.linalg.norm(sphere_center_projs - sphere_centers, dim=-1) - sphere_radii

        if return_all_atoms:
            intersections_near_all = intersections_near
            depths_all             = depths
            silhouettes_all        = silhouettes
            is_intersecting_all    = is_intersecting
            sphere_centers_all     = sphere_centers
            sphere_radii_all       = sphere_radii

        # collapse n_atoms
        if n_atoms > 1:
            atom_indices = torch.where(is_intersecting.any(dim=-1, keepdim=True),
                torch.where(is_intersecting, depths.detach(), depths.detach()+100).argmin(dim=-1, keepdim=True),
                silhouettes.detach().argmin(dim=-1, keepdim=True),
            )

            intersections_near = intersections_near.take_along_dim(atom_indices[..., None], -2).squeeze(-2)
            depths             = depths            .take_along_dim(atom_indices,            -1).squeeze(-1)
            silhouettes        = silhouettes       .take_along_dim(atom_indices,            -1).squeeze(-1)
            is_intersecting    = is_intersecting   .take_along_dim(atom_indices,            -1).squeeze(-1)
            sphere_centers     = sphere_centers    .take_along_dim(atom_indices[..., None], -2).squeeze(-2)
            sphere_radii       = sphere_radii      .take_along_dim(atom_indices,            -1).squeeze(-1)
        else:
            atom_indices       = None
            intersections_near = intersections_near.squeeze(-2)
            depths             = depths            .squeeze(-1)
            silhouettes        = silhouettes       .squeeze(-1)
            is_intersecting    = is_intersecting   .squeeze(-1)
            sphere_centers     = sphere_centers    .squeeze(-2)
            sphere_radii       = sphere_radii      .squeeze(-1)

        # early return
        if intersections_only:
            return intersections_near, is_intersecting

        # compute sphere normals
        intersection_normals = intersections_near - sphere_centers
        intersection_normals = intersection_normals / (intersection_normals.norm(dim=-1)[..., None] + 1e-9)

        if return_all_atoms:
            intersection_normals_all = intersections_near_all - sphere_centers_all
            intersection_normals_all = intersection_normals_all / (intersection_normals_all.norm(dim=-1)[..., None] + 1e-9)


        return (
            depths,               # (...) valid if hit, based on 'intersections'
            silhouettes,          # (...) always valid
            intersections_near,   # (..., 3) valid if hit, projection if not
            intersection_normals, # (..., 3) valid if hit, rejection if not
            is_intersecting,      # (...) dtype=bool
            sphere_centers,       # (..., 3) network output
            sphere_radii,         # (...) network output
            *(() if not return_all_atoms else (

            atom_indices,
            intersections_near_all,   # (..., N_ATOMS) valid if hit, based on 'intersections'
            intersection_normals_all, # (..., N_ATOMS, 3) valid if hit, rejection if not
            depths_all,               # (..., N_ATOMS) always valid
            silhouettes_all,          # (..., N_ATOMS, 3) valid if hit, projection if not
            is_intersecting_all,      # (..., N_ATOMS) dtype=bool
            sphere_centers_all,       # (..., N_ATOMS, 3) network output
            sphere_radii_all,         # (..., N_ATOMS) network output
        )))
