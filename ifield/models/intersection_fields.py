from .. import param
from ..modules.dtype import DtypeMixin
from ..utils import geometry
from ..utils.helpers import compose
from ..utils.loss import Schedulable, ensure_schedulables, HParamSchedule, HParamScheduleBase, Linear
from ..utils.operators import diff
from .conditioning import RequiresConditioner, AutoDecoderModuleMixin
from .medial_atoms import MedialAtomNet
from .orthogonal_plane import OrthogonalPlaneNet
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.nn import functional as F
from typing import TypedDict, Literal, Union, Hashable, Optional
import pytorch_lightning as pl
import torch
import os

LOG_ALL_METRICS = bool(int(os.environ.get("IFIELD_LOG_ALL_METRICS", "1")))

if __debug__:
    def broadcast_tensors(*tensors: torch.Tensor) -> list[torch.Tensor]:
        try:
            return torch.broadcast_tensors(*tensors)
        except RuntimeError as e:
            shapes = ", ".join(f"{chr(c)}.size={tuple(t.shape)}" for c, t in enumerate(tensors, ord("a")))
            raise ValueError(f"Could not broadcast tensors {shapes}.\n{str(e)}")
else:
    broadcast_tensors = torch.broadcast_tensors


class ForwardDepthMapsBatch(TypedDict):
    cam2world   : Tensor # (B, 4, 4)
    uv          : Tensor # (B, H, W)
    intrinsics  : Tensor # (B, 3, 3)

class ForwardScanRaysBatch(TypedDict):
    origins     : Tensor # (B, H, W, 3) or (B, 3)
    dirs        : Tensor # (B, H, W, 3)

class LossBatch(TypedDict):
    hits        : Tensor # (B, H, W) dtype=bool
    miss        : Tensor # (B, H, W) dtype=bool
    depths      : Tensor # (B, H, W)
    normals     : Tensor # (B, H, W, 3) NaN if not hit
    distances   : Tensor # (B, H, W, 1) NaN if not miss

class LabeledBatch(TypedDict):
    z_uid       : list[Hashable]

ForwardBatch    = Union[ForwardDepthMapsBatch, ForwardScanRaysBatch]
TrainingBatch   = Union[ForwardBatch, LossBatch, LabeledBatch]


IntersectionMode = Literal[
    "medial_sphere",
    "orthogonal_plane",
]

class IntersectionFieldModel(pl.LightningModule, RequiresConditioner, DtypeMixin):
    net: Union[MedialAtomNet, OrthogonalPlaneNet]

    @ensure_schedulables
    def __init__(self,
            # mode
            input_mode        : geometry.RayEmbedding = "plucker",
            output_mode       : IntersectionMode      = "medial_sphere",

            # network
            latent_features   : int                   = 256,
            hidden_features   : int                   = 512,
            hidden_layers     : int                   = 8,
            improve_miss_grads: bool                  = True,
            normalize_ray_dirs: bool                  = False, # the dataset is usually already normalized, but this could still be important for backprop

            # orthogonal plane
            loss_hit_cross_entropy : Schedulable = 1.0,

            # medial atoms
            loss_intersection       : Schedulable = 1,
            loss_intersection_l2    : Schedulable = 0,
            loss_intersection_proj    : Schedulable = 0,
            loss_intersection_proj_l2 : Schedulable = 0,
            loss_normal_cossim      : Schedulable = 0.25, # supervise target normal cosine similarity
            loss_normal_euclid      : Schedulable = 0,    # supervise target normal l2 distance
            loss_normal_cossim_proj : Schedulable = 0,    # supervise target normal cosine similarity
            loss_normal_euclid_proj : Schedulable = 0,    # supervise target normal l2 distance
            loss_hit_nodistance_l1  : Schedulable = 0,    # constrain no miss distance for hits
            loss_hit_nodistance_l2  : Schedulable = 32,   # constrain no miss distance for hits
            loss_miss_distance_l1   : Schedulable = 0,    # supervise target miss distance for misses
            loss_miss_distance_l2   : Schedulable = 0,    # supervise target miss distance for misses
            loss_inscription_hits   : Schedulable = 0,    # Penalize atom candidates using the supervision data of a different ray
            loss_inscription_hits_l2: Schedulable = 0,    # Penalize atom candidates using the supervision data of a different ray
            loss_inscription_miss   : Schedulable = 0,    # Penalize atom candidates using the supervision data of a different ray
            loss_inscription_miss_l2: Schedulable = 0,    # Penalize atom candidates using the supervision data of a different ray
            loss_sphere_grow_reg    : Schedulable = 0,    # maximialize sphere size
            loss_sphere_grow_reg_hit: Schedulable = 0,    # maximialize sphere size
            loss_embedding_norm     : Schedulable = "0.01**2 * Linear(15)", # DeepSDF schedules over 150 epochs. DeepSDF use 0.01**2, irobot uses 0.04**2
            loss_multi_view_reg             : Schedulable = 0, # minimize gradient w.r.t. delta ray dir, when ray origin = intersection
            loss_atom_centroid_norm_std_reg : Schedulable = 0, # minimize per-atom centroid std

            # optimization
            opt_learning_rate       : Schedulable = 1e-5,
            opt_weight_decay        : float = 0,
            opt_warmup              : float = 0,
            **kw,
            ):
        super().__init__()
        opt_warmup = Linear(opt_warmup)
        opt_warmup._param_name = "opt_warmup"
        self.save_hyperparameters()


        if "half" in input_mode:
            assert output_mode == "medial_sphere" and kw.get("n_atoms", 1) > 1

        assert output_mode in ["medial_sphere", "orthogonal_plane"]
        assert opt_weight_decay       >= 0, opt_weight_decay

        if output_mode == "orthogonal_plane":
            self.net = OrthogonalPlaneNet(
                in_features      = self.n_input_embedding_features,
                hidden_layers    = hidden_layers,
                hidden_features  = hidden_features,
                latent_features  = latent_features,
                **kw,
            )
        elif output_mode == "medial_sphere":
            self.net = MedialAtomNet(
                in_features      = self.n_input_embedding_features,
                hidden_layers    = hidden_layers,
                hidden_features  = hidden_features,
                latent_features  = latent_features,
                **kw,
            )

    def on_fit_start(self):
        if __debug__:
            for k, v in self.hparams.items():
                if isinstance(v, HParamScheduleBase):
                    v.assert_positive(self.trainer.max_epochs)

    @property
    def n_input_embedding_features(self) -> int:
        return geometry.ray_input_embedding_length(self.hparams.input_mode)

    @property
    def n_latent_features(self) -> int:
        return self.hparams.latent_features

    @property
    def latent_embeddings_init_std(self) -> float:
        return 0.01

    @property
    def is_conditioned(self):
        return self.net.is_conditioned

    @property
    def is_double_backprop(self) -> bool:
        return self.is_double_backprop_origins or self.is_double_backprop_dirs

    @property
    def is_double_backprop_origins(self) -> bool:
        prif = self.hparams.output_mode == "orthogonal_plane"
        return prif and self.hparams.loss_normal_cossim

    @property
    def is_double_backprop_dirs(self) -> bool:
        return self.hparams.loss_multi_view_reg

    @classmethod
    @compose("\n".join)
    def make_jinja_template(cls, *, exclude_list: set[str] = {}, top_level: bool = True, **kw) -> str:
        yield param.make_jinja_template(cls, top_level=top_level, **kw)
        yield MedialAtomNet.make_jinja_template(top_level=False, exclude_list={
            "in_features",
            "hidden_layers",
            "hidden_features",
            "latent_features",
        })

    def batch2rays(self, batch: ForwardBatch) -> tuple[Tensor, Tensor]:
        if "uv" in batch:
            raise NotImplementedError
            assert not (self.hparams.loss_multi_view_reg and self.training)
            ray_origins, \
            ray_dirs, \
                = geometry.camera_uv_to_rays(
                    cam2world   = batch["cam2world"],
                    uv          = batch["uv"],
                    intrinsics  = batch["intrinsics"],
                )
        else:
            ray_origins = batch["points" if self.hparams.loss_multi_view_reg and self.training else "origins"]
            ray_dirs    = batch["dirs"]
        return ray_origins, ray_dirs

    def forward(self,
            batch          : ForwardBatch,
            z              : Optional[Tensor] = None, # latent code
            *,
            return_input   : bool             = False,
            allow_nans     : bool             = False, # in output
            **kw,
            ) -> tuple[torch.Tensor, ...]:
        (
            ray_origins, # (B, 3)
            ray_dirs,    # (B, H, W, 3)
        ) = self.batch2rays(batch)

        # Ensure rays are normalized
        # NOTICE: this is slow, make sure to train with optimizations!
        assert ray_dirs.detach().norm(dim=-1).allclose(torch.ones(ray_dirs.shape[:-1], **self.device_and_dtype)),\
            ray_dirs.detach().norm(dim=-1)

        if ray_origins.ndim + 2 == ray_dirs.ndim:
            ray_origins = ray_origins[..., None, None, :]

        ray_origins, ray_dirs = broadcast_tensors(ray_origins, ray_dirs)

        if self.is_double_backprop and self.training:
            if self.is_double_backprop_dirs:
                ray_dirs.requires_grad = True
            if self.is_double_backprop_origins:
                ray_origins.requires_grad = True
            assert ray_origins.requires_grad or ray_dirs.requires_grad

        input = geometry.ray_input_embedding(
            ray_origins, ray_dirs,
            mode           = self.hparams.input_mode,
            normalize_dirs = self.hparams.normalize_ray_dirs,
            is_training    = self.training,
        )
        assert not input.detach().isnan().any()

        predictions = self.net(input, z)

        intersections = self.net.compute_intersections(
            ray_origins, ray_dirs, predictions,
            allow_nans  = allow_nans and not self.training, **kw
        )
        if return_input:
            return ray_origins, ray_dirs, input, intersections
        else:
            return intersections

    def training_step(self, batch: TrainingBatch, batch_idx: int, *, is_validation=False) -> Tensor:
        z = self.encode(batch) if self.is_conditioned else None
        assert self.is_conditioned or len(set(batch["z_uid"])) <= 1, \
            f"Network is unconditioned, but the batch has multiple uids: {set(batch['z_uid'])!r}"

        # unpack
        target_hits      = batch["hits"]      # (B, H, W) dtype=bool
        target_miss      = batch["miss"]      # (B, H, W) dtype=bool
        target_points    = batch["points"]    # (B, H, W, 3)
        target_normals   = batch["normals"]   # (B, H, W, 3) NaN if not hit
        target_distances = batch["distances"] # (B, H, W)    NaN if not miss
        assert not target_normals  [target_hits].isnan().any()
        assert not target_distances[target_miss].isnan().any()
        target_normals[target_normals.isnan()] = 0
        assert not target_normals  .isnan().any()

        # make z fit batch scheme
        if z is not None:
            z = z[..., None, None, :]

        losses  = {}
        metrics = {}
        zeros   = torch.zeros_like(target_distances)

        if self.hparams.output_mode == "medial_sphere":
            assert isinstance(self.net, MedialAtomNet)
            ray_origins, ray_dirs, plucker, (
                depths,               # (...)    float, projection if not hit
                silhouettes,          # (...)    float
                intersections,        # (..., 3) float, projection or NaN if not hit
                intersection_normals, # (..., 3) float, rejection or NaN  if not hit
                is_intersecting,      # (...)    bool, true if hit
                sphere_centers,       # (..., 3) network output
                sphere_radii,         # (...)    network output

                atom_indices,
                all_intersections,        # (..., N_ATOMS)    float, projection or NaN if not hit
                all_intersection_normals, # (..., N_ATOMS, 3) float, rejection or NaN if not hit
                all_depths,               # (..., N_ATOMS)    float, projection if not hit
                all_silhouettes,          # (..., N_ATOMS, 3) float, projection or NaN if not hit
                all_is_intersecting,      # (..., N_ATOMS)    bool, true if hit
                all_sphere_centers,       # (..., N_ATOMS, 3) network output
                all_sphere_radii,         # (..., N_ATOMS)    network output
            ) = self(batch, z,
                intersections_only   = False,
                return_all_atoms     = True,
                allow_nans           = False,
                return_input         = True,
                improve_miss_grads   = True,
            )

            # target hit supervision
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_intersection: # scores true hits
                losses["loss_intersection"] = (
                    (target_points - intersections).norm(dim=-1)
                ).where(target_hits & is_intersecting, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_intersection_l2: # scores true hits
                losses["loss_intersection_l2"] = (
                    (target_points - intersections).pow(2).sum(dim=-1)
                ).where(target_hits & is_intersecting, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_intersection_proj: # scores misses as if they were hits, using the projection
                losses["loss_intersection_proj"] = (
                    (target_points - intersections).norm(dim=-1)
                ).where(target_hits, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_intersection_proj_l2: # scores misses as if they were hits, using the projection
                losses["loss_intersection_proj_l2"] = (
                    (target_points - intersections).pow(2).sum(dim=-1)
                ).where(target_hits, zeros).mean()

            # target hit normal supervision
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_normal_cossim: # scores true hits
                losses["loss_normal_cossim"] = (
                    1 - torch.cosine_similarity(target_normals, intersection_normals, dim=-1)
                ).where(target_hits & is_intersecting, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_normal_euclid: # scores true hits
                losses["loss_normal_euclid"] = (
                    (target_normals - intersection_normals).norm(dim=-1)
                ).where(target_hits & is_intersecting, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_normal_cossim_proj: # scores misses as if they were hits
                losses["loss_normal_cossim_proj"] = (
                    1 - torch.cosine_similarity(target_normals, intersection_normals, dim=-1)
                ).where(target_hits, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_normal_euclid_proj: # scores misses as if they were hits
                losses["loss_normal_euclid_proj"] = (
                    (target_normals - intersection_normals).norm(dim=-1)
                ).where(target_hits, zeros).mean()

            # target sufficient hit radius
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_hit_nodistance_l1: # ensures hits become hits, instead of relying on the projection being right
                losses["loss_hit_nodistance_l1"] = (
                    silhouettes
                ).where(target_hits & (silhouettes > 0), zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_hit_nodistance_l2: # ensures hits become hits, instead of relying on the projection being right
                losses["loss_hit_nodistance_l2"] = (
                    silhouettes
                ).where(target_hits & (silhouettes > 0), zeros).pow(2).mean()

            # target miss supervision
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_miss_distance_l1: # only positive misses reinforcement
                losses["loss_miss_distance_l1"] = (
                    target_distances - silhouettes
                ).where(target_miss, zeros).abs().mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_miss_distance_l2: # only positive misses reinforcement
                losses["loss_miss_distance_l2"] = (
                    target_distances - silhouettes
                ).where(target_miss, zeros).pow(2).mean()

            # incentivise maximal spheres
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_sphere_grow_reg: # all atoms
                losses["loss_sphere_grow_reg"] = ((all_sphere_radii.detach() + 1) - all_sphere_radii).abs().mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_sphere_grow_reg_hit: # true hits only
                losses["loss_sphere_grow_reg_hit"] = ((sphere_radii.detach() + 1) - sphere_radii).where(target_hits & is_intersecting, zeros).abs().mean()

            # spherical latent prior
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_embedding_norm:
                losses["loss_embedding_norm"] = self.latent_embeddings.norm(dim=-1).mean()


            is_grad_enabled = torch.is_grad_enabled()

            # multi-view regularization: atom should not change when view changes
            if self.hparams.loss_multi_view_reg and is_grad_enabled:
                assert ray_dirs.requires_grad, ray_dirs
                assert plucker.requires_grad, plucker
                assert intersections.grad_fn is not None
                assert intersection_normals.grad_fn is not None

                *center_grads, radii_grads = diff.gradients(
                    sphere_centers[..., 0],
                    sphere_centers[..., 1],
                    sphere_centers[..., 2],
                    sphere_radii,
                    wrt=ray_dirs,
                )

                losses["loss_multi_view_reg"] = (
                    sum(
                        i.pow(2).sum(dim=-1)
                        for i in center_grads
                    ).where(target_hits & is_intersecting, zeros).mean()
                    +
                    radii_grads.pow(2).sum(dim=-1)
                        .where(target_hits & is_intersecting, zeros).mean()
                )

            # minimize the volume spanned by each atom
            if self.hparams.loss_atom_centroid_norm_std_reg and self.net.n_atoms > 1:
                assert len(all_sphere_centers.shape) == 5, all_sphere_centers.shape
                losses["loss_atom_centroid_norm_std_reg"] \
                    = ((
                        all_sphere_centers
                        - all_sphere_centers
                            .mean(dim=(1, 2), keepdim=True)
                    ).pow(2).sum(dim=-1) - 0.05**2).clamp(0, None).mean()

            # prif is l1, LSMAT is l2
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_inscription_hits or self.hparams.loss_inscription_miss or self.hparams.loss_inscription_hits_l2 or self.hparams.loss_inscription_miss_l2:
                b   = target_hits.shape[0]          # number of objects
                n   = target_hits.shape[1:].numel() # rays per object
                perm = torch.randperm(n, device=self.device) # ray2ray permutation
                flatten = dict(start_dim=1, end_dim=len(target_hits.shape) - 1)

                (
                    inscr_sphere_center_projs, # (b, n, n_atoms, 3)
                    inscr_intersections_near,  # (b, n, n_atoms, 3)
                    inscr_intersections_far,   # (b, n, n_atoms, 3)
                    inscr_is_intersecting,     # (b, n, n_atoms) dtype=bool
                ) = geometry.ray_sphere_intersect(
                    ray_origins.flatten(**flatten)[:, perm, None, :],
                    ray_dirs   .flatten(**flatten)[:, perm, None, :],
                    all_sphere_centers.flatten(**flatten),
                    all_sphere_radii  .flatten(**flatten),
                    return_parts = True,
                    allow_nans   = False,
                    improve_miss_grads = self.hparams.improve_miss_grads,
                )
                assert inscr_sphere_center_projs.shape == (b, n, self.net.n_atoms, 3), \
                    (inscr_sphere_center_projs.shape, (b, n, self.net.n_atoms, 3))
                inscr_silhouettes = (
                    inscr_sphere_center_projs - all_sphere_centers.flatten(**flatten)
                ).norm(dim=-1) - all_sphere_radii.flatten(**flatten)

                loss_inscription_hits = (
                        (
                            (inscr_intersections_near - target_points.flatten(**flatten)[:, perm, None, :])
                            * ray_dirs.flatten(**flatten)[:, perm, None, :]
                        ).sum(dim=-1)
                    ).where(target_hits.flatten(**flatten)[:, perm, None] & inscr_is_intersecting,
                        torch.zeros(inscr_intersections_near.shape[:-1], **self.device_and_dtype),
                    ).clamp(None, 0)
                loss_inscription_miss = (
                        inscr_silhouettes - target_distances.flatten(**flatten)[:, perm, None]
                    ).where(target_miss.flatten(**flatten)[:, perm, None],
                        torch.zeros_like(inscr_silhouettes)
                    ).clamp(None, 0)

                if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_inscription_hits:
                    losses["loss_inscription_hits"]    = loss_inscription_hits.neg().mean()
                if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_inscription_miss:
                    losses["loss_inscription_miss"]    = loss_inscription_miss.neg().mean()
                if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_inscription_hits_l2:
                    losses["loss_inscription_hits_l2"] = loss_inscription_hits.pow(2).mean()
                if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_inscription_miss_l2:
                    losses["loss_inscription_miss_l2"] = loss_inscription_miss.pow(2).mean()

            # metrics
            metrics["iou"] = (
                ((~target_miss) & is_intersecting.detach()).sum() /
                ((~target_miss) | is_intersecting.detach()).sum()
            )
            metrics["radii"] = sphere_radii.detach().mean() # with the constant applied pressure, we need to measure it this way instead

        elif self.hparams.output_mode == "orthogonal_plane":
            assert isinstance(self.net, OrthogonalPlaneNet)
            ray_origins, ray_dirs, input_embedding, (
                intersections,   # (..., 3) dtype=float
                is_intersecting, # (...)    dtype=float
            ) = self(batch, z, return_input=True, normalize_origins=True)

            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_intersection:
                losses["loss_intersection"] = (
                    (intersections - target_points).norm(dim=-1)
                ).where(target_hits, zeros).mean()
            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_intersection_l2:
                losses["loss_intersection_l2"] = (
                    (intersections - target_points).pow(2).sum(dim=-1)
                ).where(target_hits, zeros).mean()

            if (__debug__ or LOG_ALL_METRICS) or self.hparams.loss_hit_cross_entropy:
                losses["loss_hit_cross_entropy"] = (
                    F.binary_cross_entropy_with_logits(is_intersecting, (~target_miss).to(self.dtype))
                ).mean()

            if self.hparams.loss_normal_cossim and torch.is_grad_enabled():
                jac = diff.jacobian(intersections, ray_origins)
                intersection_normals = self.compute_normals_from_intersection_origin_jacobian(jac, ray_dirs)
                losses["loss_normal_cossim"] = (
                    1 - torch.cosine_similarity(target_normals, intersection_normals, dim=-1)
                ).where(target_hits, zeros).mean()

            if self.hparams.loss_normal_euclid and torch.is_grad_enabled():
                jac = diff.jacobian(intersections, ray_origins)
                intersection_normals = self.compute_normals_from_intersection_origin_jacobian(jac, ray_dirs)
                losses["loss_normal_euclid"] = (
                    (target_normals - intersection_normals).norm(dim=-1)
                ).where(target_hits, zeros).mean()

            if self.hparams.loss_multi_view_reg and torch.is_grad_enabled():
                assert ray_dirs       .requires_grad, ray_dirs
                assert intersections.grad_fn is not None
                grads = diff.gradients(
                    intersections[..., 0],
                    intersections[..., 1],
                    intersections[..., 2],
                    wrt=ray_dirs,
                )
                losses["loss_multi_view_reg"] = sum(
                    i.pow(2).sum(dim=-1)
                    for i in grads
                ).where(target_hits, zeros).mean()

            metrics["iou"] = (
                ((~target_miss) & (is_intersecting>0.5).detach()).sum() /
                ((~target_miss) | (is_intersecting>0.5).detach()).sum()
            )
        else:
            raise NotImplementedError(self.hparams.output_mode)

        # output losses and metrics

        # apply scaling:
        losses_unscaled = losses.copy() # shallow copy
        for k in list(losses.keys()):
            assert losses[k].numel() == 1, f"losses[{k!r}] shape: {losses[k].shape}"
            val_schedule: HParamSchedule = self.hparams[k]
            val = val_schedule.get(self)
            if val == 0:
                if (__debug__ or LOG_ALL_METRICS) and val_schedule.is_const:
                    del losses[k] # it was only added for unscaled logging, do not backprop
                else:
                    losses[k] = 0
            elif val != 1:
                losses[k] = losses[k] * val

        if not losses:
            raise MisconfigurationException("no loss was computed")

        losses["loss"] = sum(losses.values()) * self.hparams.opt_warmup.get(self)
        losses.update({f"unscaled_{k}": v.detach() for k, v in losses_unscaled.items()})
        losses.update({f"metric_{k}": v.detach() for k, v in metrics.items()})
        return losses


    # used by pl.callbacks.EarlyStopping, via cli.py
    @property
    def metric_early_stop(self): return (
        "unscaled_loss_intersection_proj"
        if self.hparams.output_mode == "medial_sphere" else
        "unscaled_loss_intersection"
    )

    def validation_step(self, batch: TrainingBatch, batch_idx: int) -> dict[str, Tensor]:
        losses = self.training_step(batch, batch_idx, is_validation=True)
        return losses

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(),
            lr=1 if not self.hparams.opt_learning_rate.is_const else self.hparams.opt_learning_rate.get_train_value(0),
            weight_decay=self.hparams.opt_weight_decay)
        schedules = []
        if not self.hparams.opt_learning_rate.is_const:
            schedules = [
                torch.optim.lr_scheduler.LambdaLR(adam,
                    lambda epoch: self.hparams.opt_learning_rate.get_train_value(epoch),
                ),
            ]
        return [adam], schedules

    @property
    def example_input_array(self) -> tuple[dict[str, Tensor], Tensor]:
        return (
            { # see self.batch2rays
                "origins" : torch.zeros(1, 3), # most commonly used
                "points"  : torch.zeros(1, 3), # used if self.training and self.hparams.loss_multi_view_reg
                "dirs" : torch.ones(1, 3) * torch.rsqrt(torch.tensor(3)),
            },
            torch.ones(1, self.hparams.latent_features),
        )

    @staticmethod
    def compute_normals_from_intersection_origin_jacobian(origin_jac: Tensor, ray_dirs: Tensor) -> Tensor:
        normals = sum((
            torch.cross(origin_jac[..., 0], origin_jac[..., 1], dim=-1) * -ray_dirs[..., [2]],
            torch.cross(origin_jac[..., 1], origin_jac[..., 2], dim=-1) * -ray_dirs[..., [0]],
            torch.cross(origin_jac[..., 2], origin_jac[..., 0], dim=-1) * -ray_dirs[..., [1]],
        ))
        return normals / normals.norm(dim=-1, keepdim=True)


class IntersectionFieldAutoDecoderModel(IntersectionFieldModel, AutoDecoderModuleMixin):
    def encode(self, batch: LabeledBatch) -> Tensor:
        assert not isinstance(self.trainer.strategy, pl.strategies.DataParallelStrategy)
        return self[batch["z_uid"]] # [N, Z_n]
