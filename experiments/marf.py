#!/usr/bin/env python3
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from ifield import logging
from ifield.cli import CliInterface
from ifield.data.common.scan import SingleViewUVScan
from ifield.data.coseg import read as coseg_read
from ifield.data.stanford import read as stanford_read
from ifield.datasets import stanford, coseg, common
from ifield.models import intersection_fields
from ifield.utils.operators import diff
from ifield.viewer.ray_field import ModelViewer
from munch import Munch
from pathlib import Path
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from trimesh import Trimesh
from typing import Union
import builtins
import itertools
import json
import numpy as np
import pytorch_lightning as pl
import rich
import rich.pretty
import statistics
import torch
pl.seed_everything(31337)
torch.set_float32_matmul_precision('medium')


IField = intersection_fields.IntersectionFieldAutoDecoderModel # brevity


class RayFieldAdDataModuleBase(pl.LightningDataModule, ABC):
    @property
    @abstractmethod
    def observation_ids(self) -> list[str]:
        ...

    @abstractmethod
    def mk_ad_dataset(self) -> common.AutodecoderDataset:
        ...

    @staticmethod
    @abstractmethod
    def get_trimesh_from_uid(uid) -> Trimesh:
        ...

    @staticmethod
    @abstractmethod
    def get_sphere_scan_from_uid(uid) -> SingleViewUVScan:
        ...

    def setup(self, stage=None):
        assert stage in ["fit", None] # fit is for train/val, None is for all. "test" not supported ATM

        if not self.hparams.data_dir is None:
            coseg.config.DATA_PATH = self.hparams.data_dir
        step = self.hparams.step # brevity

        dataset = self.mk_ad_dataset()
        n_items_pre_step_mapping = len(dataset)

        if step > 1:
            dataset = common.TransformExtendedDataset(dataset)

        for sx in range(step):
            for sy in range(step):
                def make_slicer(sx, sy, step) -> callable: # the closure is required
                    if step > 1:
                        return lambda t: t[sx::step, sy::step]
                    else:
                        return lambda t: t
                @dataset.map(slicer=make_slicer(sx, sy, step))
                def unpack(sample: tuple[str, SingleViewUVScan], slicer: callable):
                    scan: SingleViewUVScan = sample[1]
                    assert not scan.hits.shape[0] % step, f"{scan.hits.shape[0]=} not divisible by {step=}"
                    assert not scan.hits.shape[1] % step, f"{scan.hits.shape[1]=} not divisible by {step=}"

                    return {
                        "z_uid"     : sample[0],
                        "origins"   : scan.cam_pos,
                        "dirs"      : slicer(scan.ray_dirs),
                        "points"    : slicer(scan.points),
                        "hits"      : slicer(scan.hits),
                        "miss"      : slicer(scan.miss),
                        "normals"   : slicer(scan.normals),
                        "distances" : slicer(scan.distances),
                    }

        # Split each object into train/val with SampleSplit
        n_items = len(dataset)
        n_val   = int(n_items * self.hparams.val_fraction)
        n_train = n_items - n_val
        self.generator = torch.Generator().manual_seed(self.hparams.prng_seed)

        # split the dataset such that all steps are in same part
        assert n_items == n_items_pre_step_mapping * step * step, (n_items, n_items_pre_step_mapping, step)
        indices = [
            i*step*step + sx*step + sy
            for i in torch.randperm(n_items_pre_step_mapping, generator=self.generator).tolist()
            for sx in range(step)
            for sy in range(step)
        ]
        self.dataset_train  = Subset(dataset, sorted(indices[:n_train], key=lambda x: torch.rand(1, generator=self.generator).tolist()[0]))
        self.dataset_val    = Subset(dataset, sorted(indices[n_train:n_train+n_val], key=lambda x: torch.rand(1, generator=self.generator).tolist()[0]))

        assert len(self.dataset_train) % self.hparams.batch_size == 0
        assert len(self.dataset_val)   % self.hparams.batch_size == 0

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
            batch_size         = self.hparams.batch_size,
            drop_last          = self.hparams.drop_last,
            num_workers        = self.hparams.num_workers,
            persistent_workers = self.hparams.persistent_workers,
            pin_memory         = self.hparams.pin_memory,
            prefetch_factor    = self.hparams.prefetch_factor,
            shuffle            = self.hparams.shuffle,
            generator          = self.generator,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
            batch_size         = self.hparams.batch_size,
            drop_last          = self.hparams.drop_last,
            num_workers        = self.hparams.num_workers,
            persistent_workers = self.hparams.persistent_workers,
            pin_memory         = self.hparams.pin_memory,
            prefetch_factor    = self.hparams.prefetch_factor,
            generator          = self.generator,
        )


class StanfordUVDataModule(RayFieldAdDataModuleBase):
    skyward = "+Z"
    def __init__(self,
            data_dir           : Union[str, Path, None] = None,
            obj_names          : list[str]              = ["bunny"], # empty means all

            prng_seed          : int                    = 1337,
            step               : int                    = 2,
            batch_size         : int                    = 5,
            drop_last          : bool                   = False,
            num_workers        : int                    = 8,
            persistent_workers : bool                   = True,
            pin_memory         : int                    = True,
            prefetch_factor    : int                    = 2,
            shuffle            : bool                   = True,
            val_fraction       : float                  = 0.30,
            ):
        super().__init__()
        if not obj_names:
            obj_names = stanford_read.list_object_names()
        self.save_hyperparameters()

    @property
    def observation_ids(self) -> list[str]:
        return self.hparams.obj_names

    def mk_ad_dataset(self) -> common.AutodecoderDataset:
        return stanford.AutodecoderSingleViewUVScanDataset(
            obj_names = self.hparams.obj_names,
            data_path = self.hparams.data_dir,
        )

    @staticmethod
    def get_trimesh_from_uid(obj_name) -> Trimesh:
        import mesh_to_sdf
        mesh = stanford_read.read_mesh(obj_name)
        return mesh_to_sdf.scale_to_unit_sphere(mesh)

    @staticmethod
    def get_sphere_scan_from_uid(obj_name) -> SingleViewUVScan:
        return stanford_read.read_mesh_mesh_sphere_scan(obj_name)


class CosegUVDataModule(RayFieldAdDataModuleBase):
    skyward = "+Y"
    def __init__(self,
            data_dir           : Union[str, Path, None] = None,
            object_sets        : tuple[str]             = ["tele-aliens"], # empty means all

            prng_seed          : int                    = 1337,
            step               : int                    = 2,
            batch_size         : int                    = 5,
            drop_last          : bool                   = False,
            num_workers        : int                    = 8,
            persistent_workers : bool                   = True,
            pin_memory         : int                    = True,
            prefetch_factor    : int                    = 2,
            shuffle            : bool                   = True,
            val_fraction       : float                  = 0.30,
            ):
        super().__init__()
        if not object_sets:
            object_sets = coseg_read.list_object_sets()
        object_sets = tuple(object_sets)
        self.save_hyperparameters()

    @property
    def observation_ids(self) -> list[str]:
        return coseg_read.list_model_id_strings(self.hparams.object_sets)

    def mk_ad_dataset(self) -> common.AutodecoderDataset:
        return coseg.AutodecoderSingleViewUVScanDataset(
            object_sets = self.hparams.object_sets,
            data_path   = self.hparams.data_dir,
        )

    @staticmethod
    def get_trimesh_from_uid(string_uid):
        raise NotImplementedError

    @staticmethod
    def get_sphere_scan_from_uid(string_uid) -> SingleViewUVScan:
        uid = coseg_read.model_id_string_to_uid(string_uid)
        return coseg_read.read_mesh_mesh_sphere_scan(*uid)


def mk_cli(args=None) -> CliInterface:
    cli = CliInterface(
        module_cls     = IField,
        datamodule_cls = [StanfordUVDataModule, CosegUVDataModule],
        workdir        = Path(__file__).parent.resolve(),
        experiment_name_prefix = "ifield",
    )
    cli.trainer_defaults.update(dict(
        precision  = 16,
        min_epochs =  5,
    ))

    @cli.register_pre_training_callback
    def populate_autodecoder_z_uids(args: Namespace, config: Munch, module: IField, trainer: pl.Trainer, datamodule: RayFieldAdDataModuleBase, logger: logging.Logger):
        module.set_observation_ids(datamodule.observation_ids)
        rank = getattr(rank_zero_only, "rank", 0)
        rich.print(f"[rank {rank}] {len(datamodule.observation_ids)     = }")
        rich.print(f"[rank {rank}] {len(datamodule.observation_ids) > 1 = }")
        rich.print(f"[rank {rank}] {module.is_conditioned               = }")

    @cli.register_action(help="Interactive window with direct renderings from the model", args=[
        ("--shading",  dict(type=int, default=ModelViewer.vizmodes_shading  .index("lambertian"),              help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_shading))}}}")),
        ("--centroid", dict(type=int, default=ModelViewer.vizmodes_centroids.index("best-centroids-colored"),  help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_centroids))}}}")),
        ("--spheres",  dict(type=int, default=ModelViewer.vizmodes_spheres  .index(None),                      help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_spheres))}}}")),
        ("--analytical-normals",  dict(action="store_true")),
        ("--ground-truth",  dict(action="store_true")),
        ("--solo-atom",dict(type=int, default=None, help="Rendering mode")),
        ("--res",      dict(type=int, nargs=2, default=(210, 160), help="Rendering resolution")),
        ("--bg",       dict(choices=["map", "white", "black"], default="map")),
        ("--skyward",  dict(type=str, default="+Z", help='one of: "+X", "-X", "+Y", "-Y", ["+Z"], "-Z"')),
        ("--scale",    dict(type=int, default=3, help="Rendering scale")),
        ("--fps",      dict(type=int, default=None, help="FPS upper limit")),
        ("--cam-state",dict(type=str, default=None, help="json cam state, expored with CTRL+H")),
        ("--write",    dict(type=Path, default=None, help="Where to write a screenshot.")),
    ])
    @torch.no_grad()
    def viewer(args: Namespace, config: Munch, model: IField):
        datamodule_cls: RayFieldAdDataModuleBase = cli.get_datamodule_cls_from_config(args, config)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            model.to("cuda")
        viewer = ModelViewer(model, start_uid=next(iter(model.keys())),
            name           = config.experiment_name,
            screenshot_dir = Path(__file__).parent.parent / "images/pygame-viewer",
            res            = args.res,
            skyward        = args.skyward,
            scale          = args.scale,
            mesh_gt_getter = datamodule_cls.get_trimesh_from_uid,
        )
        viewer.display_mode_shading  = args.shading
        viewer.display_mode_centroid = args.centroid
        viewer.display_mode_spheres  = args.spheres
        if args.ground_truth:       viewer.display_mode_normals = viewer.vizmodes_normals.index("ground_truth")
        if args.analytical_normals: viewer.display_mode_normals = viewer.vizmodes_normals.index("analytical")
        viewer.atom_index_solo       = args.solo_atom
        viewer.fps_cap               = args.fps
        viewer.display_sphere_map_bg = { "map": True, "white": 255, "black": 0 }[args.bg]
        if args.cam_state is not None:
            viewer.cam_state         = json.loads(args.cam_state)
        if args.write is None:
            viewer.run()
        else:
            assert args.write.suffix == ".png", args.write.name
            viewer.render_headless(args.write,
                n_frames       = 1,
                fps            = 1,
                state_callback = None,
            )

    @cli.register_action(help="Prerender direct renderings from the model", args=[
        ("output_path",dict(type=Path, help="Where to store the output. We recommend a .mp4 suffix.")),
        ("uids",       dict(type=str, nargs="*")),
        ("--frames",   dict(type=int, default=60, help="Number of per interpolation. Default is 60")),
        ("--fps",      dict(type=int, default=60, help="Default is 60")),
        ("--shading",  dict(type=int, default=ModelViewer.vizmodes_shading  .index("lambertian"),             help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_shading))}}}")),
        ("--centroid", dict(type=int, default=ModelViewer.vizmodes_centroids.index("best-centroids-colored"), help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_centroids))}}}")),
        ("--spheres",  dict(type=int, default=ModelViewer.vizmodes_spheres  .index(None),                     help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_spheres))}}}")),
        ("--analytical-normals",  dict(action="store_true")),
        ("--solo-atom",dict(type=int, default=None, help="Rendering mode")),
        ("--res",      dict(type=int, nargs=2, default=(240, 240), help="Rendering resolution. Default is 240 240")),
        ("--bg",       dict(choices=["map", "white", "black"], default="map")),
        ("--skyward",  dict(type=str, default="+Z", help='one of: "+X", "-X", "+Y", "-Y", ["+Z"], "-Z"')),
        ("--bitrate",  dict(type=str, default="1500k", help="Encoding bitrate. Default is 1500k")),
        ("--cam-state",dict(type=str, default=None, help="json cam state, expored with CTRL+H")),
    ])
    @torch.no_grad()
    def render_video_interpolation(args: Namespace, config: Munch, model: IField, **kw):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            model.to("cuda")
        uids = args.uids or list(model.keys())
        assert len(uids) > 1
        if not args.uids: uids.append(uids[0])
        viewer = ModelViewer(model, uids[0],
            name           = config.experiment_name,
            screenshot_dir = Path(__file__).parent.parent / "images/pygame-viewer",
            res            = args.res,
            skyward        = args.skyward,
        )
        if args.cam_state is not None:
            viewer.cam_state         = json.loads(args.cam_state)
        viewer.display_mode_shading  = args.shading
        viewer.display_mode_centroid = args.centroid
        viewer.display_mode_spheres  = args.spheres
        if args.analytical_normals: viewer.display_mode_normals = viewer.vizmodes_normals.index("analytical")
        viewer.atom_index_solo       = args.solo_atom
        viewer.display_sphere_map_bg = { "map": True, "white": 255, "black": 0 }[args.bg]
        def state_callback(self: ModelViewer, frame: int):
            if frame % args.frames:
                self.lambertian_color = (0.8, 0.8, 1.0)
            else:
                self.lambertian_color = (1.0, 1.0, 1.0)
            self.fps = args.frames
            idx = frame // args.frames + 1
            if idx != len(uids):
                self.current_uid = uids[idx]
        print(f"Writing video to {str(args.output_path)!r}...")
        viewer.render_headless(args.output_path,
            n_frames       = args.frames * (len(uids)-1) + 1,
            fps            = args.fps,
            state_callback = state_callback,
            bitrate        = args.bitrate,
        )

    @cli.register_action(help="Prerender direct renderings from the model", args=[
        ("output_path",dict(type=Path, help="Where to store the output. We recommend a .mp4 suffix.")),
        ("--frames",   dict(type=int, default=180, help="Number of frames. Default is 180")),
        ("--fps",      dict(type=int, default=60, help="Default is 60")),
        ("--shading",  dict(type=int, default=ModelViewer.vizmodes_shading  .index("lambertian"),             help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_shading))}}}")),
        ("--centroid", dict(type=int, default=ModelViewer.vizmodes_centroids.index("best-centroids-colored"), help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_centroids))}}}")),
        ("--spheres",  dict(type=int, default=ModelViewer.vizmodes_spheres  .index(None),                     help=f"Rendering mode. {{{', '.join(f'{i}: {m!r}'for i, m in enumerate(ModelViewer.vizmodes_spheres))}}}")),
        ("--analytical-normals",  dict(action="store_true")),
        ("--solo-atom",dict(type=int, default=None, help="Rendering mode")),
        ("--res",      dict(type=int, nargs=2, default=(320, 240), help="Rendering resolution. Default is 320 240")),
        ("--bg",       dict(choices=["map", "white", "black"], default="map")),
        ("--skyward",  dict(type=str, default="+Z", help='one of: "+X", "-X", "+Y", "-Y", ["+Z"], "-Z"')),
        ("--bitrate",  dict(type=str, default="1500k", help="Encoding bitrate. Default is 1500k")),
        ("--cam-state",dict(type=str, default=None, help="json cam state, expored with CTRL+H")),
    ])
    @torch.no_grad()
    def render_video_spin(args: Namespace, config: Munch, model: IField, **kw):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            model.to("cuda")
        viewer = ModelViewer(model, start_uid=next(iter(model.keys())),
            name           = config.experiment_name,
            screenshot_dir = Path(__file__).parent.parent / "images/pygame-viewer",
            res            = args.res,
            skyward        = args.skyward,
        )
        if args.cam_state is not None:
            viewer.cam_state         = json.loads(args.cam_state)
        viewer.display_mode_shading  = args.shading
        viewer.display_mode_centroid = args.centroid
        viewer.display_mode_spheres  = args.spheres
        if args.analytical_normals: viewer.display_mode_normals = viewer.vizmodes_normals.index("analytical")
        viewer.atom_index_solo       = args.solo_atom
        viewer.display_sphere_map_bg = { "map": True, "white": 255, "black": 0 }[args.bg]
        cam_rot_x_init = viewer.cam_rot_x
        def state_callback(self: ModelViewer, frame: int):
            self.cam_rot_x = cam_rot_x_init + 3.14 * (frame / args.frames) * 2
        print(f"Writing video to {str(args.output_path)!r}...")
        viewer.render_headless(args.output_path,
            n_frames       = args.frames,
            fps            = args.fps,
            state_callback = state_callback,
            bitrate        = args.bitrate,
        )

    @cli.register_action(help="foo", args=[
        ("fname",             dict(type=Path, help="where to write json")),
        ("-t", "--transpose", dict(action="store_true", help="transpose the output")),
        ("--single-shape",    dict(action="store_true", help="break after first shape")),
        ("--batch-size",      dict(type=int, default=40_000, help="tradeoff between vram usage and efficiency")),
        ("--n-cd",            dict(type=int, default=30_000, help="Number of points to use when computing chamfer distance")),
        ("--filter-outliers", dict(action="store_true", help="like in PRIF")),
    ])
    @torch.enable_grad()
    def compute_scores(args: Namespace, config: Munch, model: IField, **kw):
        datamodule_cls: RayFieldAdDataModuleBase = cli.get_datamodule_cls_from_config(args, config)
        model.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            model.to("cuda")

        def T(array: np.ndarray, **kw) -> torch.Tensor:
            if isinstance(array, torch.Tensor): return array
            return torch.tensor(array, device=model.device, dtype=model.dtype if isinstance(array, np.floating) else None, **kw)

        MEDIAL = model.hparams.output_mode == "medial_sphere"
        if not MEDIAL: assert model.hparams.output_mode == "orthogonal_plane"


        uids = sorted(model.keys())
        if args.single_shape: uids = [uids[0]]
        rich.print(f"{datamodule_cls.__name__            = }")
        rich.print(f"{len(uids)                          = }")

        # accumulators for IoU and F-Score, CD and COS

        # sum reduction:
        n            = defaultdict(int)
        n_gt_hits    = defaultdict(int)
        n_gt_miss    = defaultdict(int)
        n_gt_missing = defaultdict(int)
        n_outliers   = defaultdict(int)
        p_mse        = defaultdict(int)
        s_mse        = defaultdict(int)
        cossim_med   = defaultdict(int) # medial normals
        cossim_jac   = defaultdict(int) # jacovian normals
        TP,FN,FP,TN  = [defaultdict(int) for _ in range(4)] # IoU and f-score
        # mean reduction:
        cd_dist    = {} # chamfer distance
        cd_cos_med = {} # chamfer medial normals
        cd_cos_jac = {} # chamfer jacovian normals
        all_metrics = dict(
            n=n, n_gt_hits=n_gt_hits, n_gt_miss=n_gt_miss, n_gt_missing=n_gt_missing, p_mse=p_mse,
            cossim_jac=cossim_jac,
            TP=TP, FN=FN, FP=FP, TN=TN, cd_dist=cd_dist,
            cd_cos_jac=cd_cos_jac,
        )
        if MEDIAL:
            all_metrics["s_mse"]      = s_mse
            all_metrics["cossim_med"] = cossim_med
            all_metrics["cd_cos_med"] = cd_cos_med
        if args.filter_outliers:
            all_metrics["n_outliers"] = n_outliers

        t = datetime.now()
        for uid in tqdm(uids, desc="Dataset", position=0, leave=True, disable=len(uids)<=1):
            sphere_scan_gt = datamodule_cls.get_sphere_scan_from_uid(uid)

            z      = model[uid].detach()

            all_intersections    = []
            all_medial_normals   = []
            all_jacobian_normals = []

            step = args.batch_size
            for i in tqdm(range(0, sphere_scan_gt.hits.shape[0], step), desc=f"Item {uid!r}", position=1, leave=False):
                # prepare batch and gt
                origins      = T(sphere_scan_gt.cam_pos  [i:i+step, :], requires_grad = True)
                dirs         = T(sphere_scan_gt.ray_dirs [i:i+step, :])
                gt_hits      = T(sphere_scan_gt.hits     [i:i+step])
                gt_miss      = T(sphere_scan_gt.miss     [i:i+step])
                gt_missing   = T(sphere_scan_gt.missing  [i:i+step])
                gt_points    = T(sphere_scan_gt.points   [i:i+step, :])
                gt_normals   = T(sphere_scan_gt.normals  [i:i+step, :])
                gt_distances = T(sphere_scan_gt.distances[i:i+step])

                # forward
                if MEDIAL:
                    (
                        depths,
                        silhouettes,
                        intersections,
                        medial_normals,
                        is_intersecting,
                        sphere_centers,
                        sphere_radii,
                    ) = model({
                            "origins" : origins,
                            "dirs"    : dirs,
                        }, z, intersections_only=False, allow_nans=False)
                else:
                    silhouettes = medial_normals = None
                    intersections, is_intersecting = model({
                            "origins" : origins,
                            "dirs"    : dirs,
                        }, z, normalize_origins = True)
                    is_intersecting = is_intersecting > 0.5
                jac = diff.jacobian(intersections, origins, detach=True)

                # outlier removal (PRIF)
                if args.filter_outliers:
                    outliers = jac.norm(dim=-2).norm(dim=-1) > 5
                    n_outliers[uid] += outliers[is_intersecting].sum().item()
                    # We count filtered points as misses
                    is_intersecting &= ~outliers

                model.zero_grad()
                jacobian_normals = model.compute_normals_from_intersection_origin_jacobian(jac, dirs)

                all_intersections   .append(intersections   .detach()[is_intersecting.detach(), :])
                all_medial_normals  .append(medial_normals  .detach()[is_intersecting.detach(), :]) if MEDIAL else None
                all_jacobian_normals.append(jacobian_normals.detach()[is_intersecting.detach(), :])

                # accumulate metrics
                with torch.no_grad():
                    n                    [uid] += dirs.shape[0]
                    n_gt_hits            [uid] += gt_hits.sum().item()
                    n_gt_miss            [uid] += gt_miss.sum().item()
                    n_gt_missing         [uid] += gt_missing.sum().item()
                    p_mse                [uid] += (gt_points   [gt_hits, :] - intersections[gt_hits, :]).norm(2, dim=-1).pow(2).sum().item()
                    if MEDIAL: s_mse     [uid] += (gt_distances[gt_miss]    - silhouettes  [gt_miss]   )                .pow(2).sum().item()
                    if MEDIAL: cossim_med[uid] += (1-F.cosine_similarity(gt_normals[gt_hits, :], medial_normals  [gt_hits, :], dim=-1).abs()).sum().item() # to match what pytorch3d does for CD
                    cossim_jac           [uid] += (1-F.cosine_similarity(gt_normals[gt_hits, :], jacobian_normals[gt_hits, :], dim=-1).abs()).sum().item() # to match what pytorch3d does for CD
                    not_intersecting = ~is_intersecting
                    TP                   [uid] += ((gt_hits | gt_missing) &  is_intersecting).sum().item() # True  Positive
                    FN                   [uid] += ((gt_hits | gt_missing) & not_intersecting).sum().item() # False Negative
                    FP                   [uid] += (gt_miss                &  is_intersecting).sum().item() # False Positive
                    TN                   [uid] += (gt_miss                & not_intersecting).sum().item() # True  Negative

            all_intersections    = torch.cat(all_intersections,    dim=0)
            all_medial_normals   = torch.cat(all_medial_normals,   dim=0) if MEDIAL else None
            all_jacobian_normals = torch.cat(all_jacobian_normals, dim=0)

            hits = sphere_scan_gt.hits # brevity
            print()

            assert all_intersections.shape[0] >= args.n_cd
            idx_cd_pred  = torch.randperm(all_intersections.shape[0])[:args.n_cd]
            idx_cd_gt    = torch.randperm(hits.sum())                [:args.n_cd]

            print("cd... ", end="")
            tt = datetime.now()
            loss_cd, loss_cos_jac  = chamfer_distance(
                x         = all_intersections       [None, :,    :][:, idx_cd_pred, :].detach(),
                x_normals = all_jacobian_normals    [None, :,    :][:, idx_cd_pred, :].detach(),
                y         = T(sphere_scan_gt.points [None, hits, :][:, idx_cd_gt,   :]),
                y_normals = T(sphere_scan_gt.normals[None, hits, :][:, idx_cd_gt,   :]),
                batch_reduction = "sum", point_reduction = "sum",
            )
            if MEDIAL: _, loss_cos_med = chamfer_distance(
                x         = all_intersections       [None, :,    :][:, idx_cd_pred, :].detach(),
                x_normals = all_medial_normals      [None, :,    :][:, idx_cd_pred, :].detach(),
                y         = T(sphere_scan_gt.points [None, hits, :][:, idx_cd_gt,   :]),
                y_normals = T(sphere_scan_gt.normals[None, hits, :][:, idx_cd_gt,   :]),
                batch_reduction = "sum", point_reduction = "sum",
            )
            print(datetime.now() - tt)

            cd_dist    [uid] = loss_cd.item()
            cd_cos_med [uid] = loss_cos_med.item() if MEDIAL else None
            cd_cos_jac [uid] = loss_cos_jac.item()

        print()
        model.zero_grad(set_to_none=True)
        print("Total time:",    datetime.now() - t)
        print("Time per item:", (datetime.now() - t) / len(uids)) if len(uids) > 1 else None

        sum   = lambda *xs: builtins  .sum  (itertools.chain(*(x.values() for x in xs)))
        mean  = lambda *xs: statistics.mean (itertools.chain(*(x.values() for x in xs)))
        stdev = lambda *xs: statistics.stdev(itertools.chain(*(x.values() for x in xs)))
        n_cd  = args.n_cd
        P = sum(TP)/(sum(TP, FP))
        R = sum(TP)/(sum(TP, FN))
        print(f"{mean(n)                            = :11.1f}      (rays per object)")
        print(f"{mean(n_gt_hits)                    = :11.1f}      (gt rays hitting per object)")
        print(f"{mean(n_gt_miss)                    = :11.1f}      (gt rays missing per object)")
        print(f"{mean(n_gt_missing)                 = :11.1f}      (gt rays unknown per object)")
        print(f"{mean(n_outliers)                   = :11.1f}      (gt rays unknown per object)") if args.filter_outliers else None
        print(f"{n_cd                               = :11.0f}      (cd rays per object)")
        print(f"{mean(n_gt_hits)   / mean(n)        = :11.8f}      (fraction rays hitting per object)")
        print(f"{mean(n_gt_miss)   / mean(n)        = :11.8f}      (fraction rays missing per object)")
        print(f"{mean(n_gt_missing)/ mean(n)        = :11.8f}      (fraction rays unknown per object)")
        print(f"{mean(n_outliers)  / mean(n)        = :11.8f}      (fraction rays unknown per object)") if args.filter_outliers else None
        print(f"{sum(TP)/sum(n)                     = :11.8f}      (total ray TP)")
        print(f"{sum(TN)/sum(n)                     = :11.8f}      (total ray TN)")
        print(f"{sum(FP)/sum(n)                     = :11.8f}      (total ray FP)")
        print(f"{sum(FN)/sum(n)                     = :11.8f}      (total ray FN)")
        print(f"{sum(TP, FN, FP)/sum(n)             = :11.8f}      (total ray union)")
        print(f"{sum(TP)/sum(TP, FN, FP)            = :11.8f}      (total ray IoU)")
        print(f"{sum(TP)/(sum(TP, FP))              = :11.8f} -> P (total ray precision)")
        print(f"{sum(TP)/(sum(TP, FN))              = :11.8f} -> R (total ray recall)")
        print(f"{2*(P*R)/(P+R)                      = :11.8f}      (total ray F-score)")
        print(f"{sum(p_mse)/sum(n_gt_hits)          = :11.8f}      (mean ray intersection mean squared error)")
        print(f"{sum(s_mse)/sum(n_gt_miss)          = :11.8f}      (mean ray silhoutette  mean squared error)")
        print(f"{sum(cossim_med)/sum(n_gt_hits)     = :11.8f}      (mean ray medial reduced cosine similarity)") if MEDIAL else None
        print(f"{sum(cossim_jac)/sum(n_gt_hits)     = :11.8f}      (mean ray analytical reduced cosine similarity)")
        print(f"{mean(cd_dist)   /n_cd * 1e3        = :11.8f}      (mean chamfer distance)")
        print(f"{mean(cd_cos_med)/n_cd              = :11.8f}      (mean chamfer reduced medial cossim distance)") if MEDIAL else None
        print(f"{mean(cd_cos_jac)/n_cd              = :11.8f}      (mean chamfer reduced analytical cossim distance)")
        print(f"{stdev(cd_dist)   /n_cd * 1e3       = :11.8f}      (stdev chamfer distance)")                           if len(cd_dist) > 1    else None
        print(f"{stdev(cd_cos_med)/n_cd             = :11.8f}      (stdev chamfer reduced medial cossim distance)")     if len(cd_cos_med) > 1 and MEDIAL else None
        print(f"{stdev(cd_cos_jac)/n_cd             = :11.8f}      (stdev chamfer reduced analytical cossim distance)") if len(cd_cos_jac) > 1 else None

        if args.transpose:
            all_metrics, old_metrics = defaultdict(dict), all_metrics
            for m, table in old_metrics.items():
                for uid, vals in table.items():
                    all_metrics[uid][m] = vals
            all_metrics["_hparams"] = dict(n_cd=args.n_cd)
        else:
            all_metrics["n_cd"]  = args.n_cd

        if str(args.fname) == "-":
            print("{", ',\n'.join(
                f"  {json.dumps(k)}: {json.dumps(v)}"
                for k, v in all_metrics.items()
            ), "}", sep="\n")
        else:
            args.fname.parent.mkdir(parents=True, exist_ok=True)
            with args.fname.open("w") as f:
                json.dump(all_metrics, f, indent=2)

    return cli


if __name__ == "__main__":
    mk_cli().run()
