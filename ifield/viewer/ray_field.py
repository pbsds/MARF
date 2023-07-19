from ..data.common.scan import SingleViewUVScan
import mesh_to_sdf.scan as sdf_scan
from ..models import intersection_fields
from ..utils import geometry, helpers
from ..utils.operators import diff
from .common import InteractiveViewer
from matplotlib import cm
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
from typing import Hashable, Optional, Callable
from munch import Munch
import functools
import itertools
import numpy as np
import random
from pathlib import Path
import shutil
import subprocess
import torch
from trimesh import Trimesh
import trimesh.transformations as T


class ModelViewer(InteractiveViewer):
    lambertian_color = (1.0, 1.0, 1.0)
    max_cols = 200
    max_cols = 32

    def __init__(self,
            model         : intersection_fields.IntersectionFieldAutoDecoderModel,
            start_uid     : Hashable,
            skyward       : str                                  = "+Z",
            mesh_gt_getter: Callable[[Hashable], Trimesh] | None = None,
            *a, **kw):
        self.model = model
        self.model.eval()
        self.current_uid = self._prev_uid = start_uid
        self.all_uids = list(model.keys())

        self.mesh_gt_getter = mesh_gt_getter
        self.current_gt_mesh: tuple[Hashable, Trimesh] = (None, None)

        self.display_mode_normals   = self.vizmodes_normals  .index("medial" if self.model.hparams.output_mode == "medial_sphere" else "analytical")
        self.display_mode_shading   = self.vizmodes_shading  .index("lambertian")
        self.display_mode_centroid  = self.vizmodes_centroids.index("best-centroids-colored")
        self.display_mode_spheres   = self.vizmodes_spheres  .index(None)
        self.display_mode_variation = 0

        self.display_sphere_map_bg = True
        self.atom_radius_offset = 0
        self.atom_index_solo = None
        self.export_medial_surface_mesh = False

        self.light_angle1 = 0
        self.light_angle2 = 0

        self.obj_rot = {
            "-X": torch.tensor(T.rotation_matrix(angle= np.pi/2, direction=(0, 1, 0))[:3, :3], **model.device_and_dtype).T,
            "+X": torch.tensor(T.rotation_matrix(angle=-np.pi/2, direction=(0, 1, 0))[:3, :3], **model.device_and_dtype).T,
            "-Y": torch.tensor(T.rotation_matrix(angle= np.pi/2, direction=(1, 0, 0))[:3, :3], **model.device_and_dtype).T,
            "+Y": torch.tensor(T.rotation_matrix(angle=-np.pi/2, direction=(1, 0, 0))[:3, :3], **model.device_and_dtype).T,
            "-Z": torch.tensor(T.rotation_matrix(angle= np.pi,   direction=(1, 0, 0))[:3, :3], **model.device_and_dtype).T,
            "+Z": torch.eye(3, **model.device_and_dtype),
        }[str(skyward).upper()]
        self.obj_rot_inv = torch.linalg.inv(self.obj_rot)

        super().__init__(*a, **kw)

    vizmodes_normals = (
        "medial",
        "analytical",
        "ground_truth",
    )
    vizmodes_shading = (
        None, # just atoms or medial axis
        "colored-lambertian",
        "lambertian",
        "shade-best-radii",
        "shade-all-radii",
        "translucent",
        "normal",
        "centroid-grad-norm",   # backprop
        "anisotropic",          # backprop
        "curvature",            # backprop
        "glass",
        "double-glass",
    )
    vizmodes_centroids = (
        None,
        "best-centroids",
        "all-centroids",
        "best-centroids-colored",
        "all-centroids-colored",
        "miss-centroids-colored",
        "all-miss-centroids-colored",
    )
    vizmodes_spheres = (
        None,
        "intersecting-sphere",
        "intersecting-sphere-colored",
        "best-sphere",
        "best-sphere-colored",
        "all-spheres-colored",
    )

    def get_display_mode(self) -> tuple[str, str, Optional[str], Optional[str]]:
        MARF = self.model.hparams.output_mode == "medial_sphere"
        if isinstance(self.display_mode_normals,  str): self.display_mode_normals  = self.vizmodes_shading  .index(self.display_mode_normals)
        if isinstance(self.display_mode_shading,  str): self.display_mode_shading  = self.vizmodes_shading  .index(self.display_mode_shading)
        if isinstance(self.display_mode_centroid, str): self.display_mode_centroid = self.vizmodes_centroids.index(self.display_mode_centroid)
        if isinstance(self.display_mode_spheres,  str): self.display_mode_spheres  = self.vizmodes_spheres  .index(self.display_mode_spheres)
        out = (
            self.vizmodes_normals  [self.display_mode_normals  % len(self.vizmodes_normals)],
            self.vizmodes_shading  [self.display_mode_shading  % len(self.vizmodes_shading)],
            self.vizmodes_centroids[self.display_mode_centroid % len(self.vizmodes_centroids)] if MARF else None,
            self.vizmodes_spheres  [self.display_mode_spheres  % len(self.vizmodes_spheres)]   if MARF else None,
        )
        self.set_caption(" & ".join(i for i in out if i is not None))
        return out

    @property
    def cam_state(self):
        return super().cam_state | {
            "light_angle1" : self.light_angle1,
            "light_angle2" : self.light_angle2,
        }

    @cam_state.setter
    def cam_state(self, new_state):
        InteractiveViewer.cam_state.fset(self, new_state)
        self.light_angle1 = new_state.get("light_angle1", self.light_angle1)
        self.light_angle2 = new_state.get("light_angle2", self.light_angle2)

    def get_current_conditioning(self) -> Optional[torch.Tensor]:
        if not self.model.is_conditioned:
            return None

        prev_uid    = self._prev_uid                        # to determine if target has changed
        next_z      = self.model[prev_uid].detach()         # interpolation target
        prev_z      = getattr(self, "_prev_z",      next_z) # interpolation source
        epoch       = getattr(self, "_prev_epoch",  0)      # interpolation factor

        if not self.is_headless:
            now = self.t
            t = (now - epoch) / 1 # 1 second
        else:
            now = self.frame_idx
            t = (now - epoch) / self.fps # 1 second
        assert t >= 0

        if t < 1:
            next_z = next_z*t + prev_z*(1-t)

        if prev_uid != self.current_uid:
            self._prev_uid    = self.current_uid
            self._prev_z      = next_z
            self._prev_epoch  = now

        return next_z

    def get_current_ground_truth(self) -> Trimesh | None:
        if self.mesh_gt_getter is None:
            return None
        uid, mesh = self.current_gt_mesh
        try:
            if uid != self.current_uid:
                print("Loading ground truth mesh...")
                mesh = self.mesh_gt_getter(self.current_uid)
                self.current_gt_mesh = self.current_uid, mesh
        except NotImplementedError:
            self.current_gt_mesh = self.current_uid, None
            return None
        return mesh

    def handle_keys_pressed(self, pressed):
        td = super().handle_keys_pressed(pressed)
        mod = pressed[self.constants.K_LALT] or pressed[self.constants.K_RALT]
        if not mod and pressed[self.constants.K_f]: self.light_angle1 -= td * 0.5
        if not mod and pressed[self.constants.K_g]: self.light_angle1 += td * 0.5
        if     mod and pressed[self.constants.K_f]: self.light_angle2 += td * 0.5
        if     mod and pressed[self.constants.K_g]: self.light_angle2 -= td * 0.5
        return td

    def handle_key_down(self, key, keys_pressed):
        super().handle_key_down(key, keys_pressed)
        shift = keys_pressed[self.constants.K_LSHIFT] or keys_pressed[self.constants.K_RSHIFT]
        if key == self.constants.K_o:
            i = self.all_uids.index(self.current_uid)
            i = (i - 1) % len(self.all_uids)
            self.current_uid = self.all_uids[i]
            print(self.current_uid)
        if key == self.constants.K_p:
            i = self.all_uids.index(self.current_uid)
            i = (i + 1) % len(self.all_uids)
            self.current_uid = self.all_uids[i]
            print(self.current_uid)
        if key == self.constants.K_SPACE:
            self.display_sphere_map_bg = {
                True : 255,
                255  : 0,
                0    : True,
            }.get(self.display_sphere_map_bg, True)
        if key == self.constants.K_u: self.export_medial_surface_mesh = True
        if key == self.constants.K_x: self.display_mode_normals  += -1 if shift else 1
        if key == self.constants.K_c: self.display_mode_shading  += -1 if shift else 1
        if key == self.constants.K_v: self.display_mode_centroid += -1 if shift else 1
        if key == self.constants.K_b: self.display_mode_spheres  += -1 if shift else 1
        if key == self.constants.K_e: self.display_mode_variation+= -1 if shift else 1
        if key == self.constants.K_c: self.display_mode_variation = 0
        if key == self.constants.K_0: self.atom_index_solo = None
        if key == self.constants.K_1: self.atom_index_solo = 0 if self.atom_index_solo != 0 else None
        if key == self.constants.K_2: self.atom_index_solo = 1 if self.atom_index_solo != 1 else None
        if key == self.constants.K_3: self.atom_index_solo = 2 if self.atom_index_solo != 2 else None
        if key == self.constants.K_4: self.atom_index_solo = 3 if self.atom_index_solo != 3 else None
        if key == self.constants.K_5: self.atom_index_solo = 4 if self.atom_index_solo != 4 else None
        if key == self.constants.K_6: self.atom_index_solo = 5 if self.atom_index_solo != 5 else None
        if key == self.constants.K_7: self.atom_index_solo = 6 if self.atom_index_solo != 6 else None
        if key == self.constants.K_8: self.atom_index_solo = 7 if self.atom_index_solo != 7 else None
        if key == self.constants.K_9: self.atom_index_solo = self.atom_index_solo + (-1 if shift else 1) if self.atom_index_solo is not None else 0

    def handle_mouse_button_down(self, pos, button, keys_pressed):
        super().handle_mouse_button_down(pos, button, keys_pressed)
        if button in (1, 3):
            self.display_mode_spheres += 1 if button == 1 else -1

    def handle_mousewheel(self, flipped, x, y, keys_pressed):
        shift = keys_pressed[self.constants.K_LSHIFT] or keys_pressed[self.constants.K_RSHIFT]
        if not shift:
            super().handle_mousewheel(flipped, x, y, keys_pressed)
        else:
            self.atom_radius_offset += 0.005 * y
            print()
            print("atom_radius_offset:", self.atom_radius_offset)

    def setup(self):
        if not self.is_headless:
            print(dedent("""
                WASD + PG Up/Down - translate
                ARROWS            - rotate

                (SHIFT+) C        - Next/(Prev) shading mode
                (SHIFT+) V        - Next/(Prev) centroids mode
                (SHIFT+) B        - Next/(Prev) sphere mode
                Mouse L/ R        - Next/ Prev  sphere mode
                (SHIFT+) E        - Next/(Prev) variation (for quick experimentation within a shading mode)
                SHIFT + Scroll    - Offset atom radius
                ALT + Scroll      - Modify FoV       (_true_ zoom)
                Mouse Scroll      - Translate in/out ("zoom", moves camera to/from to point of focus)
                Alt+PG Up/Down    - Translate in/out ("zoom", moves camera to/from to point of focus)

                F / G             - rotate light left / right
                ALT+ F / G        - rotate light up / down
                CTRL / SHIFT      - faster/slower rotation
                O / P             - prev/next object
                1-9               - solo atom
                0                 - show all atoms
                + / -             - decrease/increase pixel scale
                R                 - rotate continuously
                H / SHIFT+H / CTRL+H - load/save/print camera state
                Enter             - reset camera state
                Y                 - save screenshot
                U                 - save mesh of centroids
                Space             - cycle sphere map background
                Q                 - quit
            """).strip())

        fname = Path(__file__).parent.resolve() / "assets/texturify_pano-1-4.jpg"
        self.load_sphere_map(fname)

        if self.model.hparams.output_mode == "medial_sphere":
            @self.model.net.register_forward_hook
            def atom_offset_radius_and_solo(model, input, output):
                slice = (..., [i+3 for i in range(0, output.shape[-1], 4)])
                output[slice] += self.atom_radius_offset * output[slice].sign()
                if self.atom_index_solo is not None:
                    x = self.atom_index_solo * 4
                    x = x % output.shape[-1]
                    output = output[..., list(range(x, x+4))]
                return output
            self._atom_offset_radius_and_solo_hook = atom_offset_radius_and_solo

    def teardown(self):
        if hasattr(self, "_atom_offset_radius_and_solo_hook"):
            self._atom_offset_radius_and_solo_hook.remove()
            del self._atom_offset_radius_and_solo_hook

    @torch.no_grad()
    def render_frame(self, pixel_view: np.ndarray): # (W, H, 3) dtype=uint8
        MARF = self.model.hparams.output_mode == "medial_sphere"
        PRIF = self.model.hparams.output_mode == "orthogonal_plane"
        assert (MARF or PRIF) and MARF != PRIF
        device_and_dtype = self.model.device_and_dtype
        device           = self.model.device
        dtype            = self.model.dtype

        (
            vizmode_normals,
            vizmode_shading,
            vizmode_centroids,
            vizmode_spheres,
        ) = self.get_display_mode()

        dirs, origins = self.raydirs_and_cam
        origins = origins.detach().clone().to(**device_and_dtype)
        dirs    = dirs   .detach().clone().to(**device_and_dtype)

        if vizmode_normals != "ground_truth" or self.get_current_ground_truth() is None:

            # enable grad or not
            do_jac            = PRIF or vizmode_normals == "analytical"
            do_jac_medial     = MARF and "centroid-grad-norm" in (vizmode_shading or "")
            do_shape_operator = "anisotropic" in (vizmode_shading or "") or "curvature" in (vizmode_shading or "")
            do_grad = do_jac or do_jac_medial or do_shape_operator
            if do_grad:
                origins = origins.broadcast_to(dirs.shape)

            self.model.eval()
            latent = self.get_current_conditioning()
            if self.max_cols is None or self.max_cols > dirs.shape[0]:
                chunks = [slice(None)]
            else:
                chunks = [slice(col, col+self.max_cols) for col in range(0, dirs.shape[0], self.max_cols)]
            forward_chunks = []
            for chunk in chunks:
                self.model.zero_grad()
                origins_chunk = origins[chunk if origins.ndim != 1 else slice(None)] @ self.obj_rot
                dirs_chunk    = dirs   [chunk]                                       @ self.obj_rot
                if do_grad:
                    origins_chunk.requires_grad = dirs_chunk.requires_grad = True

                @forward_chunks.append
                @(lambda f: f(origins_chunk, dirs_chunk))
                @torch.set_grad_enabled(do_grad)
                def forward_chunk(origins, dirs) -> Munch:
                    if PRIF:
                        intersections, is_intersecting = self.model(dict(origins=origins, dirs=dirs), z=latent, normalize_origins=True)
                        is_intersecting = is_intersecting > 0.5
                    elif MARF:
                        (
                            depths, silhouettes, intersections,
                            intersection_normals, is_intersecting,
                            sphere_centers, sphere_radii,

                            atom_indices,
                            all_intersections, all_intersection_normals, all_depths, all_silhouettes, all_is_intersecting,
                            all_sphere_centers, all_sphere_radii,
                        ) = self.model.forward(dict(origins=origins, dirs=dirs), z=latent,
                            intersections_only = False,
                            return_all_atoms   = True,
                        )

                    if do_jac:
                        jac = diff.jacobian(intersections, origins, detach=not do_shape_operator)
                        intersection_normals = self.model.compute_normals_from_intersection_origin_jacobian(jac, dirs.detach())

                    if do_jac_medial:
                        sphere_centers_jac = diff.jacobian(sphere_centers, origins, detach=True)

                    if do_shape_operator:
                        hess = diff.jacobian(intersection_normals, origins, detach=True)[is_intersecting, :, :]
                        N = intersection_normals.detach()[is_intersecting, :]
                        TM = (torch.eye(3, device=device) - N[..., None, :]*N[..., :, None]) # projection onto tangent plane
                        # shape operator, i.e. total derivative of the surface normal w.r.t. the tangent space
                        shape_operator = hess @ TM

                    return Munch((k, v.detach()) for k, v in locals().items() if isinstance(v, torch.Tensor))

            intersections          = torch.cat([chunk.intersections        for chunk in forward_chunks], dim=0)
            is_intersecting        = torch.cat([chunk.is_intersecting      for chunk in forward_chunks], dim=0)
            intersection_normals   = torch.cat([chunk.intersection_normals for chunk in forward_chunks], dim=0)
            if MARF:
                all_sphere_centers = torch.cat([chunk.all_sphere_centers   for chunk in forward_chunks], dim=0)
                all_sphere_radii   = torch.cat([chunk.all_sphere_radii     for chunk in forward_chunks], dim=0)
                atom_indices       = torch.cat([chunk.atom_indices         for chunk in forward_chunks], dim=0)
                silhouettes        = torch.cat([chunk.silhouettes          for chunk in forward_chunks], dim=0)
                sphere_centers     = torch.cat([chunk.sphere_centers       for chunk in forward_chunks], dim=0)
                sphere_radii       = torch.cat([chunk.sphere_radii         for chunk in forward_chunks], dim=0)
            if do_jac_medial:
                sphere_centers_jac = torch.cat([chunk.sphere_centers_jac   for chunk in forward_chunks], dim=0)
            if do_shape_operator:
                shape_operator     = torch.cat([chunk.shape_operator       for chunk in forward_chunks], dim=0)

            n_atoms = all_sphere_centers.shape[-2] if MARF else 1

            intersections            = intersections            @ self.obj_rot_inv
            intersection_normals     = intersection_normals     @ self.obj_rot_inv
            sphere_centers           = sphere_centers           @ self.obj_rot_inv if sphere_centers           is not None else None
            all_sphere_centers       = all_sphere_centers       @ self.obj_rot_inv if all_sphere_centers       is not None else None

        else: # render ground truth mesh
            # HACK: we use a thread to not break the pygame opengl context
            with ThreadPoolExecutor(max_workers=1) as p:
                scan = p.submit(sdf_scan.Scan, self.get_current_ground_truth(),
                    camera_transform  = self.cam2world.numpy(),
                    resolution        = self.res[1],
                    calculate_normals = True,
                    fov               = self.cam_fov_y,
                    z_near            = 0.001,
                    z_far             = 50,
                    no_flip_backfaced_normals = True
                ).result()
            n_atoms, MARF, PRIF = 1, False, True
            is_intersecting = torch.zeros(self.res, dtype=bool)
            is_intersecting[ (self.res[0]-self.res[1]) // 2 : (self.res[0]-self.res[1]) // 2 + self.res[1], : ] = torch.tensor(scan.depth_buffer != 0, dtype=bool)
            intersections        = torch.zeros((*is_intersecting.shape, 3), dtype=dtype)
            intersection_normals = torch.zeros((*is_intersecting.shape, 3), dtype=dtype)
            intersections       [is_intersecting] = torch.tensor(scan.points,  dtype=dtype)
            intersection_normals[is_intersecting] = torch.tensor(scan.normals, dtype=dtype)
            is_intersecting      = is_intersecting     .flip(1).to(device)
            intersections        = intersections       .flip(1).to(device)
            intersection_normals = intersection_normals.flip(1).to(device)

        mask    = is_intersecting.cpu()

        mx, my = self.mouse_position
        w,  h  = dirs.shape[:2]

        # fill white
        if self.display_sphere_map_bg == True:
            self.blit_sphere_map_mask(pixel_view)
        else:
            pixel_view[:] = self.display_sphere_map_bg

        # draw to buffer

        to_cam   = -dirs.detach()

        # light direction
        extra = np.pi if vizmode_shading == "translucent" else 0
        LM = torch.tensor(T.rotation_matrix(angle=self.light_angle2,         direction=(0, 1, 0))[:3, :3], dtype=dtype)
        LM = torch.tensor(T.rotation_matrix(angle=self.light_angle1 + extra, direction=(1, 0, 0))[:3, :3], dtype=dtype) @ LM
        to_light = (self.cam2world[:3, :3] @ LM @ torch.tensor((1, 1, 3), dtype=dtype)).to(device)[None, :]
        to_light = to_light / to_light.norm(dim=-1, keepdim=True)

        # used to color different atom candidates
        color_set = tuple(map(helpers.hex2tuple,
            itertools.chain(
                mcolors.TABLEAU_COLORS.values(),
                #list(mcolors.TABLEAU_COLORS.values())[::-1],
                #['#f8481c', '#c20078', '#35530a', '#010844', '#a8ff04'],
                mcolors.XKCD_COLORS.values(),
            )
        ))
        color_per_atom = (*zip(*zip(range(n_atoms), itertools.cycle(color_set))),)[1]


        # shade hits

        if vizmode_shading is None:
            pass
        elif vizmode_shading == "colored-lambertian":
            if n_atoms > 1:
                color = torch.tensor(color_per_atom, device=device)[(*atom_indices[is_intersecting].T,)]
            else:
                color = torch.tensor(color_set[(0 if self.atom_index_solo is None else self.atom_index_solo) % len(color_set)], device=device)
            lambertian = torch.einsum("id,id->i",
                intersection_normals[is_intersecting, :],
                to_light,
            )[..., None]

            pixel_view[mask, :] = (color *
                torch.einsum("id,id->i",
                    intersection_normals[is_intersecting, :],
                    to_cam[is_intersecting, :],
                )[..., None]).int().cpu()
            pixel_view[mask, :] = (
                255   * lambertian.clamp(0, 1).pow(32) +
                color * (lambertian + 0.25).clamp(0, 1) * (1-lambertian.clamp(0, 1).pow(32))
            ).cpu()
        elif vizmode_shading == "lambertian":
            lambertian = torch.einsum("id,id->i",
                intersection_normals[is_intersecting, :],
                to_light,
            )[..., None].clamp(0, 1)

            if self.lambertian_color == (1.0, 1.0, 1.0):
                pixel_view[mask, :] = (255 * lambertian).cpu()
            else:
                color = 255*torch.tensor(self.lambertian_color, device=device)
                pixel_view[mask, :] = (color *
                    torch.einsum("id,id->i",
                        intersection_normals[is_intersecting, :],
                        to_cam[is_intersecting, :],
                    )[..., None]).int().cpu()
                pixel_view[mask, :] = (
                    255   * lambertian.clamp(0, 1).pow(32) +
                    color * (lambertian + 0.25).clamp(0, 1) * (1-lambertian.clamp(0, 1).pow(32))
                ).cpu()
        elif vizmode_shading == "translucent" and MARF:
            lambertian = torch.einsum("id,id->i",
                intersection_normals[is_intersecting, :],
                to_light,
            )[..., None].abs().clamp(0, 1)

            distortion = 0.08
            power = 16
            ambient = 0
            thickness = sphere_radii[is_intersecting].detach()
            if self.display_mode_variation % 2:
                thickness = thickness.mean()

            color1 = torch.tensor((1, 0.5, 0.5), **device_and_dtype) # subsurface
            color2 = torch.tensor((0,   1,   1), **device_and_dtype) # diffuse

            l = to_light + intersection_normals[is_intersecting, :] * distortion
            d = (to_cam[is_intersecting, :] * -l).sum(dim=-1).clamp(0, None).pow(power)
            f = (d + ambient) * (1/(0.05 + thickness))

            pixel_view[((dirs * to_light).sum(dim=-1) > 0.99).cpu(), :] = 255 # draw light source

            pixel_view[mask, :] = (255 * (
                color2 * (0.05 + lambertian*0.15) +
                color1 * 0.3 * f[..., None]
            ).clamp(0, 1)).cpu()
        elif vizmode_shading == "anisotropic" and vizmode_normals != "ground_truth":
            eigvals, eigvecs = torch.linalg.eig(shape_operator.mT) # slow, complex output, not sorted
            eigvals, indices = eigvals.abs().sort(dim=-1)
            eigvecs = (eigvecs.abs() * eigvecs.real.sign()).take_along_dim(indices[..., None, :], dim=-1)
            eigvecs = eigvecs.mT

            s = self.display_mode_variation % 5
            if s in (0, 1):
                # try to keep these below 0.2:
                if s == 0: a1, a2 = 0.05, 0.3
                if s == 1: a1, a2 = 0.3, 0.05

                # == Ward anisotropic specular reflectance ==

                # G.J. Ward, Measuring and modeling anisotropic reflection, in:
                # Proceedings of the 19th Annual Conference on Computer Graphics and
                # Interactive Techniques, 1992: pp. 265–272.

                eigvecs /= eigvecs.norm(dim=-1, keepdim=True)

                N = intersection_normals[is_intersecting, :]
                H = to_cam[is_intersecting, :] + to_light
                H = H / H.norm(dim=-1, keepdim=True)
                specular = (1/(4*torch.pi * a1*a2 * torch.sqrt((
                    (N * to_cam[is_intersecting, :]).sum(dim=-1) *
                    (N * to_light                  ).sum(dim=-1)
                )))) * torch.exp(
                    -2 * (
                        ((H * eigvecs[..., 2, :]).sum(dim=-1) / a1).pow(2)
                        +
                        ((H * eigvecs[..., 1, :]).sum(dim=-1) / a2).pow(2)
                    ) / (
                        1 + (N * H).sum(dim=-1)
                    )
                )
                specular = specular.clamp(0, None).nan_to_num(0, 0, 0)
                lambertian = torch.einsum("id,id->i", N, to_light ).clamp(0, None)

                color1 = 0.4 * torch.tensor((1,   1,   1), **device_and_dtype) # specular
                color2 = 0.4 * torch.tensor((0,   1,   1), **device_and_dtype) # diffuse
                pixel_view[mask, :] = (255 * (
                    color1 * specular  [..., None] +
                    color2 * lambertian[..., None]
                ).clamp(0, 1)).int().cpu()
            if s == 2:
                pixel_view[mask, :] = (255 * (
                    eigvecs[..., 2, :].abs().clamp(0, 1) # orientation only
                )).int().cpu()
            elif s == 3:
                pixel_view[mask, :] = (255 * (
                    eigvecs[..., 1, :].abs().clamp(0, 1) # orientation only
                )).int().cpu()
            elif s == 4:
                pixel_view[mask, :] = (255 * (
                    eigvecs[..., 0, :].abs().clamp(0, 1) # orientation only
                )).int().cpu()
        elif vizmode_shading == "shade-best-radii" and MARF:
            lambertian = torch.einsum("id,id->i",
                intersection_normals[is_intersecting, :],
                to_light,
            )[..., None]

            radii = sphere_radii[is_intersecting]
            radii = radii - 0.04
            radii = radii / 0.4

            colors = cm.plasma(radii.clamp(0, 1).cpu())[..., :3]
            pixel_view[mask, :] = 255 * (
                lambertian.pow(32).clamp(0, 1).cpu().numpy() +
                colors * (lambertian + 0.25).clamp(0, 1).cpu().numpy() * (1-lambertian.pow(32).clamp(0, 1)).cpu().numpy()
            )
        elif vizmode_shading == "shade-all-radii" and MARF:
            radii = sphere_radii[is_intersecting][..., None]
            radii /= radii.max()
            if n_atoms > 1:
                color = torch.tensor(color_per_atom, device=device)[(*atom_indices[is_intersecting].T,)]
            else:
                color = torch.tensor(color_set[(0 if self.atom_index_solo is None else self.atom_index_solo) % len(color_set)], device=device)
            pixel_view[mask, :] = (color * radii).int().cpu()
        elif vizmode_shading == "normal":
            normal = intersection_normals[is_intersecting, :]
            pixel_view[mask, :] = (255 * (normal * 0.5 + 0.5) ).int().cpu()
        elif vizmode_shading == "curvature" and vizmode_normals != "ground_truth":
            eigvals = torch.linalg.eigvals(shape_operator.mT) # complex output, not sorted

            # we sort them by absolute magnitude, not the real component
            _, indices = (eigvals.abs() * eigvals.real.sign()).sort(dim=-1)
            eigvals = eigvals.real.take_along_dim(indices, dim=-1)

            s = self.display_mode_variation % (6 if MARF else 5)
            if s==0: out = (eigvals[..., [0, 2]].mean(dim=-1, keepdim=True) / 25).tanh() # mean curvature
            if s==1: out = (eigvals[..., [0, 2]].prod(dim=-1, keepdim=True) / 25).tanh() # gaussian curvature
            if s==2: out = (eigvals[..., [2]] / 25).tanh() # maximum principal curvature - k1
            if s==3: out = (eigvals[..., [1]] / 25).tanh() # some curvature
            if s==4: out = (eigvals[..., [0]] / 25).tanh() # minimum principal curvature - k2
            if s==5: out = ((sphere_radii[is_intersecting][..., None].detach() - 1 / eigvals[..., [2]].clamp(1e-8, None)) * 5).tanh().clamp(0, None)

            lambertian = torch.einsum("id,id->i",
                intersection_normals[is_intersecting, :],
                to_light,
            )[..., None]

            pixel_view[mask, :] = (255 * (lambertian+0.5).clamp(0, 1) * torch.cat((
                1+out.clamp(-1, 0),
                1-out.abs(),
                1-out.clamp(0, 1),
            ), dim=-1)).int().cpu()
        elif vizmode_shading == "centroid-grad-norm" and MARF:
            asd = sphere_centers_jac[is_intersecting, :, :].norm(dim=-2).mean(dim=-1, keepdim=True)
            asd -= asd.min()
            asd /= asd.max()
            pixel_view[mask, :] = (255 * asd).cpu()
        elif "glass" in vizmode_shading:
            normals = intersection_normals[is_intersecting, :]
            to_cam_ = to_cam              [is_intersecting, :]
            # "Empiricial Approximation" of fresnel
            # https://developer.download.nvidia.com/CgTutorial/cg_tutorial_chapter07.html via
            # http://kylehalladay.com/blog/tutorial/2014/02/18/Fresnel-Shaders-From-The-Ground-Up.html
            cos = torch.einsum("id,id->i", normals, to_cam_ )[..., None]
            bias, scale, power = 0, 4, 3
            fresnel = (bias + scale*(1-cos)**power).clamp(0, 1)

            #reflection
            reflection = -to_cam_ - 2*(-cos)*normals

            #refraction
            r = 1 / 1.5 # refractive index, air -> glass
            refraction = -r*to_cam_ + (r*cos - (1-r**2*(1-cos**2)).sqrt()) * normals
            exit_point = intersections[is_intersecting, :]

            # reflect the refraction over the plane defined by the refraction direction and the sphere center, resulting in the second refraction
            if vizmode_shading == "double-glass" and MARF:
                cos2 = torch.einsum("id,id->i", refraction, -to_cam_ )[..., None]
                pn = -to_cam_ - cos2*refraction
                pn /= pn.norm(dim=-1, keepdim=True)

                refraction = -to_cam_ - 2*torch.einsum("id,id->i", pn, -to_cam_ )[..., None]*pn

                exit_point -= sphere_centers[is_intersecting, :]
                exit_point = exit_point - 2*torch.einsum("id,id->i", pn, exit_point )[..., None]*pn
                exit_point += sphere_centers[is_intersecting, :]

            fresnel = np.asanyarray(fresnel.cpu())
            pixel_view[mask, :] \
                = self.lookup_sphere_map_dirs(reflection, intersections[is_intersecting, :]) * fresnel \
                + self.lookup_sphere_map_dirs(refraction, exit_point) * (1-fresnel)
        else: # flat
            pixel_view[mask, :] = 80

        if not MARF: return

        # overlay medial atoms

        if vizmode_spheres is not None:
            # show miss distance in red
            s = silhouettes.detach()[~is_intersecting].clamp(0, 1)
            s /= s.max()
            pixel_view[~mask, 1] = (s * 255).cpu()
            pixel_view[:, 2] = pixel_view[:, 1]

            mouse_hits = 0 <= mx < w and 0 <= my < h and mask[mx, my]
            draw_intersecting = "intersecting-sphere" in vizmode_spheres
            draw_best         = "best-sphere"         in vizmode_spheres
            draw_color        = "-sphere-colored"     in vizmode_spheres
            draw_all          = "all-spheres-colored" in vizmode_spheres

            def get_nears():
                if draw_all:
                    projected, near, far, is_intersecting = geometry.ray_sphere_intersect(
                        torch.tensor(origins),
                        torch.tensor(dirs[..., None, :]),
                        sphere_centers = all_sphere_centers[mx, my][None, None, ...],
                        sphere_radii   = all_sphere_radii  [mx, my][None, None, ...],
                        allow_nans     = False,
                        return_parts   = True,
                    )

                    depths          = (near - origins).norm(dim=-1)
                    atom_indices_   = torch.where(is_intersecting, depths.detach(), depths.detach()+100).argmin(dim=-1, keepdim=True)
                    is_intersecting = is_intersecting.any(dim=-1)
                    projected       = None
                    near            = near.take_along_dim(atom_indices_[..., None], -2).squeeze(-2)
                    far             = None
                    sphere_centers_ = all_sphere_centers[mx, my][None, None, ...].take_along_dim(atom_indices_[..., None], -2).squeeze(-2)

                    normals = near[is_intersecting, :] - sphere_centers_[is_intersecting, :]
                    normals /= torch.linalg.norm(normals, dim=-1)[..., None]

                    color = torch.tensor(color_per_atom, device=device)[(*atom_indices_[is_intersecting].T,)]
                    yield color, projected, near, far, is_intersecting, normals

                if (mouse_hits and draw_intersecting) or draw_best:
                    projected, near, far, is_intersecting = geometry.ray_sphere_intersect(
                        torch.tensor(origins),
                        torch.tensor(dirs),
                        # unit-sphere by default
                        sphere_centers = sphere_centers[mx, my][None, None, ...],
                        sphere_radii   = sphere_radii  [mx, my][None, None, ...],
                        return_parts   = True,
                    )

                    normals = near[is_intersecting, :] - sphere_centers[mx, my][None, ...]
                    normals /= torch.linalg.norm(normals, dim=-1)[..., None]
                    color = (255, 255, 255) if not draw_color else color_per_atom[atom_indices[mx, my]]
                    yield torch.tensor(color, device=device), projected, near, far, is_intersecting, normals

            # draw sphere with lambertian shading
            for color, projected, near, far, is_intersecting_2, normals in get_nears():
                lambertian = torch.einsum("...id,...id->...i", normals, to_light )[..., None]
                pixel_view[is_intersecting_2.cpu(), :] = (
                    255*lambertian.pow(32).clamp(0, 1) +
                    color * (lambertian + 0.25).clamp(0, 1) * (1-lambertian.pow(32).clamp(0, 1))
                ).cpu()

        # overlay points / sphere centers

        if vizmode_centroids is not None:
            cam2world_inv = torch.tensor(self.cam2world_inv, **device_and_dtype)
            intrinsics    = torch.tensor(self.intrinsics,    **device_and_dtype)

            def get_coords():
                miss_centroid = "miss-centroids"    in vizmode_centroids
                mask = is_intersecting if not miss_centroid else ~is_intersecting
                if vizmode_centroids in ("all-centroids-colored", "all-miss-centroids-colored"):
                    # we use temporal dithering to the show all overlapping centers
                    for color, atom_index in sorted(zip(itertools.chain(color_set), range(n_atoms)), key=lambda x: random.random()):
                        yield color, all_sphere_centers[..., atom_index, :][mask], mask
                elif "all-centroids" in vizmode_centroids:
                    yield (80, 150, 80), all_sphere_centers[mask].reshape(-1, 3), mask # [:, 3]

                if "centroids-colored" in vizmode_centroids:
                    if n_atoms == 1:
                        color = color_set[(0 if self.atom_index_solo is None else self.atom_index_solo) % len(color_set)]
                    else:
                        color = torch.tensor(color_per_atom, device=device)[(*atom_indices[mask].T,)].cpu()
                else:
                    color = (0, 0, 0)
                yield color, sphere_centers[mask], mask

            for i, (color, coords, coord_mask) in enumerate(get_coords()):
                if self.export_medial_surface_mesh:
                    fname = self.mk_dump_fname("ply", uid=i)
                    p = torch.zeros_like(sphere_centers)
                    c = torch.zeros_like(sphere_centers)
                    p[coord_mask, :] = coords
                    c[coord_mask, :] = torch.tensor(color, device=p.device) / 255
                    SingleViewUVScan(
                        hits   = ( mask).numpy(),
                        miss   = (~mask).numpy(),
                        points = p.cpu().numpy(),
                        colors = c.cpu().numpy(),
                        normals=None, distances=None, cam_pos=None,
                        cam_mat4=None, proj_mat4=None, transforms=None,
                    ).to_mesh().export(str(fname), file_type="ply")
                    print("dumped", fname)
                    if shutil.which("f3d"):
                        subprocess.Popen(["f3d", "-gsy", "--up=+z", "--bg-color=1,1,1", fname], close_fds=True)

                coords = torch.cat((coords, torch.ones((*coords.shape[:-1], 1), **device_and_dtype)), dim=-1)

                coords = torch.einsum("...ij,...kj->...ki", cam2world_inv, coords)[..., :3]
                coords = geometry.project(coords[..., 0], coords[..., 1], coords[..., 2], intrinsics)

                in_view = functools.reduce(torch.mul, (
                    coords[:, 0] <  pixel_view.shape[1],
                    coords[:, 0] >= 0,
                    coords[:, 1] <  pixel_view.shape[0],
                    coords[:, 1] >= 0,
                )).cpu()

                coords = coords[in_view, :]
                if not isinstance(color, tuple):
                    color = color[in_view, :]

                pixel_view[(*coords[..., [1, 0]].int().T.cpu(),)] = color

            self.export_medial_surface_mesh = False
