from ..utils import geometry
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from pytorch3d.transforms import euler_angles_to_matrix
from tqdm import tqdm
from typing import Sequence, Callable, TypedDict
import imageio
import shlex
import json
import numpy as np
import os
import time
import torch
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

IVec2 = tuple[int, int]
IVec3 = tuple[int, int, int]
Vec2  = tuple[float|int, float|int]
Vec3  = tuple[float|int, float|int, float|int]

class CamState(TypedDict, total=False):
    distance : float
    pos_x    : float
    pos_y    : float
    pos_z    : float
    rot_x    : float
    rot_y    : float
    fov_y    : float



class InteractiveViewer(ABC):
    constants = pygame.constants # saves an import

    # realtime
    t  : float # time since start
    td : float # time delta since last frame

    # offline
    is_headless : bool
    fps         : int
    frame_idx   : int

    fill_color = (255, 255, 255)

    def __init__(self, name: str, res: IVec2 = (640, 480), scale: int= 1, screenshot_dir: Path = "."):
        self.name           = name
        self.res            = res
        self.scale          = scale
        self.screenshot_dir = Path(screenshot_dir)

        self.is_headless = False

        self.cam_distance = 2.0
        self.cam_pos_x    = 0.0 # look-at and rotation pivot
        self.cam_pos_y    = 0.0 # look-at and rotation pivot
        self.cam_pos_z    = 0.0 # look-at and rotation pivot
        self.cam_rot_x    = 0.5 * torch.pi # radians
        self.cam_rot_y    = -0.5 * torch.pi # radians
        self.cam_fov_y    = 60.0 / 180.0 * 3.1415 # radians
        self.keep_rotating = False
        self.initial_camera_state = self.cam_state
        self.fps_cap = None

    @property
    def cam_state(self) -> CamState:
        return dict(
            distance = self.cam_distance,
            pos_x    = self.cam_pos_x,
            pos_y    = self.cam_pos_y,
            pos_z    = self.cam_pos_z,
            rot_x    = self.cam_rot_x,
            rot_y    = self.cam_rot_y,
            fov_y    = self.cam_fov_y,
        )

    @cam_state.setter
    def cam_state(self, new_state: CamState):
        self.cam_distance = new_state.get("distance", self.cam_distance)
        self.cam_pos_x    = new_state.get("pos_x",    self.cam_pos_x)
        self.cam_pos_y    = new_state.get("pos_y",    self.cam_pos_y)
        self.cam_pos_z    = new_state.get("pos_z",    self.cam_pos_z)
        self.cam_rot_x    = new_state.get("rot_x",    self.cam_rot_x)
        self.cam_rot_y    = new_state.get("rot_y",    self.cam_rot_y)
        self.cam_fov_y    = new_state.get("fov_y",    self.cam_fov_y)

    @property
    def scaled_res(self) -> IVec2:
        return (
            self.res[0] * self.scale,
            self.res[1] * self.scale,
        )

    def setup(self):
        pass

    def teardown(self):
        pass

    @abstractmethod
    def render_frame(self, pixel_view: np.ndarray): # (W, H, 3) dtype=uint8
        ...

    def handle_key_up(self, key: int, keys_pressed: Sequence[bool]):
        pass

    def handle_key_down(self, key: int, keys_pressed: Sequence[bool]):
        mod  = keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]
        mod2 = keys_pressed[pygame.K_LCTRL]  or keys_pressed[pygame.K_RCTRL]
        if key == pygame.K_r:
            self.keep_rotating = True
            self.cam_rot_x += self.td
        if key == pygame.K_MINUS:
            self.scale += 1
            if __debug__: print()
            print(f"== Scale = {self.scale} ==")
        if key == pygame.K_PLUS and self.scale > 1:
            self.scale -= 1
            if __debug__: print()
            print(f"== Scale = {self.scale} ==")
        if key == pygame.K_RETURN:
            self.cam_state = self.initial_camera_state
        if key == pygame.K_h:
            if mod2:
                print(shlex.quote(json.dumps(self.cam_state)))
            elif mod:
                with (self.screenshot_dir / "camera.json").open("w") as f:
                    json.dump(self.cam_state, f)
                    print("Wrote", self.screenshot_dir / "camera.json")
            else:
                with (self.screenshot_dir / "camera.json").open("r") as f:
                    self.cam_state = json.load(f)
                    print("Read", self.screenshot_dir / "camera.json")

    def handle_keys_pressed(self, pressed: Sequence[bool]) -> float:
        mod1 = pressed[pygame.K_LCTRL]  or pressed[pygame.K_RCTRL]
        mod2 = pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT]
        mod3 = pressed[pygame.K_LALT]   or pressed[pygame.K_RALT]
        td = self.td * (0.5 if mod2 else (6 if mod1 else 2))

        if pressed[pygame.K_UP]:    self.cam_rot_y += td
        if pressed[pygame.K_DOWN]:  self.cam_rot_y -= td
        if pressed[pygame.K_LEFT]:  self.cam_rot_x += td
        if pressed[pygame.K_RIGHT]: self.cam_rot_x -= td
        if pressed[pygame.K_PAGEUP]   and mod3: self.cam_distance -= td
        if pressed[pygame.K_PAGEDOWN] and mod3: self.cam_distance += td

        if any(pressed[i] for i in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]):
            self.keep_rotating = False
        if self.keep_rotating: self.cam_rot_x += self.td * 0.25

        if pressed[pygame.K_w]: self.cam_pos_x -= td * np.cos(-self.cam_rot_x)
        if pressed[pygame.K_w]: self.cam_pos_y += td * np.sin(-self.cam_rot_x)
        if pressed[pygame.K_s]: self.cam_pos_x += td * np.cos(-self.cam_rot_x)
        if pressed[pygame.K_s]: self.cam_pos_y -= td * np.sin(-self.cam_rot_x)
        if pressed[pygame.K_a]: self.cam_pos_x += td * np.sin(self.cam_rot_x)
        if pressed[pygame.K_a]: self.cam_pos_y -= td * np.cos(self.cam_rot_x)
        if pressed[pygame.K_d]: self.cam_pos_x -= td * np.sin(self.cam_rot_x)
        if pressed[pygame.K_d]: self.cam_pos_y += td * np.cos(self.cam_rot_x)
        if pressed[pygame.K_PAGEUP]   and not mod3: self.cam_pos_z -= td
        if pressed[pygame.K_PAGEDOWN] and not mod3: self.cam_pos_z += td

        return td

    def handle_mouse_button_up(self, pos: IVec2, button: int, keys_pressed: Sequence[bool]):
        pass

    def handle_mouse_button_down(self, pos: IVec2, button: int, keys_pressed: Sequence[bool]):
        pass

    def handle_mouse_motion(self, pos: IVec2, rel: IVec2, buttons: Sequence[bool], keys_pressed: Sequence[bool]):
        pass

    def handle_mousewheel(self, flipped: bool, x: int, y: int, keys_pressed: Sequence[bool]):
        if keys_pressed[pygame.K_LALT] or keys_pressed[pygame.K_RALT]:
            self.cam_fov_y -= y * 0.015
        else:
            self.cam_distance -= y * 0.2

    _current_caption = None
    def set_caption(self, title: str, *a, **kw):
        if self._current_caption != title and not self.is_headless:
            print(f"set_caption: {title!r}")
            self._current_caption = title
            return pygame.display.set_caption(title, *a, **kw)

    @property
    def mouse_position(self) -> IVec2:
        mx, my = pygame.mouse.get_pos() if not self.is_headless else (0, 0)
        return (
            mx // self.scale,
            my // self.scale,
        )

    @property
    def uvs(self) -> torch.Tensor: # (w, h, 2) dtype=float32
        res = tuple(self.res)
        if not getattr(self, "_uvs_res", None) == res:
            U, V = torch.meshgrid(
                torch.arange(self.res[1]).to(torch.float32),
                torch.arange(self.res[0]).to(torch.float32),
                indexing="xy",
            )
            self._uvs_res, self._uvs = res, torch.stack((U, V), dim=-1)
        return self._uvs

    @property
    def cam2world(self) -> torch.Tensor: # (4, 4) dtype=float32
        if getattr(self, "_cam2world_cam_rot_y",    None) is not self.cam_rot_y \
        or getattr(self, "_cam2world_cam_rot_x",    None) is not self.cam_rot_x \
        or getattr(self, "_cam2world_cam_pos_x",    None) is not self.cam_pos_x \
        or getattr(self, "_cam2world_cam_pos_y",    None) is not self.cam_pos_y \
        or getattr(self, "_cam2world_cam_pos_z",    None) is not self.cam_pos_z \
        or getattr(self, "_cam2world_cam_distance", None) is not self.cam_distance:
            self._cam2world_cam_rot_y    = self.cam_rot_y
            self._cam2world_cam_rot_x    = self.cam_rot_x
            self._cam2world_cam_pos_x    = self.cam_pos_x
            self._cam2world_cam_pos_y    = self.cam_pos_y
            self._cam2world_cam_pos_z    = self.cam_pos_z
            self._cam2world_cam_distance = self.cam_distance

            a = torch.eye(4)
            a[2, 3] = self.cam_distance
            b = torch.eye(4)
            b[:3, :3] = euler_angles_to_matrix(torch.tensor((self.cam_rot_x, self.cam_rot_y, 0)), "ZYX")
            b[0:3, 3] -= torch.tensor(( self.cam_pos_x, self.cam_pos_y, self.cam_pos_z, ))
            self._cam2world = b @ a

            self._cam2world_inv = None
        return self._cam2world

    @property
    def cam2world_inv(self) -> torch.Tensor: # (4, 4) dtype=float32
        if getattr(self, "_cam2world_inv", None) is None:
            self._cam2world_inv = torch.linalg.inv(self._cam2world)
        return self._cam2world_inv

    @property
    def intrinsics(self) -> torch.Tensor: # (3, 3) dtype=float32
        if getattr(self, "_intrinsics_res",       None) is not self.res \
        or getattr(self, "_intrinsics_cam_fov_y", None) is not self.cam_fov_y:
            self._intrinsics_res       = res       = self.res
            self._intrinsics_cam_fov_y = cam_fov_y = self.cam_fov_y

            self._intrinsics = torch.eye(3)
            p = torch.sin(torch.tensor(cam_fov_y / 2))
            s = (res[1] / 2)
            self._intrinsics[0, 0] = s/p              # fx - focal length x
            self._intrinsics[1, 1] = s/p              # fy - focal length y
            self._intrinsics[0, 2] = (res[1] - 1) / 2 # cx - optical center x
            self._intrinsics[1, 2] = (res[0] - 1) / 2 # cy - optical center y
        return self._intrinsics

    @property
    def raydirs_and_cam(self) -> tuple[torch.Tensor, torch.Tensor]: # (w, h, 3) and (3) dtype=float32
        if getattr(self, "_raydirs_and_cam_cam2world",  None) is not self.cam2world \
        or getattr(self, "_raydirs_and_cam_intrinsics", None) is not self.intrinsics \
        or getattr(self, "_raydirs_and_cam_uvs",        None) is not self.uvs:
            self._raydirs_and_cam_cam2world  = cam2world  = self.cam2world
            self._raydirs_and_cam_intrinsics = intrinsics = self.intrinsics
            self._raydirs_and_cam_uvs        = uvs        = self.uvs

            #cam_pos   = (cam2world @ torch.tensor([0, 0, 0, 1], dtype=torch.float32))[:3]
            cam_pos   = cam2world[:3, -1]

            dirs = -geometry.get_ray_directions(uvs, cam2world[None, ...], intrinsics[None, ...]).squeeze(-1)

            self._raydirs_and_cam = (dirs, cam_pos)
        return (
            self._raydirs_and_cam[0],
            self._raydirs_and_cam[1],
        )

    def run(self):
        self.is_headless = False
        pygame.display.init() # we do not use the mixer, which often hangs on quit
        try:
            window = pygame.display.set_mode(self.scaled_res, flags=pygame.RESIZABLE)
            buffer = pygame.surface.Surface(self.res)

            window.fill(self.fill_color)
            buffer.fill(self.fill_color)
            pygame.display.flip()

            pixel_view = pygame.surfarray.pixels3d(buffer) # (W, H, 3)

            current_scale = self.scale
            def remake_window_buffer(window_size: IVec2):
                nonlocal buffer, pixel_view, current_scale
                self.res = (
                    window_size[0] // self.scale,
                    window_size[1] // self.scale,
                )
                buffer = pygame.surface.Surface(self.res)
                pixel_view = pygame.surfarray.pixels3d(buffer)
                current_scale = self.scale

            print()

            self.setup()

            is_running = True
            clock = pygame.time.Clock()
            epoch = t_prev = time.time()
            self.frame_idx = -1
            while is_running:
                self.frame_idx += 1
                if not self.fps_cap is None: clock.tick(self.fps_cap)
                t = time.time()
                self.td = t - t_prev
                t_prev = t
                self.t = t - epoch
                print("\rFPS:", 1/self.td, " "*10, end="")

                self.render_frame(pixel_view)

                pygame.transform.scale(buffer, window.get_size(), window)
                pygame.display.flip()

                keys_pressed = pygame.key.get_pressed()
                self.handle_keys_pressed(keys_pressed)

                for event in pygame.event.get():
                    if event.type == pygame.VIDEORESIZE:
                        print()
                        print("== resize window ==")
                        remake_window_buffer(event.size)
                    elif event.type == pygame.QUIT:
                        is_running = False
                    elif event.type == pygame.KEYUP:
                        self.handle_key_up(event.key, keys_pressed)
                    elif event.type == pygame.KEYDOWN:
                        self.handle_key_down(event.key, keys_pressed)
                        if event.key == pygame.K_q:
                            is_running = False
                        elif event.key == pygame.K_y:
                            fname = self.mk_dump_fname("png")
                            fname.parent.mkdir(parents=True, exist_ok=True)
                            pygame.image.save(buffer.copy(), fname)
                            print()
                            print("Saved", fname)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.handle_mouse_button_up(event.pos, event.button, keys_pressed)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_mouse_button_down(event.pos, event.button, keys_pressed)
                    elif event.type == pygame.MOUSEMOTION:
                        self.handle_mouse_motion(event.pos, event.rel, event.buttons, keys_pressed)
                    elif event.type == pygame.MOUSEWHEEL:
                        self.handle_mousewheel(event.flipped, event.x, event.y, keys_pressed)

                if current_scale != self.scale:
                    remake_window_buffer(window.get_size())

        finally:
            self.teardown()
            print()
            pygame.quit()

    def render_headless(self, output_path: str, *, n_frames: int, fps: int, state_callback: Callable[["InteractiveViewer", int], None] | None, resolution=None, bitrate=None, **kw):
        self.is_headless = True
        self.fps         = fps

        buffer     = pygame.surface.Surface(self.res if resolution is None else resolution)
        pixel_view = pygame.surfarray.pixels3d(buffer) # (W, H, 3)

        def do():
            try:
                self.setup()
                for frame in tqdm(range(n_frames), **kw, disable=n_frames==1):
                    self.frame_idx = frame
                    if state_callback is not None:
                        state_callback(self, frame)

                    self.render_frame(pixel_view)

                    yield pixel_view.copy().swapaxes(0,1)
            finally:
                self.teardown()

        output_path = Path(output_path)
        if output_path.suffix == ".png":
            if n_frames > 1 and "%" not in output_path.name: raise ValueError
            output_path.parent.mkdir(parents=True, exist_ok=True)
            for i, framebuffer in enumerate(do()):
                with imageio.get_writer(output_path.parent / output_path.name.replace("%", f"{i:04}")) as writer:
                    writer.append_data(framebuffer)
        else: # ffmpeg - https://imageio.readthedocs.io/en/v2.9.0/format_ffmpeg.html#ffmpeg
            with imageio.get_writer(output_path, fps=fps, bitrate=bitrate) as writer:
                for framebuffer in do():
                    writer.append_data(framebuffer)

    def load_sphere_map(self, fname):
        self._sphere_surf = pygame.image.load(fname)
        self._sphere_map = pygame.surfarray.pixels3d(self._sphere_surf)

    def lookup_sphere_map_dirs(self, dirs, origins):
        near, far = geometry.ray_sphere_intersect(
            torch.tensor(origins),
            torch.tensor(dirs),
            sphere_radii = torch.tensor(origins).norm(dim=-1) * 2,
        )
        hits = far.detach()

        x = hits[..., 0]
        y = hits[..., 1]
        z = hits[..., 2]
        theta = (z / hits.norm(dim=-1)).acos()
        phi = (y/x).atan()
        phi[(x<0) & (y>=0)] += 3.14
        phi[(x<0) & (y< 0)] -= 3.14

        w, h = self._sphere_map.shape[:2]

        return self._sphere_map[
            ((phi   / (2*torch.pi) * w).int() % w).cpu(),
            ((theta / (1*torch.pi) * h).int() % h).cpu(),
        ]

    def blit_sphere_map_mask(self, pixel_view, mask=None):
        dirs, origin = self.raydirs_and_cam
        if mask is None: mask = (slice(None), slice(None))
        pixel_view[mask] \
            = self.lookup_sphere_map_dirs(dirs, origin[None, None, :])

    def mk_dump_fname(self, suffix: str, uid=None) -> Path:
        name = self.name.split("-")[-1] if len(self.name) > 160 else self.name
        if uid is not None: name = f"{name}-{uid}"
        return self.screenshot_dir / f"pygame-viewer-{datetime.now():%Y%m%d-%H%M%S}-{name}.{suffix}"
