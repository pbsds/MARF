#!/usr/bin/env python3
from .data.common.scan import SingleViewScan, SingleViewUVScan
from datetime import datetime
import re
import click
import gzip
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
import trimesh.transformations as T

__doc__ = """
Here are a bunch of helper scripts exposed as cli command by poetry
"""


# these entrypoints are exposed by poetry as shell commands

@click.command()
@click.argument("h5file")
@click.argument("key", default="")
def show_h5_items(h5file: str, key: str):
    "Show contents of HDF5 dataset"
    f = h5.File(h5file, "r")
    if not key:
        mlen = max(map(len, f.keys()))
        for i in sorted(f.keys()):
            print(i.ljust(mlen), ":",
                str (f[i].dtype).ljust(10),
                repr(f[i].shape).ljust(16),
                "mean:", f[i][:].mean()
            )
    else:
        if not f[key].shape:
            print(f[key].value)
        else:
            print(f[key][:])


@click.command()
@click.argument("h5file")
@click.argument("key", default="")
def show_h5_img(h5file: str, key: str):
    "Show a 2D HDF5 dataset as an image"
    f = h5.File(h5file, "r")
    if not key:
        mlen = max(map(len, f.keys()))
        for i in sorted(f.keys()):
            print(i.ljust(mlen), ":", str(f[i].dtype).ljust(10), f[i].shape)
    else:
        plt.imshow(f[key])
        plt.show()


@click.command()
@click.argument("h5file")
@click.option("--force-distances",  is_flag=True, help="Always show miss distances.")
@click.option("--uv",               is_flag=True, help="Load as UV scan cloud and convert it.")
@click.option("--show-unit-sphere", is_flag=True, help="Show the unit sphere.")
@click.option("--missing",          is_flag=True, help="Show miss points that are not hits nor misses as purple.")
def show_h5_scan_cloud(
        h5file           : str,
        force_distances  : bool = False,
        uv               : bool = False,
        missing          : bool = False,
        show_unit_sphere        = False,
        ):
    "Show a SingleViewScan HDF5 dataset"
    print("Reading data...")
    t = datetime.now()
    if uv:
        scan = SingleViewUVScan.from_h5_file(h5file)
        if missing and scan.any_missing:
            if not scan.has_missing:
                scan.fill_missing_points()
            points_missing = scan.points[scan.missing]
        else:
            missing = False
        if not scan.is_single_view:
            scan.cam_pos = None
        scan = scan.to_scan()
    else:
        scan = SingleViewScan.from_h5_file(h5file)
        if missing:
            uvscan = scan.to_uv_scan()
            if scan.any_missing:
                uvscan.fill_missing_points()
                points_missing = uvscan.points[uvscan.missing]
            else:
                missing = False
    print("loadtime: ", datetime.now() - t)

    if force_distances and not scan.has_miss_distances:
        print("Computing miss distances...")
        scan.compute_miss_distances()
        use_miss_distances = force_distances
    print("Constructing scene...")
    if not scan.has_colors:
        scan.colors_hit  = np.zeros_like(scan.points_hit)
        scan.colors_miss = np.zeros_like(scan.points_miss)
        scan.colors_hit [:] = ( 31/255, 119/255, 180/255)
        scan.colors_miss[:] = (243/255, 156/255,  18/255)
        use_miss_distances = True
    if scan.has_miss_distances and use_miss_distances:
        sdm = scan.distances_miss / scan.distances_miss.max()
        sdm = sdm[..., None]
        scan.colors_miss \
            = np.array([0.8, 0, 0])[None, :] * sdm \
            + np.array([0, 1, 0.2])[None, :] * (1-sdm)


    scene = pyrender.Scene()

    scene.add(pyrender.Mesh.from_points(scan.points_hit,  colors=scan.colors_hit, normals=scan.normals_hit))
    scene.add(pyrender.Mesh.from_points(scan.points_miss, colors=scan.colors_miss))

    if missing:
        scene.add(pyrender.Mesh.from_points(points_missing, colors=(np.array((0xff, 0x00, 0xff))/255)[None, :].repeat(points_missing.shape[0], axis=0)))

    # camera:
    if not scan.points_cam is None:
        camera_mesh = trimesh.creation.uv_sphere(radius=scan.points_hit_std.max()*0.2)
        camera_mesh.visual.vertex_colors = [0.0, 0.8, 0.0]
        tfs = np.tile(np.eye(4), (len(scan.points_cam), 1, 1))
        tfs[:,:3,3] = scan.points_cam
        scene.add(pyrender.Mesh.from_trimesh(camera_mesh, poses=tfs))

    # UV sphere:
    if show_unit_sphere:
        unit_sphere_mesh = trimesh.creation.uv_sphere(radius=1)
        unit_sphere_mesh.invert()
        unit_sphere_mesh.visual.vertex_colors = [0.8, 0.8, 0.0]
        scene.add(pyrender.Mesh.from_trimesh(unit_sphere_mesh, poses=np.eye(4)[None, ...]))

    print("Launch!")
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


@click.command()
@click.argument("meshfile")
@click.option('--aabb', is_flag=True)
@click.option('--z-skyward', is_flag=True)
def show_model(
        meshfile  : str,
        aabb      : bool,
        z_skyward : bool,
        ):
    "Show a 3D model with pyrender, supports .gz suffix"
    if meshfile.endswith(".gz"):
        with gzip.open(meshfile, "r") as f:
            mesh = trimesh.load(f, file_type=meshfile.split(".", 1)[1].removesuffix(".gz"))
    else:
        mesh = trimesh.load(meshfile)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if aabb:
        from .data.common.mesh import rotate_to_closest_axis_aligned_bounds
        mesh.apply_transform(rotate_to_closest_axis_aligned_bounds(mesh, fail_ok=True))

    if z_skyward:
        mesh.apply_transform(T.rotation_matrix(np.pi/2, (1, 0, 0)))

    print(
        *(i.strip() for i in pyrender.Viewer.__doc__.splitlines() if re.search(r"- ``[a-z0-9]``: ", i)),
        sep="\n"
    )

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    pyrender.Viewer(scene, use_raymond_lighting=True)
