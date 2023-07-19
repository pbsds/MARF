#!/usr/bin/env python3
import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from . import config, read
from ...utils.helpers import make_relative
from pathlib import Path
from textwrap import dedent
import argparse



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=dedent("""
    Preprocess the Stanford models. Depends on `download-stanford` having been run.
    """), formatter_class=argparse.RawTextHelpFormatter)

    arg = parser.add_argument # brevity

    arg("objects", nargs="*", default=[],
        help="Which objects to process, defaults to all downloaded")
    arg("--dir", default=str(config.DATA_PATH),
        help=f"The target directory. Default is {make_relative(config.DATA_PATH, Path.cwd()).__str__()!r}")
    arg("--force", action="store_true",
        help="Overwrite existing files")
    arg("--list", action="store_true",
        help="List the downloaded models available for preprocessing")
    arg("--list-pages", type=int, default=None,
        help="List the downloaded models available for preprocessing, paginated into N pages.")
    arg("--page", nargs=2, type=int, default=[0, 1],
        help="Subset of parts to compute. Use to parallelize. (page, total), page is 0 indexed")

    arg2 = parser.add_argument_group("preprocessing targets").add_argument # brevity
    arg2("--precompute-mesh-sv-scan-clouds", action="store_true",
        help="Compute single-view hit+miss point clouds from 100 synthetic scans.")
    arg2("--precompute-mesh-sv-scan-uvs", action="store_true",
        help="Compute single-view hit+miss UV clouds from 100 synthetic scans.")
    arg2("--precompute-mesh-sphere-scan", action="store_true",
        help="Compute a sphere-view hit+miss cloud cast from n to n unit sphere points.")

    arg3 = parser.add_argument_group("ray-scan modifiers").add_argument # brevity
    arg3("--n-sphere-points", type=int, default=4000,
        help="The number of unit-sphere points to sample rays from. Final result: n*(n-1).")
    arg3("--compute-miss-distances", action="store_true",
        help="Compute the distance to the nearest hit for each miss in the hit+miss clouds.")
    arg3("--fill-missing-uv-points", action="store_true",
        help="TODO")
    arg3("--no-filter-backhits", action="store_true",
        help="Do not filter scan hits on backside of mesh faces.")
    arg3("--no-unit-sphere", action="store_true",
        help="Do not center the objects to the unit sphere.")
    arg3("--convert-ok", action="store_true",
        help="Allow reusing point clouds for uv clouds and vice versa. (does not account for other hparams)")
    arg3("--debug", action="store_true",
        help="Abort on failiure.")

    arg5 = parser.add_argument_group("Shared modifiers").add_argument # brevity
    arg5("--scan-resolution", type=int, default=400,
        help="The resolution of the depth map rendered to sample points. Becomes x*x")

    return parser

# entrypoint
def cli(parser: argparse.ArgumentParser = make_parser()):
    args = parser.parse_args()
    if not any(getattr(args, k) for k in dir(args) if k.startswith("precompute_")) and not (args.list or args.list_pages):
        parser.error("no preprocessing target selected") # exits

    config.DATA_PATH = Path(args.dir)
    obj_names = args.objects or read.list_object_names()

    if args.list:
        print(*obj_names, sep="\n")
        parser.exit()

    if args.list_pages is not None:
        print(*(
            f"--page {i} {args.list_pages} {obj_name}"
            for obj_name in obj_names
            for i in range(args.list_pages)
        ), sep="\n")
        parser.exit()

    if args.precompute_mesh_sv_scan_clouds:
        read.precompute_mesh_scan_point_clouds(
            obj_names,
            compute_miss_distances = args.compute_miss_distances,
            no_filter_backhits     = args.no_filter_backhits,
            no_unit_sphere         = args.no_unit_sphere,
            convert_ok             = args.convert_ok,
            page                   = args.page,
            force                  = args.force,
            debug                  = args.debug,
        )
    if args.precompute_mesh_sv_scan_uvs:
        read.precompute_mesh_scan_uvs(
            obj_names,
            compute_miss_distances = args.compute_miss_distances,
            fill_missing_points    = args.fill_missing_uv_points,
            no_filter_backhits     = args.no_filter_backhits,
            no_unit_sphere         = args.no_unit_sphere,
            convert_ok             = args.convert_ok,
            page                   = args.page,
            force                  = args.force,
            debug                  = args.debug,
        )
    if args.precompute_mesh_sphere_scan:
        read.precompute_mesh_sphere_scan(
            obj_names,
            sphere_points          = args.n_sphere_points,
            compute_miss_distances = args.compute_miss_distances,
            no_filter_backhits     = args.no_filter_backhits,
            no_unit_sphere         = args.no_unit_sphere,
            page                   = args.page,
            force                  = args.force,
            debug                  = args.debug,
        )

if __name__ == "__main__":
    cli()
