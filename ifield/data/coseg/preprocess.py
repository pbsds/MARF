#!/usr/bin/env python3
import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from . import config, read
from ...utils.helpers import make_relative
from pathlib import Path
from textwrap import dedent
import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=dedent("""
    Preprocess the COSEG dataset. Depends on `download-coseg --shapes ...` having been run.
    """), formatter_class=argparse.RawTextHelpFormatter)

    arg = parser.add_argument # brevity

    arg("items", nargs="*", default=[],
        help="Which object-set[/model-id] to process, defaults to all downloaded. Format: OBJECT-SET[/MODEL-ID]")
    arg("--dir", default=str(config.DATA_PATH),
        help=f"The target directory. Default is {make_relative(config.DATA_PATH, Path.cwd()).__str__()!r}")
    arg("--force", action="store_true",
        help="Overwrite existing files")
    arg("--list-models", action="store_true",
        help="List the downloaded models available for preprocessing")
    arg("--list-object-sets", action="store_true",
        help="List the downloaded object-sets available for preprocessing")
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

    arg3 = parser.add_argument_group("modifiers").add_argument # brevity
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

    return parser

# entrypoint
def cli(parser=make_parser()):
    args = parser.parse_args()
    if not any(getattr(args, k) for k in dir(args) if k.startswith("precompute_")) and not (args.list_models or args.list_object_sets or args.list_pages):
        parser.error("no preprocessing target selected") # exits

    config.DATA_PATH = Path(args.dir)

    object_sets = [i for i in args.items if "/" not in i]
    models      = [i.split("/") for i in args.items if "/" in i]

    # convert/expand synsets to models
    # they are mutually exclusive
    if object_sets: assert not models
    if models:  assert not object_sets
    if not models:
        models = read.list_model_ids(tuple(object_sets) or None)

    if args.list_models:
        try:
            print(*(f"{object_set_id}/{model_id}" for object_set_id, model_id in models), sep="\n")
        except BrokenPipeError:
            pass
        parser.exit()

    if args.list_object_sets:
        try:
            print(*sorted(set(object_set_id for object_set_id, model_id in models)), sep="\n")
        except BrokenPipeError:
            pass
        parser.exit()

    if args.list_pages is not None:
        try:
            print(*(
                f"--page {i} {args.list_pages} {object_set_id}/{model_id}"
                for object_set_id, model_id in models
                for i in range(args.list_pages)
            ), sep="\n")
        except BrokenPipeError:
            pass
        parser.exit()

    if args.precompute_mesh_sv_scan_clouds:
        read.precompute_mesh_scan_point_clouds(
            models,
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
            models,
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
            models,
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
