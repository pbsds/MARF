#!/usr/bin/env python3
from . import config
from ...utils.helpers import make_relative
from ..common import download
from pathlib import Path
from textwrap import dedent
import argparse
import io
import zipfile



def is_downloaded(*a, **kw):
    return download.is_downloaded(*a, dbfiles=config.IS_DOWNLOADED_DB, **kw)

def download_and_extract(target_dir: Path, url_dict: dict[str, str], *, force=False, silent=False) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)

    ret = False
    for url, fname in url_dict.items():
        if not force:
            if is_downloaded(target_dir, url): continue
        if not download.check_url(url):
            print("ERROR:", url)
            continue
        ret = True

        if force or not (target_dir / "archives" / fname).is_file():

            data = download.download_data(url, silent=silent, label=fname)
            assert url.endswith(".zip")

            print("writing...")

            (target_dir / "archives").mkdir(parents=True, exist_ok=True)
            with (target_dir / "archives" / fname).open("wb") as f:
                f.write(data)
            del data

        print(f"extracting {fname}...")

        with zipfile.ZipFile(target_dir / "archives" / fname, 'r') as f:
            f.extractall(target_dir / Path(fname).stem.removesuffix("-shapes").removesuffix("-gt"))

        is_downloaded(target_dir, url, add=True)

    return ret

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=dedent("""
    Download The COSEG Shape Dataset.
    More info: http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm

    Example:

    download-coseg --shapes chairs
    """), formatter_class=argparse.RawTextHelpFormatter)

    arg = parser.add_argument

    arg("sets", nargs="*", default=[],
        help="Which set to download, defaults to none.")
    arg("--all", action="store_true",
        help="Download all sets")
    arg("--dir", default=str(config.DATA_PATH),
        help=f"The target directory. Default is {make_relative(config.DATA_PATH, Path.cwd()).__str__()!r}")

    arg("--shapes", action="store_true",
        help="Download the 3d shapes for each chosen set")
    arg("--gts", action="store_true",
        help="Download the ground-truth segmentation data for each chosen set")

    arg("--list", action="store_true",
        help="Lists all the sets")
    arg("--list-urls", action="store_true",
        help="Lists the urls to download")
    arg("--list-sizes", action="store_true",
        help="Lists the download size of each set")
    arg("--silent", action="store_true",
        help="")
    arg("--force", action="store_true",
        help="Download again even if already downloaded")

    return parser

# entrypoint
def cli(parser=make_parser()):
    args = parser.parse_args()

    assert set(config.SHAPES.keys()) == set(config.GROUND_TRUTHS.keys())

    set_names = sorted(set(args.sets))
    if args.all:
        assert not set_names, "--all is mutually exclusive from manually selected sets"
        set_names = sorted(config.SHAPES.keys())

    if args.list:
        print(*config.SHAPES.keys(), sep="\n")
        exit()

    if args.list_sizes:
        print(*(f"{set_name:<15}{config.SHAPES[set_name].download_size_str}" for set_name in (set_names or config.SHAPES.keys())), sep="\n")
        exit()

    try:
        url_dict \
            = {config.SHAPES[set_name].url        : config.SHAPES[set_name].fname        for set_name in set_names if args.shapes} \
            | {config.GROUND_TRUTHS[set_name].url : config.GROUND_TRUTHS[set_name].fname for set_name in set_names if args.gts}
    except KeyError:
        print("Error: unrecognized object name:", *set(set_names).difference(config.SHAPES.keys()), sep="\n")
        exit(1)

    if not url_dict:
        if set_names and not (args.shapes or args.gts):
            print("Error: Provide at least one of --shapes of --gts")
        else:
            print("Error: No object set was selected for download!")
        exit(1)

    if args.list_urls:
        print(*url_dict.keys(), sep="\n")
        exit()

    print("Download start")
    any_downloaded = download_and_extract(
        target_dir = Path(args.dir),
        url_dict   = url_dict,
        force      = args.force,
        silent     = args.silent,
    )
    if not any_downloaded:
        print("Everything has already been downloaded, skipping.")

if __name__ == "__main__":
    cli()
