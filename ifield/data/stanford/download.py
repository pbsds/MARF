#!/usr/bin/env python3
from . import config
from ...utils.helpers import make_relative
from ..common import download
from pathlib import Path
from textwrap import dedent
from typing import Iterable
import argparse
import io
import tarfile


def is_downloaded(*a, **kw):
    return download.is_downloaded(*a, dbfiles=config.IS_DOWNLOADED_DB, **kw)

def download_and_extract(target_dir: Path, url_list: Iterable[str], *, force=False, silent=False) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)

    ret = False
    for url in url_list:
        if not force:
            if is_downloaded(target_dir, url): continue
        if not download.check_url(url):
            print("ERROR:", url)
            continue
        ret = True

        data = download.download_data(url, silent=silent, label=str(Path(url).name))

        print("extracting...")
        if url.endswith(".ply.gz"):
            fname = target_dir / "meshes" / url.split("/")[-1].lower()
            fname.parent.mkdir(parents=True, exist_ok=True)
            with fname.open("wb") as f:
                f.write(data)
        elif url.endswith(".tar.gz"):
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                for member in tar.getmembers():
                    if not member.isfile(): continue
                    if member.name.startswith("/"): continue
                    if member.name.startswith("."): continue
                    if Path(member.name).name.startswith("."): continue
                    tar.extract(member, target_dir / "meshes")
            del tar
        else:
            raise NotImplementedError(f"Extraction for {str(Path(url).name)} unknown")

        is_downloaded(target_dir, url, add=True)
        del data

    return ret

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=dedent("""
    Download The Stanford 3D Scanning Repository models.
    More info: http://graphics.stanford.edu/data/3Dscanrep/

    Example:

    download-stanford bunny
    """), formatter_class=argparse.RawTextHelpFormatter)

    arg = parser.add_argument

    arg("objects", nargs="*", default=[],
        help="Which objects to download, defaults to none.")
    arg("--all", action="store_true",
        help="Download all objects")
    arg("--dir", default=str(config.DATA_PATH),
        help=f"The target directory. Default is {make_relative(config.DATA_PATH, Path.cwd()).__str__()!r}")

    arg("--list", action="store_true",
        help="Lists all the objects")
    arg("--list-urls", action="store_true",
        help="Lists the urls to download")
    arg("--list-sizes", action="store_true",
        help="Lists the download size of each model")
    arg("--silent", action="store_true",
        help="")
    arg("--force", action="store_true",
        help="Download again even if already downloaded")

    return parser

# entrypoint
def cli(parser=make_parser()):
    args = parser.parse_args()

    obj_names = sorted(set(args.objects))
    if args.all:
        assert not obj_names
        obj_names = sorted(config.MODELS.keys())
    if not obj_names and args.list_urls: config.MODELS.keys()

    if args.list:
        print(*config.MODELS.keys(), sep="\n")
        exit()

    if args.list_sizes:
        print(*(f"{obj_name:<15}{config.MODELS[obj_name].download_size_str}" for obj_name in (obj_names or config.MODELS.keys())), sep="\n")
        exit()

    try:
        url_list = [config.MODELS[obj_name].url for obj_name in obj_names]
    except KeyError:
        print("Error: unrecognized object name:", *set(obj_names).difference(config.MODELS.keys()), sep="\n")
        exit(1)

    if not url_list:
        print("Error: No object set was selected for download!")
        exit(1)

    if args.list_urls:
        print(*url_list, sep="\n")
        exit()


    print("Download start")
    any_downloaded = download_and_extract(
        target_dir = Path(args.dir),
        url_list   = url_list,
        force      = args.force,
        silent     = args.silent,
    )
    if not any_downloaded:
        print("Everything has already been downloaded, skipping.")

if __name__ == "__main__":
    cli()
