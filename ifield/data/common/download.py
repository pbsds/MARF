from ...utils.helpers import make_relative
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional
import io
import os
import json
import requests

PathLike = Union[os.PathLike, str]

__doc__ = """
Here are some helper functions for processing data.
"""

def check_url(url): # HTTP HEAD
    return requests.head(url).ok

def download_stream(
        url         : str,
        file_object,
        block_size  : int           = 1024,
        silent      : bool          = False,
        label       : Optional[str] = None,
        ):
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("content-length", 0))
    if not silent:
        progress_bar = tqdm(total=total_size , unit="iB", unit_scale=True, desc=label)

    for chunk in resp.iter_content(block_size):
        if not silent:
            progress_bar.update(len(chunk))
        file_object.write(chunk)

    if not silent:
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")

def download_data(
        url        : str,
        block_size : int           = 1024,
        silent     : bool          = False,
        label      : Optional[str] = None,
        ) -> bytearray:
    f = io.BytesIO()
    download_stream(url, f, block_size=block_size, silent=silent, label=label)
    f.seek(0)
    return bytearray(f.read())

def download_file(
        url        : str,
        fname      : Union[Path, str],
        block_size : int              = 1024,
        silent                        = False,
        ):
    if not isinstance(fname, Path):
        fname = Path(fname)
    with fname.open("wb") as f:
        download_stream(url, f, block_size=block_size, silent=silent, label=make_relative(fname, Path.cwd()).name)

def is_downloaded(
        target_dir : PathLike,
        url        : str,
        *,
        add        : bool = False,
        dbfiles    : Union[list[PathLike], PathLike],
        ):
    if not isinstance(target_dir, os.PathLike):
        target_dir = Path(target_dir)
    if not isinstance(dbfiles, list):
        dbfiles = [dbfiles]
    if not dbfiles:
        raise ValueError("'dbfiles' empty")
    downloaded = set()
    for dbfile_fname in dbfiles:
        dbfile_fname = target_dir / dbfile_fname
        if dbfile_fname.is_file():
            with open(dbfile_fname, "r") as f:
                downloaded.update(json.load(f)["downloaded"])

    if add and url not in downloaded:
        downloaded.add(url)
        with open(dbfiles[0], "w") as f:
            data = {"downloaded": sorted(downloaded)}
            json.dump(data, f, indent=2, sort_keys=True)
        return True

    return url in downloaded
