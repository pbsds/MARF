from .h5_dataclasses import H5Dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Hashable, Optional, Callable
import os

DEBUG = bool(os.environ.get("IFIELD_DEBUG", ""))

__doc__ = """
Here are some helper functions for processing data.
"""

# multiprocessing does not work due to my rediculous use of closures, which seemingly cannot be pickled
# paralelize it in the shell instead

def precompute_data(
        computer     : Callable[[Hashable], Optional[H5Dataclass]],
        identifiers  : list[Hashable],
        output_paths : list[Path],
        page         : tuple[int, int] = (0, 1),
        *,
        force        : bool = False,
        debug        : bool = False,
        ):
    """
    precomputes data and stores them as HDF5 datasets using `.to_file(path: Path)`
    """

    page, n_pages = page
    assert len(identifiers) == len(output_paths)

    total = len(identifiers)
    identifier_max_len = max(map(len, map(str, identifiers)))
    t_epoch = None
    def log(state: str, is_start = False):
        nonlocal t_epoch
        if is_start: t_epoch = datetime.now()
        td = timedelta(0) if is_start else datetime.now() - t_epoch
        print(" - "
            f"{str(index+1).rjust(len(str(total)))}/{total}: "
            f"{str(identifier).ljust(identifier_max_len)} @ {td}: {state}"
        )

    print(f"precompute_data(computer={computer.__module__}.{computer.__qualname__}, identifiers=..., force={force}, page={page})")
    t_begin = datetime.now()
    failed = []

    # pagination
    page_size = total // n_pages + bool(total % n_pages)
    jobs = list(zip(identifiers, output_paths))[page_size*page : page_size*(page+1)]

    for index, (identifier, output_path) in enumerate(jobs, start=page_size*page):
        if not force and output_path.exists() and output_path.stat().st_size > 0:
            continue

        log("compute", is_start=True)

        # compute
        try:
            res = computer(identifier)
        except Exception as e:
            failed.append(identifier)
            log(f"failed compute: {e.__class__.__name__}: {e}")
            if DEBUG or debug: raise e
            continue
        if res is None:
            failed.append(identifier)
            log("no result")
            continue

        # write to file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            res.to_h5_file(output_path)
        except Exception as e:
            failed.append(identifier)
            log(f"failed write: {e.__class__.__name__}: {e}")
            if output_path.is_file(): output_path.unlink() # cleanup
            if DEBUG or debug: raise e
            continue

        log("done")

    print("precompute_data finished in", datetime.now() - t_begin)
    print("failed:", failed or None)
