from ..utils.helpers import make_relative
from pathlib import Path
from typing import Optional
import os
import warnings


def data_path_get(dataset_name: str, no_warn: bool = False) -> Path:
    dataset_envvar = f"IFIELD_DATA_MODELS_{dataset_name.replace(*'-_').upper()}"
    if dataset_envvar in os.environ:
        data_path = Path(os.environ[dataset_envvar])
    elif "IFIELD_DATA_MODELS" in os.environ:
        data_path = Path(os.environ["IFIELD_DATA_MODELS"]) / dataset_name
    else:
        data_path = Path(__file__).resolve().parent.parent.parent / "data" / "models" / dataset_name
    if not data_path.is_dir() and not no_warn:
        warnings.warn(f"{make_relative(data_path, Path.cwd()).__str__()!r} is not a directory!")
    return data_path

def data_path_persist(dataset_name: Optional[str], path: os.PathLike) -> os.PathLike:
    "Persist the datapath, ensuring subprocesses also will use it. The path passes through."

    if dataset_name is None:
        os.environ["IFIELD_DATA_MODELS"] = str(path)
    else:
        os.environ[f"IFIELD_DATA_MODELS_{dataset_name.replace(*'-_').upper()}"] = str(path)

    return path
