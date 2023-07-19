from . import common
from ..data.stanford import config
from ..data.stanford import read
from ..data.common import scan
from typing import Iterable, Optional, Union
import os


class SingleViewUVScanDataset(common.H5Dataset):
    def __init__(self,
            obj_names   : Iterable[str],
            identifiers : Optional[Iterable[str]]       = None,
            data_path   : Union[str, os.PathLike, None] = None,
            ):
        if not obj_names:
            raise ValueError("'obj_names' cannot be empty!")
        if identifiers is None:
            identifiers = read.list_mesh_scan_identifiers()
        if data_path is not None:
            config.DATA_PATH = data_path
        fnames = read.list_mesh_scan_uv_h5_fnames(obj_names, identifiers)
        super().__init__(
            h5_dataclass_cls = scan.SingleViewUVScan,
            fnames           = fnames,
        )

class AutodecoderSingleViewUVScanDataset(common.AutodecoderDataset):
    def __init__(self,
            obj_names   : Iterable[str],
            identifiers : Optional[Iterable[str]]       = None,
            data_path   : Union[str, os.PathLike, None] = None,
            ):
        if identifiers is None:
            identifiers = read.list_mesh_scan_identifiers()
        super().__init__(
            keys    = [obj_name for obj_name in obj_names for _ in range(len(identifiers))],
            dataset = SingleViewUVScanDataset(obj_names, identifiers, data_path=data_path),
        )


class SphereScanDataset(common.H5Dataset):
    def __init__(self,
            obj_names   : Iterable[str],
            data_path   : Union[str, os.PathLike, None] = None,
            ):
        if not obj_names:
            raise ValueError("'obj_names' cannot be empty!")
        if data_path is not None:
            config.DATA_PATH = data_path
        fnames = read.list_mesh_sphere_scan_h5_fnames(obj_names)
        super().__init__(
            h5_dataclass_cls = scan.SingleViewUVScan,
            fnames           = fnames,
        )

class AutodecoderSphereScanDataset(common.AutodecoderDataset):
    def __init__(self,
            obj_names   : Iterable[str],
            data_path   : Union[str, os.PathLike, None] = None,
            ):
        super().__init__(
            keys    = obj_names,
            dataset = SphereScanDataset(obj_names, data_path=data_path),
        )
