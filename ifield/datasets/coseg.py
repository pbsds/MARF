from . import common
from ..data.coseg import config
from ..data.coseg import read
from ..data.common import scan
from typing import Iterable, Optional, Union
import os


class SingleViewUVScanDataset(common.H5Dataset):
    def __init__(self,
            object_sets : tuple[str],
            identifiers : Optional[Iterable[str]]       = None,
            data_path   : Union[str, os.PathLike, None] = None,
            ):
        if not object_sets:
            raise ValueError("'object_sets' cannot be empty!")
        if identifiers is None:
            identifiers = read.list_mesh_scan_identifiers()
        if data_path is not None:
            config.DATA_PATH = data_path
        models = read.list_model_ids(object_sets)
        fnames = read.list_mesh_scan_uv_h5_fnames(models, identifiers)
        super().__init__(
            h5_dataclass_cls = scan.SingleViewUVScan,
            fnames           = fnames,
        )

class AutodecoderSingleViewUVScanDataset(common.AutodecoderDataset):
    def __init__(self,
            object_sets : tuple[str],
            identifiers : Optional[Iterable[str]]       = None,
            data_path   : Union[str, os.PathLike, None] = None,
            ):
        if identifiers is None:
            identifiers = read.list_mesh_scan_identifiers()
        # here do this step first, such that all the duplicate strings reference the same object
        super().__init__(
            keys    = [key for key in read.list_model_id_strings(object_sets) for _ in range(len(identifiers))],
            dataset = SingleViewUVScanDataset(object_sets, identifiers, data_path=data_path),
        )
