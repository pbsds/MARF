from ..config import data_path_get, data_path_persist
from collections import namedtuple
import os


# Data source:
# http://graphics.stanford.edu/data/3Dscanrep/

__ALL__ = ["config", "Model", "MODELS"]

@(lambda x: x()) # singleton
class config:
    DATA_PATH = property(
        doc  = """
            Path to the dataset. The following envvars override it:
            ${IFIELD_DATA_MODELS}/stanford
            ${IFIELD_DATA_MODELS_STANFORD}
        """,
        fget = lambda self:       data_path_get    ("stanford"),
        fset = lambda self, path: data_path_persist("stanford", path),
    )

    @property
    def IS_DOWNLOADED_DB(self) -> list[os.PathLike]:
        return [
            self.DATA_PATH / "downloaded.json",
        ]

    Model = namedtuple("Model", "url mesh_fname download_size_str")
    MODELS: dict[str, Model] = {
        "bunny": Model(
            "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz",
            "bunny/reconstruction/bun_zipper.ply",
            "4.89M",
        ),
        "drill_bit": Model(
            "http://graphics.stanford.edu/pub/3Dscanrep/drill.tar.gz",
            "drill/reconstruction/drill_shaft_vrip.ply",
            "555k",
        ),
        "happy_buddha": Model(
            # religious symbol
            "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz",
            "happy_recon/happy_vrip.ply",
            "14.5M",
        ),
        "dragon": Model(
            # symbol of Chinese culture
            "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
            "dragon_recon/dragon_vrip.ply",
            "11.2M",
        ),
        "armadillo": Model(
            "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
            "armadillo.ply.gz",
            "3.87M",
        ),
        "lucy": Model(
            # Christian angel
            "http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz",
            "lucy.ply",
            "322M",
        ),
        "asian_dragon": Model(
            # symbol of Chinese culture
            "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz",
            "xyzrgb_dragon.ply.gz",
            "70.5M",
        ),
        "thai_statue": Model(
            # Hindu religious significance
            "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz",
            "xyzrgb_statuette.ply.gz",
            "106M",
        ),
    }
