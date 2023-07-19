from ..config import data_path_get, data_path_persist
from collections import namedtuple
import os


# Data source:
# http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm

__ALL__ = ["config", "Model", "MODELS"]

Archive = namedtuple("Archive", "url fname download_size_str")

@(lambda x: x()) # singleton
class config:
    DATA_PATH = property(
        doc  = """
            Path to the dataset. The following envvars override it:
            ${IFIELD_DATA_MODELS}/coseg
            ${IFIELD_DATA_MODELS_COSEG}
        """,
        fget = lambda self:       data_path_get    ("coseg"),
        fset = lambda self, path: data_path_persist("coseg", path),
    )

    @property
    def IS_DOWNLOADED_DB(self) -> list[os.PathLike]:
        return [
            self.DATA_PATH / "downloaded.json",
        ]

    SHAPES: dict[str, Archive] = {
        "candelabra"  : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Candelabra/shapes.zip",   "candelabra-shapes.zip",   "3,3M"),
        "chair"       : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Chair/shapes.zip",        "chair-shapes.zip",        "3,2M"),
        "four-legged" : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Four-legged/shapes.zip",  "four-legged-shapes.zip",  "2,9M"),
        "goblets"     : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Goblets/shapes.zip",      "goblets-shapes.zip",      "500K"),
        "guitars"     : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Guitars/shapes.zip",      "guitars-shapes.zip",      "1,9M"),
        "lampes"      : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Lampes/shapes.zip",       "lampes-shapes.zip",       "2,4M"),
        "vases"       : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Vases/shapes.zip",        "vases-shapes.zip",        "5,5M"),
        "irons"       : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Irons/shapes.zip",        "irons-shapes.zip",        "1,2M"),
        "tele-aliens" : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Tele-aliens/shapes.zip",  "tele-aliens-shapes.zip",  "15M"),
        "large-vases" : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/shapes.zip",  "large-vases-shapes.zip",  "6,2M"),
        "large-chairs": Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/shapes.zip", "large-chairs-shapes.zip", "14M"),
    }
    GROUND_TRUTHS: dict[str, Archive] = {
        "candelabra"  : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Candelabra/gt.zip",   "candelabra-gt.zip",   "68K"),
        "chair"       : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Chair/gt.zip",        "chair-gt.zip",        "20K"),
        "four-legged" : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Four-legged/gt.zip",  "four-legged-gt.zip",  "24K"),
        "goblets"     : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Goblets/gt.zip",      "goblets-gt.zip",      "4,0K"),
        "guitars"     : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Guitars/gt.zip",      "guitars-gt.zip",      "12K"),
        "lampes"      : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Lampes/gt.zip",       "lampes-gt.zip",       "60K"),
        "vases"       : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Vases/gt.zip",        "vases-gt.zip",        "40K"),
        "irons"       : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Irons/gt.zip",        "irons-gt.zip",        "8,0K"),
        "tele-aliens" : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Tele-aliens/gt.zip",  "tele-aliens-gt.zip",  "72K"),
        "large-vases" : Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/gt.zip",  "large-vases-gt.zip",  "68K"),
        "large-chairs": Archive("http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/gt.zip", "large-chairs-gt.zip", "116K"),
    }
