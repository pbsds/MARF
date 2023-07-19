from .. import param
from ..modules import fc
from ..utils import geometry
from ..utils.helpers import compose
from textwrap import indent, dedent
from torch import nn, Tensor
from typing import Optional
import warnings

class OrthogonalPlaneNet(nn.Module):
    """

    """

    def __init__(self,
            in_features     : int,
            latent_features : int,
            hidden_features : int,
            hidden_layers   : int,
            **kw,
            ):
        super().__init__()

        self.fc = fc.FCBlock(
            in_features      = in_features,
            hidden_layers    = hidden_layers,
            hidden_features  = hidden_features,
            out_features     = 2,  # (plane_offset, is_intersecting)
            outermost_linear = True,
            latent_features  = latent_features,
            **kw,
        )

    @property
    def is_conditioned(self):
        return self.fc.is_conditioned

    @classmethod
    @compose("\n".join)
    def make_jinja_template(cls, *, exclude_list: set[str] = {}, top_level: bool = True, **kw) -> str:
        yield param.make_jinja_template(cls, top_level=top_level, exclude_list=exclude_list, **kw)
        yield param.make_jinja_template(fc.FCBlock, top_level=False, exclude_list={
            "in_features",
            "hidden_layers",
            "hidden_features",
            "out_features",
            "outermost_linear",
        })

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        if __debug__ and self.is_conditioned and z is None:
            warnings.warn(f"{self.__class__.__qualname__} is conditioned, but the forward pass was not supplied with a conditioning tensor.")
        return self.fc(x, z)

    @staticmethod
    def compute_intersections(
            ray_origins  : Tensor, # (..., 3)
            ray_dirs     : Tensor, # (..., 3)
            predictions  : Tensor, # (..., 2)
            *,
            normalize_origins = True,
            return_signed_displacements = False,
            allow_nans = False, # MARF compat
            atom_random_prob  = None, # MARF compat
            atom_dropout_prob = None, # MARF compat
            ) -> tuple[(Tensor,)*5]:
        assert ray_origins.shape[:-1] == ray_dirs.shape[:-1] == predictions.shape[:-1], \
            (ray_origins.shape, ray_dirs.shape, predictions.shape)
        assert predictions.shape[-1] == 2, \
            predictions.shape

        assert not allow_nans

        if normalize_origins:
            ray_origins = geometry.project_point_on_ray(0, ray_origins, ray_dirs)

        # unpack predictions
        signed_displacements = predictions[..., 0]
        is_intersecting      = predictions[..., 1]

        # compute intersections
        intersections = ray_origins - signed_displacements[..., None] * ray_dirs

        return (
            intersections,
            is_intersecting,
            *((signed_displacements,) if return_signed_displacements else ()),
        )




OrthogonalPlaneNet.__doc__ = __doc__ = f"""
{dedent(OrthogonalPlaneNet.__doc__).strip()}

# Config template:

```yaml
{OrthogonalPlaneNet.make_jinja_template()}
```
"""
