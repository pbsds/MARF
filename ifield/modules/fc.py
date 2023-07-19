from . import siren
from .. import param
from ..utils.helpers import compose, run_length_encode, MetaModuleProxy
from collections import OrderedDict
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import nn, Tensor
from torch.nn.utils.weight_norm import WeightNorm
from torchmeta.modules import MetaModule, MetaSequential
from typing import Iterable, Literal, Optional, Union, Callable
import itertools
import math
import torch

__doc__ = """
`fc` is short for "Fully Connected"
"""

def broadcast_tensors_except(*tensors: Tensor, dim: int) -> list[Tensor]:
    if dim == -1:
        shapes = [ i.shape[:dim] for i in tensors ]
    else:
        shapes = [ (*i.shape[:dim], i.shape[dim+1:]) for i in tensors ]
    target_shape = list(torch.broadcast_shapes(*shapes))
    if dim == -1:
        target_shape.append(-1)
    elif dim < 0:
        target_shape.insert(dim+1, -1)
    else:
        target_shape.insert(dim, -1)

    return [ i.broadcast_to(target_shape) for i in tensors ]


EPS = 1e-8

Nonlinearity = Literal[
    None,
    "relu",
    "leaky_relu",
    "silu",
    "softplus",
    "elu",
    "selu",
    "sine",
    "sigmoid",
    "tanh",
]

Normalization = Literal[
    None,
    "batchnorm",
    "batchnorm_na",
    "layernorm",
    "layernorm_na",
    "weightnorm",
]

class ReprHyperparametersMixin(HyperparametersMixin):
    def extra_repr(self):
        this = ", ".join(f"{k}={v!r}" for k, v in self.hparams.items())
        rest = super().extra_repr()
        if rest:
            return f"{this}, {rest}"
        else:
            return this

class MultilineReprHyperparametersMixin(HyperparametersMixin):
    def extra_repr(self):
        items = [f"{k}={v!r}" for k, v in self.hparams.items()]
        this  = "\n".join(
            ", ".join(filter(bool, i)) + ","
            for i in itertools.zip_longest(items[0::3], items[1::3], items[2::3])
        )
        rest = super().extra_repr()
        if rest:
            return f"{this}, {rest}"
        else:
            return this


class BatchLinear(nn.Linear):
    """
    A linear (meta-)layer that can deal with batched weight matrices and biases,
    as for instance output by a hypernetwork.
    """
    __doc__ = nn.Linear.__doc__
    _meta_forward_pre_hooks = None

    def register_forward_pre_hook(self, hook: Callable) -> torch.utils.hooks.RemovableHandle:
        if not isinstance(hook, WeightNorm):
            return super().register_forward_pre_hook(hook)

        if self._meta_forward_pre_hooks is None:
            self._meta_forward_pre_hooks = OrderedDict()

        handle = torch.utils.hooks.RemovableHandle(self._meta_forward_pre_hooks)
        self._meta_forward_pre_hooks[handle.id] = hook
        return handle

    def forward(self, input: Tensor, params: Optional[dict[str, Tensor]]=None):
        if params is None or not isinstance(self, MetaModule):
            params = OrderedDict(self.named_parameters())
        if self._meta_forward_pre_hooks is not None:
            proxy = MetaModuleProxy(self, params)
            for hook in self._meta_forward_pre_hooks.values():
                hook(proxy, [input])

        weight = params["weight"]
        bias = params.get("bias", None)

        # transpose weights
        weight = weight.permute(*range(len(weight.shape) - 2), -1, -2) # does not jit

        output = input.unsqueeze(-2).matmul(weight).squeeze(-2)

        if bias is not None:
            output = output + bias

        return output


class MetaBatchLinear(BatchLinear, MetaModule):
    pass


class CallbackConcatLayer(nn.Module):
    "A tricky way to enable skip connections in sequentials models"
    def __init__(self, tensor_getter: Callable[[], tuple[Tensor, ...]]):
        super().__init__()
        self.tensor_getter = tensor_getter

    def forward(self, x):
        ys = self.tensor_getter()
        return torch.cat(broadcast_tensors_except(x, *ys, dim=-1), dim=-1)


class ResidualSkipConnectionEndLayer(nn.Module):
    """
    Residual skip connections that can be added to a nn.Sequential
    """

    class ResidualSkipConnectionStartLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self._stored_tensor = None

        def forward(self, x):
            assert self._stored_tensor is None
            self._stored_tensor = x
            return x

        def get(self):
            assert self._stored_tensor is not None
            x = self._stored_tensor
            self._stored_tensor = None
            return x

    def __init__(self):
        super().__init__()
        self._stored_tensor = None
        self._start = self.ResidualSkipConnectionStartLayer()

    def forward(self, x):
        skip = self._start.get()
        return x + skip

    @property
    def start(self) -> ResidualSkipConnectionStartLayer:
        return self._start

    @property
    def end(self) -> "ResidualSkipConnectionEndLayer":
        return self


ResidualMode = Literal[
    None,
    "identity",
]

class FCLayer(MultilineReprHyperparametersMixin, MetaSequential):
    """
    A single fully connected (FC) layer
    """

    def __init__(self,
            in_features     : int,
            out_features    : int,
            *,
            nonlinearity    : Nonlinearity  = "relu",
            normalization   : Normalization = None,
            is_first        : bool          = False,   # used for SIREN initialization
            is_final        : bool          = False,   # used for fan_out init
            dropout_prob    : float         = 0.0,
            negative_slope  : float         = 0.01,    # only for nonlinearity="leaky_relu", default is normally 0.01
            omega_0         : float         = 30,      # only for nonlinearity="sine"
            residual_mode   : ResidualMode  = None,
            _no_meta        : bool          = False,   # set to true in hypernetworks
            **_
            ):
        super().__init__()
        self.save_hyperparameters()

        # improve repr
        if nonlinearity != "leaky_relu":
            self.hparams.pop("negative_slope")
        if nonlinearity != "sine":
            self.hparams.pop("omega_0")

        Linear = nn.Linear if _no_meta else MetaBatchLinear

        def make_layer() -> Iterable[nn.Module]:
            # residual start
            if residual_mode is not None:
                residual_layer = ResidualSkipConnectionEndLayer()
                yield "res_a", residual_layer.start

            linear = Linear(in_features, out_features)

            # initialize
            if nonlinearity in {"relu", "leaky_relu", "silu", "softplus"}:
                nn.init.kaiming_uniform_(linear.weight, a=negative_slope, nonlinearity=nonlinearity, mode="fan_in" if not is_final else "fan_out")
            elif nonlinearity == "elu":
                nn.init.normal_(linear.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(linear.weight.size(-1)))
            elif nonlinearity == "selu":
                nn.init.normal_(linear.weight, std=1 / math.sqrt(linear.weight.size(-1)))
            elif nonlinearity == "sine":
                siren.init_weights_(linear, omega_0, is_first)
            elif nonlinearity in {"sigmoid", "tanh"}:
                nn.init.xavier_normal_(linear.weight)
            elif nonlinearity is None:
                pass # this is effectively uniform(-1/sqrt(in_features), 1/sqrt(in_features))
            else:
                raise NotImplementedError(nonlinearity)

            # linear + normalize
            if normalization is None:
                yield "linear", linear
            elif normalization == "batchnorm":
                yield "linear", linear
                yield "norm", nn.BatchNorm1d(out_features, affine=True)
            elif normalization == "batchnorm_na":
                yield "linear", linear
                yield "norm", nn.BatchNorm1d(out_features, affine=False)
            elif normalization == "layernorm":
                yield "linear", linear
                yield "norm", nn.LayerNorm([out_features], elementwise_affine=True)
            elif normalization == "layernorm_na":
                yield "linear", linear
                yield "norm", nn.LayerNorm([out_features], elementwise_affine=False)
            elif normalization == "weightnorm":
                yield "linear", nn.utils.weight_norm(linear)
            else:
                raise NotImplementedError(normalization)

            # activation
            inplace = False
            if   nonlinearity is None         : pass
            elif nonlinearity == "relu"       : yield nonlinearity, nn.ReLU(inplace=inplace)
            elif nonlinearity == "leaky_relu" : yield nonlinearity, nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
            elif nonlinearity == "silu"       : yield nonlinearity, nn.SiLU(inplace=inplace)
            elif nonlinearity == "softplus"   : yield nonlinearity, nn.Softplus()
            elif nonlinearity == "elu"        : yield nonlinearity, nn.ELU(inplace=inplace)
            elif nonlinearity == "selu"       : yield nonlinearity, nn.SELU(inplace=inplace)
            elif nonlinearity == "sine"       : yield nonlinearity, siren.Sine(omega_0)
            elif nonlinearity == "sigmoid"    : yield nonlinearity, nn.Sigmoid()
            elif nonlinearity == "tanh"       : yield nonlinearity, nn.Tanh()
            else                              : raise NotImplementedError(f"{nonlinearity=}")

            # dropout
            if dropout_prob > 0:
                if nonlinearity == "selu":
                    yield "adropout", nn.AlphaDropout(p=dropout_prob)
                else:
                    yield "dropout", nn.Dropout(p=dropout_prob)

            # residual end
            if residual_mode is not None:
                yield "res_b", residual_layer.end

        for name, module in make_layer():
            self.add_module(name.replace("-", "_"), module)

    @property
    def nonlinearity(self) -> Optional[nn.Module]:
        "alias to the activation function submodule"
        if self.hparams.nonlinearity is None:
            return None
        return getattr(self, self.hparams.nonlinearity.replace("-", "_"))

    def initialize_weights():
        raise NotImplementedError


class FCBlock(MultilineReprHyperparametersMixin, MetaSequential):
    """
    A block of FC layers
    """
    def __init__(self,
            in_features               : int,
            hidden_features           : int,
            hidden_layers             : int,
            out_features              : int,
            normalization             : Normalization = None,
            nonlinearity              : Nonlinearity  = "relu",
            dropout_prob              : float         = 0.0,
            outermost_linear          : bool          = True, # whether last linear is nonlinear
            latent_features           : Optional[int] = None,
            concat_skipped_layers     : Union[list[int], bool] = [],
            concat_conditioned_layers : Union[list[int], bool] = [],
            **kw,
            ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(concat_skipped_layers, bool):
            concat_skipped_layers = list(range(hidden_layers+2)) if concat_skipped_layers else []
        if isinstance(concat_conditioned_layers, bool):
            concat_conditioned_layers = list(range(hidden_layers+2)) if concat_conditioned_layers else []
        if len(concat_conditioned_layers) != 0 and latent_features is None:
            raise ValueError("Layers marked to be conditioned without known number of latent features")
        concat_skipped_layers     = [i if i >= 0 else hidden_layers+2-abs(i) for i in concat_skipped_layers]
        concat_conditioned_layers = [i if i >= 0 else hidden_layers+2-abs(i) for i in concat_conditioned_layers]
        self._concat_x_layers: frozenset[int] = frozenset(concat_skipped_layers)
        self._concat_z_layers: frozenset[int] = frozenset(concat_conditioned_layers)
        if len(self._concat_x_layers) != len(concat_skipped_layers):
            raise ValueError(f"Duplicates found in {concat_skipped_layers = }")
        if len(self._concat_z_layers) != len(concat_conditioned_layers):
            raise ValueError(f"Duplicates found in {concat_conditioned_layers = }")
        if not all(isinstance(i, int) for i in self._concat_x_layers):
            raise TypeError(f"Expected only integers in {concat_skipped_layers = }")
        if not all(isinstance(i, int) for i in self._concat_z_layers):
            raise TypeError(f"Expected only integers in {concat_conditioned_layers = }")

        def make_layers() -> Iterable[nn.Module]:
            def make_concat_layer(*idxs: int) -> int:
                x_condition_this_layer = any(idx in self._concat_x_layers for idx in idxs)
                z_condition_this_layer = any(idx in self._concat_z_layers for idx in idxs)
                if x_condition_this_layer and z_condition_this_layer:
                    yield CallbackConcatLayer(lambda: (self._current_x, self._current_z))
                elif x_condition_this_layer:
                    yield CallbackConcatLayer(lambda: (self._current_x,))
                elif z_condition_this_layer:
                    yield CallbackConcatLayer(lambda: (self._current_z,))

                return in_features*x_condition_this_layer + (latent_features or 0)*z_condition_this_layer

            added = yield from make_concat_layer(0)

            yield FCLayer(
                in_features   = in_features + added,
                out_features  = hidden_features,
                nonlinearity  = nonlinearity,
                normalization = normalization,
                dropout_prob  = dropout_prob,
                is_first      = True,
                is_final      = False,
                **kw,
            )

            for i in range(hidden_layers):
                added = yield from make_concat_layer(i+1)

                yield FCLayer(
                    in_features   = hidden_features  + added,
                    out_features  = hidden_features,
                    nonlinearity  = nonlinearity,
                    normalization = normalization,
                    dropout_prob  = dropout_prob,
                    is_first      = False,
                    is_final      = False,
                    **kw,
                )

            added = yield from make_concat_layer(hidden_layers+1)

            nl = nonlinearity

            yield FCLayer(
                in_features   = hidden_features + added,
                out_features  = out_features,
                nonlinearity  = None if outermost_linear else nl,
                normalization = None if outermost_linear else normalization,
                dropout_prob  = 0.0  if outermost_linear else dropout_prob,
                is_first      = False,
                is_final      = True,
                **kw,
            )

        for i, module in enumerate(make_layers()):
            self.add_module(str(i), module)

    @property
    def is_conditioned(self) -> bool:
        "Whether z is used or not"
        return bool(self._concat_z_layers)

    @classmethod
    @compose("\n".join)
    def make_jinja_template(cls, *, exclude_list: set[str] = {}, top_level: bool = True, **kw) -> str:
        @compose(" ".join)
        def as_jexpr(values: Union[list[int]]):
            yield "{{"
            for val, count in run_length_encode(values):
                yield f"[{val!r}]*{count!r}"
            yield "}}"
        yield param.make_jinja_template(cls, top_level=top_level, exclude_list=exclude_list)
        yield param.make_jinja_template(FCLayer, top_level=False, exclude_list=exclude_list | {
            "in_features",
            "out_features",
            "nonlinearity",
            "normalization",
            "dropout_prob",
            "is_first",
            "is_final",
        })

    def forward(self, input: Tensor, z: Optional[Tensor] = None, *, params: Optional[dict[str, Tensor]]=None):
        assert not self.is_conditioned or z is not None
        if z is not None and z.ndim < input.ndim:
            z = z[(*(None,)*(input.ndim - z.ndim), ...)]
        self._current_x = input
        self._current_z = z
        return super().forward(input, params=params)
