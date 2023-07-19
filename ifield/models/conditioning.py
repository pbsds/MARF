from abc import ABC, abstractmethod
from torch import nn, Tensor
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
from typing import Hashable, Union, Optional, KeysView, ValuesView, ItemsView, Any, Sequence
import torch


class RequiresConditioner(nn.Module, ABC): # mixin

    @property
    @abstractmethod
    def n_latent_features(self) -> int:
        "This should provide the width of the conditioning feature vector"
        ...

    @property
    @abstractmethod
    def latent_embeddings_init_std(self) -> float:
        "This should provide the standard deviation to initialize the latent features with. DeepSDF uses 0.01."
        ...

    @property
    @abstractmethod
    def latent_embeddings() -> Optional[Tensor]:
        """This property should return a tensor cotnaining all stored embeddings, for use in computing auto-decoder losses"""
        ...

    @abstractmethod
    def encode(self, batch: Any, batch_idx: int, optimizer_idx: int) -> Tensor:
        "This should, given a training batch, return the encoded conditioning vector"
        ...


class AutoDecoderModuleMixin(RequiresConditioner, ABC):
    """
    Populates dunder methods making it behave as a mapping.
    The mapping indexes into a stored set of learnable embedding vectors.

    Based on the auto-decoder architecture of
    J.J. Park, P. Florence, J. Straub, R. Newcombe, S. Lovegrove, DeepSDF:
    Learning Continuous Signed Distance Functions for Shape Representation, in:
    2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    IEEE, Long Beach, CA, USA, 2019: pp. 165–174.
    https://doi.org/10.1109/CVPR.2019.00025.
    """

    _autodecoder_mapping: dict[Hashable, int]
    autodecoder_embeddings: nn.Parameter

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        @self._register_load_state_dict_pre_hook
        def hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if f"{prefix}_autodecoder_mapping" in state_dict:
                state_dict[f"{prefix}{_EXTRA_STATE_KEY_SUFFIX}"] = state_dict.pop(f"{prefix}_autodecoder_mapping")

        class ICanBeLoadedFromCheckpointsAndChangeShapeStopBotheringMePyTorchAndSitInTheCornerIKnowWhatIAmDoing(nn.UninitializedParameter):
            def copy_(self, other):
                self.materialize(other.shape, other.device, other.dtype)
                return self.copy_(other)
        self.autodecoder_embeddings = ICanBeLoadedFromCheckpointsAndChangeShapeStopBotheringMePyTorchAndSitInTheCornerIKnowWhatIAmDoing()

    # nn.Module interface

    def get_extra_state(self):
        return {
            "ad_uids": getattr(self, "_autodecoder_mapping", {}),
        }

    def set_extra_state(self, obj):
        if "ad_uids" not in obj: # backward compat
            self._autodecoder_mapping = obj
        else:
            self._autodecoder_mapping = obj["ad_uids"]

    # RequiresConditioner interface

    @property
    def latent_embeddings(self) -> Tensor:
        return self.autodecoder_embeddings

    # my interface

    def set_observation_ids(self, z_uids: set[Hashable]):
        assert self.latent_embeddings_init_std is not None, f"{self.__module__}.{self.__class__.__qualname__}.latent_embeddings_init_std"
        assert self.n_latent_features          is not None, f"{self.__module__}.{self.__class__.__qualname__}.n_latent_features"
        assert self.latent_embeddings_init_std > 0, self.latent_embeddings_init_std
        assert self.n_latent_features          > 0, self.n_latent_features

        self._autodecoder_mapping = {
            k: i
            for i, k in enumerate(sorted(set(z_uids)))
        }

        if not len(z_uids) == len(self._autodecoder_mapping):
            raise ValueError(f"Observation identifiers are not unique! {z_uids = }")

        self.autodecoder_embeddings = nn.Parameter(
            torch.Tensor(len(self._autodecoder_mapping), self.n_latent_features)
                .normal_(mean=0, std=self.latent_embeddings_init_std)
                .to(self.device, self.dtype)
        )

    def add_key(self, z_uid: Hashable, z: Optional[Tensor] = None):
        if z_uid in self._autodecoder_mapping:
            raise ValueError(f"Observation identifier {z_uid!r} not unique!")

        self._autodecoder_mapping[z_uid] = len(self._autodecoder_mapping)
        self.autodecoder_embeddings
        raise NotImplementedError

    def __delitem__(self, z_uid: Hashable):
        i = self._autodecoder_mapping.pop(z_uid)
        for k, v in list(self._autodecoder_mapping.items()):
            if v > i:
                self._autodecoder_mapping[k] -= 1

        with torch.no_grad():
            self.autodecoder_embeddings = nn.Parameter(torch.cat((
                    self.autodecoder_embeddings.detach()[:i,   :],
                    self.autodecoder_embeddings.detach()[i+1:, :],
                ), dim=0))

    def __contains__(self, z_uid: Hashable) -> bool:
        return z_uid in self._autodecoder_mapping

    def __getitem__(self, z_uids: Union[Hashable, Sequence[Hashable]]) -> Tensor:
        if isinstance(z_uids, tuple) or isinstance(z_uids, list):
            key = tuple(map(self._autodecoder_mapping.__getitem__, z_uids))
        else:
            key = self._autodecoder_mapping[z_uids]
        return self.autodecoder_embeddings[key, :]

    def __iter__(self):
        return self._autodecoder_mapping.keys()

    def keys(self) -> KeysView[Hashable]:
        """
        lists the identifiers of each code
        """
        return self._autodecoder_mapping.keys()

    def values(self) -> ValuesView[Tensor]:
        return list(self.autodecoder_embeddings)

    def items(self) -> ItemsView[Hashable, Tensor]:
        """
        lists all the learned codes / latent vectors with their identifiers as keys
        """
        return {
            k : self.autodecoder_embeddings[i]
            for k, i in self._autodecoder_mapping.items()
        }.items()

class EncoderModuleMixin(RequiresConditioner, ABC):
    @property
    def latent_embeddings(self) -> None:
        return None
