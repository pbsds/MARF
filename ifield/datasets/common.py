from ..data.common.h5_dataclasses import H5Dataclass, PathLike
from torch.utils.data import Dataset, IterableDataset
from typing import Any, Iterable, Hashable, TypeVar, Iterator, Callable
from functools import partial, lru_cache
import inspect


T    = TypeVar("T")
T_H5 = TypeVar("T_H5", bound=H5Dataclass)


class TransformableDatasetMixin:
    def __init_subclass__(cls):
        if getattr(cls, "_transformable_mixin_no_override_getitem", False):
            pass
        elif issubclass(cls, Dataset):
            if cls.__getitem__ is not cls._transformable_mixin_getitem_wrapper:
                cls._transformable_mixin_inner_getitem = cls.__getitem__
                cls.__getitem__ = cls._transformable_mixin_getitem_wrapper
        elif issubclass(cls, IterableDataset):
            if cls.__iter__ is not cls._transformable_mixin_iter_wrapper:
                cls._transformable_mixin_inner_iter = cls.__iter__
                cls.__iter__ = cls._transformable_mixin_iter_wrapper
        else:
            raise TypeError(f"{cls.__name__!r} is neither a Dataset nor a IterableDataset!")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._transforms = []

    # works as a decorator
    def map(self: T, func: callable = None, /, args=[], **kw) -> T:
        def wrapper(func) -> T:
            if args or kw:
                func = partial(func, *args, **kw)
            self._transforms.append(func)
            return self

        if func is None:
            return wrapper
        else:
            return wrapper(func)


    def _transformable_mixin_getitem_wrapper(self, index: int):
        if not self._transforms:
            out = self._transformable_mixin_inner_getitem(index) # (TransformableDatasetMixin, no transforms)
        else:
            out = self._transformable_mixin_inner_getitem(index) # (TransformableDatasetMixin, has transforms)
            for f in self._transforms:
                out = f(out) # (TransformableDatasetMixin)
        return out

    def _transformable_mixin_iter_wrapper(self):
        if not self._transforms:
            out = self._transformable_mixin_inner_iter() # (TransformableDatasetMixin, no transforms)
        else:
            out = self._transformable_mixin_inner_iter() # (TransformableDatasetMixin, has transforms)
            for f in self._transforms:
                out = map(f, out) # (TransformableDatasetMixin)
        return out


class TransformedDataset(Dataset, TransformableDatasetMixin):
    # used to wrap an another dataset
    def __init__(self, dataset: Dataset, transforms: Iterable[callable]):
        super().__init__()
        self.dataset = dataset
        for i in transforms:
            self.map(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index] # (TransformedDataset)


class TransformExtendedDataset(Dataset, TransformableDatasetMixin):
    _transformable_mixin_no_override_getitem = True
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * len(self._transforms)

    def __getitem__(self, index: int):
        n = len(self._transforms)
        assert n > 0, f"{len(self._transforms) = }"

        item      = index // n
        transform = self._transforms[index % n]
        return transform(self.dataset[item])


class CachedDataset(Dataset):
    # used to wrap an another dataset
    def __init__(self, dataset: Dataset, cache_size: int | None):
        super().__init__()
        self.dataset = dataset
        if cache_size is not None and cache_size > 0:
            self.cached_getter = lru_cache(cache_size, self.dataset.__getitem__)
        else:
            self.cached_getter = self.dataset.__getitem__

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.cached_getter(index)


class AutodecoderDataset(Dataset, TransformableDatasetMixin):
    def __init__(self,
            keys    : Iterable[Hashable],
            dataset : Dataset,
            ):
        super().__init__()
        self.ad_mapping = list(keys)
        self.dataset    = dataset
        if len(self.ad_mapping) != len(dataset):
            raise ValueError(f"__len__ mismatch between keys and dataset: {len(self.ad_mapping)} != {len(dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Hashable, Any]:
        return self.ad_mapping[index], self.dataset[index] # (AutodecoderDataset)

    def keys(self) -> list[Hashable]:
        return self.ad_mapping

    def values(self) -> Iterator:
        return iter(self.dataset)

    def items(self) -> Iterable[tuple[Hashable, Any]]:
        return zip(self.ad_mapping, self.dataset)


class FunctionDataset(Dataset, TransformableDatasetMixin):
    def __init__(self,
            getter     : Callable[[Hashable], T],
            keys       : list[Hashable],
            cache_size : int | None = None,
            ):
        super().__init__()
        if cache_size is not None and cache_size > 0:
            getter = lru_cache(cache_size)(getter)
        self.getter = getter
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> T:
        return self.getter(self.keys[index])

class H5Dataset(FunctionDataset):
    def __init__(self,
            h5_dataclass_cls   : type[T_H5],
            fnames             : list[PathLike],
            **kw,
            ):
        super().__init__(
            getter = h5_dataclass_cls.from_h5_file,
            keys   = fnames,
            **kw,
        )

class PaginatedH5Dataset(Dataset, TransformableDatasetMixin):
    def __init__(self,
            h5_dataclass_cls   : type[T_H5],
            fnames             : list[PathLike],
            n_pages            : int           = 10,
            require_even_pages : bool          = True,
            ):
        super().__init__()
        self.h5_dataclass_cls   = h5_dataclass_cls
        self.fnames             = fnames
        self.n_pages            = n_pages
        self.require_even_pages = require_even_pages

    def __len__(self) -> int:
        return len(self.fnames) * self.n_pages

    def __getitem__(self, index: int) -> T_H5:
        item = index // self.n_pages
        page = index %  self.n_pages

        return self.h5_dataclass_cls.from_h5_file( # (PaginatedH5Dataset)
            fname   = self.fname[item],
            page    = page,
            n_pages = self.n_pages,
            require_even_pages = self.require_even_pages,
        )
