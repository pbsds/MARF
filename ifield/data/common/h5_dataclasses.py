#!/usr/bin/env python3
from abc import abstractmethod, ABCMeta
from collections import namedtuple
from pathlib import Path
import copy
import dataclasses
import functools
import h5py as h5
import hdf5plugin
import numpy as np
import operator
import os
import sys
import typing

__all__ = [
    "DataclassMeta",
    "Dataclass",
    "H5Dataclass",
    "H5Array",
    "H5ArrayNoSlice",
]

T        = typing.TypeVar("T")
NoneType = type(None)
PathLike = typing.Union[os.PathLike, str]
H5Array         = typing._alias(np.ndarray, 0, inst=False, name="H5Array")
H5ArrayNoSlice  = typing._alias(np.ndarray, 0, inst=False, name="H5ArrayNoSlice")

DataclassField = namedtuple("DataclassField", [
    "name",
    "type",
    "is_optional",
    "is_array",
    "is_sliceable",
    "is_prefix",
])

def strip_optional(val: type) -> type:
    if typing.get_origin(val) is typing.Union:
        union = set(typing.get_args(val))
        if len(union - {NoneType}) == 1:
            val, = union - {NoneType}
        else:
            raise TypeError(f"Non-'typing.Optional' 'typing.Union' is not supported: {typing._type_repr(val)!r}")
    return val

def is_array(val, *, _inner=False):
    """
    Hacky way to check if a value or type is an array.
    The hack omits having to depend on large frameworks such as pytorch or pandas
    """
    val = strip_optional(val)
    if val is H5Array or val is H5ArrayNoSlice:
        return True

    if typing._type_repr(val) in (
        "numpy.ndarray",
        "torch.Tensor",
    ):
        return True
    if not _inner:
        return is_array(type(val), _inner=True)
    return False

def prod(numbers: typing.Iterable[T], initial: typing.Optional[T] = None) -> T:
    if initial is not None:
        return functools.reduce(operator.mul, numbers, initial)
    else:
        return functools.reduce(operator.mul, numbers)

class DataclassMeta(type):
    def __new__(
            mcls,
            name  : str,
            bases : tuple[type, ...],
            attrs : dict[str, typing.Any],
            **kwargs,
            ):
        cls = super().__new__(mcls, name, bases, attrs, **kwargs)
        if sys.version_info[:2] >= (3, 10) and not hasattr(cls, "__slots__"):
            cls = dataclasses.dataclass(slots=True)(cls)
        else:
            cls = dataclasses.dataclass(cls)
        return cls

class DataclassABCMeta(DataclassMeta, ABCMeta):
    pass

class Dataclass(metaclass=DataclassMeta):
    def __getitem__(self, key: str) -> typing.Any:
        if key in self.keys():
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: typing.Any):
        if key in self.keys():
            return setattr(self, key, value)
        raise KeyError(key)

    def keys(self) -> typing.KeysView:
        return self.as_dict().keys()

    def values(self) -> typing.ValuesView:
        return self.as_dict().values()

    def items(self) -> typing.ItemsView:
        return self.as_dict().items()

    def as_dict(self, properties_to_include: set[str] = None, **kw) -> dict[str, typing.Any]:
        out = dataclasses.asdict(self, **kw)
        for name in (properties_to_include or []):
            out[name] = getattr(self, name)
        return out

    def as_tuple(self, properties_to_include: list[str]) -> tuple:
        out = dataclasses.astuple(self)
        if not properties_to_include:
            return out
        else:
            return (
                *out,
                *(getattr(self, name) for name in properties_to_include),
            )

    def copy(self: T, *, deep=True) -> T:
        return (copy.deepcopy if deep else copy.copy)(self)

class H5Dataclass(Dataclass):
    # settable with class params:
    _prefix      : str  = dataclasses.field(init=False, repr=False, default="")
    _n_pages     : int  = dataclasses.field(init=False, repr=False, default=10)
    _require_all : bool = dataclasses.field(init=False, repr=False, default=False)

    def __init_subclass__(cls,
            prefix      : typing.Optional[str]  = None,
            n_pages     : typing.Optional[int]  = None,
            require_all : typing.Optional[bool] = None,
            **kw,
            ):
        super().__init_subclass__(**kw)
        assert dataclasses.is_dataclass(cls)
        if prefix      is not None: cls._prefix      = prefix
        if n_pages     is not None: cls._n_pages     = n_pages
        if require_all is not None: cls._require_all = require_all

    @classmethod
    def _get_fields(cls) -> typing.Iterable[DataclassField]:
        for field in dataclasses.fields(cls):
            if not field.init:
                continue
            assert field.name not in ("_prefix", "_n_pages", "_require_all"), (
                f"{field.name!r} can not be in {cls.__qualname__}.__init__.\n"
                "Set it with dataclasses.field(default=YOUR_VALUE, init=False, repr=False)"
            )
            if isinstance(field.type, str):
                raise TypeError("Type hints are strings, perhaps avoid using `from __future__ import annotations`")

            type_inner = strip_optional(field.type)
            is_prefix  = typing.get_origin(type_inner) is dict and typing.get_args(type_inner)[:1] == (str,)
            field_type = typing.get_args(type_inner)[1] if is_prefix else field.type
            if field.default is None or typing.get_origin(field.type) is typing.Union and NoneType in typing.get_args(field.type):
                field_type = typing.Optional[field_type]

            yield DataclassField(
                name         = field.name,
                type         = strip_optional(field_type),
                is_optional  = typing.get_origin(field_type) is typing.Union and NoneType in typing.get_args(field_type),
                is_array     = is_array(field_type),
                is_sliceable = is_array(field_type) and strip_optional(field_type) is not H5ArrayNoSlice,
                is_prefix    = is_prefix,
            )

    @classmethod
    def from_h5_file(cls       : type[T],
            fname              : typing.Union[PathLike, str],
            *,
            page               : typing.Optional[int] = None,
            n_pages            : typing.Optional[int] = None,
            read_slice         : slice                = slice(None),
            require_even_pages : bool                 = True,
            ) -> T:
        if not isinstance(fname, Path):
            fname = Path(fname)
        if n_pages is None:
            n_pages = cls._n_pages
        if not fname.exists():
            raise FileNotFoundError(str(fname))
        if not h5.is_hdf5(fname):
            raise TypeError(f"Not a HDF5 file: {str(fname)!r}")

        # if this class has no fields, print a example class:
        if not any(field.init for field in dataclasses.fields(cls)):
            with h5.File(fname, "r") as f:
                klen = max(map(len, f.keys()))
                example_cls = f"\nclass {cls.__name__}(Dataclass, require_all=True):\n" + "\n".join(
                    f"    {k.ljust(klen)} : "
                    + (
                        "H5Array" if prod(v.shape, 1) > 1 else (
                        "float"   if issubclass(v.dtype.type, np.floating) else (
                        "int"     if issubclass(v.dtype.type, np.integer) else (
                        "bool"    if issubclass(v.dtype.type, np.bool_) else (
                        "typing.Any"
                    ))))).ljust(14 + 1)
                    + f" #{repr(v).split(':', 1)[1].removesuffix('>')}"
                    for k, v in f.items()
                )
            raise NotImplementedError(f"{cls!r} has no fields!\nPerhaps try the following:{example_cls}")

        fields_consumed = set()

        def make_kwarg(
                file  : h5.File,
                keys  : typing.KeysView,
                field : DataclassField,
                ) -> tuple[str, typing.Any]:
            if field.is_optional:
                if field.name not in keys:
                    return field.name, None
            if field.is_sliceable:
                if page is not None:
                    n_items = int(f[cls._prefix + field.name].shape[0])
                    page_len = n_items // n_pages
                    modulus  = n_items %  n_pages
                    if modulus: page_len += 1 # round up
                    if require_even_pages and modulus:
                        raise ValueError(f"Field {field.name!r} {tuple(f[cls._prefix + field.name].shape)} is not cleanly divisible into {n_pages} pages")
                    this_slice = slice(
                        start = page_len * page,
                        stop  = page_len * (page+1),
                        step  = read_slice.step, # inherit step
                    )
                else:
                    this_slice = read_slice
            else:
                this_slice = slice(None) # read all

            # array or scalar?
            def read_dataset(var):
                # https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data
                if field.is_array:
                    return var[this_slice]
                if var.shape == (1,):
                    return var[0]
                else:
                    return var[()]

            if field.is_prefix:
                fields_consumed.update(
                    key
                    for key in keys if key.startswith(f"{cls._prefix}{field.name}_")
                )
                return field.name, {
                    key.removeprefix(f"{cls._prefix}{field.name}_") : read_dataset(file[key])
                    for key in keys if key.startswith(f"{cls._prefix}{field.name}_")
                }
            else:
                fields_consumed.add(cls._prefix + field.name)
                return field.name, read_dataset(file[cls._prefix + field.name])

        with h5.File(fname, "r") as f:
            keys = f.keys()
            init_dict = dict( make_kwarg(f, keys, i) for i in cls._get_fields() )

            try:
                out = cls(**init_dict)
            except Exception as e:
                class_attrs = set(field.name for field in dataclasses.fields(cls))
                file_attr   = set(init_dict.keys())
                raise e.__class__(f"{e}. {class_attrs=}, {file_attr=}, diff={class_attrs.symmetric_difference(file_attr)}") from e

            if cls._require_all:
                fields_not_consumed = set(keys) - fields_consumed
                if fields_not_consumed:
                    raise ValueError(f"Not all HDF5 fields consumed: {fields_not_consumed!r}")

        return out

    def to_h5_file(self,
            fname : PathLike,
            mkdir : bool     = False,
            ):
        if not isinstance(fname, Path):
            fname = Path(fname)
        if not fname.parent.is_dir():
            if mkdir:
                fname.parent.mkdir(parents=True)
            else:
                raise NotADirectoryError(fname.parent)

        with h5.File(fname, "w") as f:
            for field in type(self)._get_fields():
                if field.is_optional and getattr(self, field.name) is None:
                    continue
                value = getattr(self, field.name)
                if field.is_array:
                    if any(type(i) is not np.ndarray for i in (value.values() if field.is_prefix else [value])):
                        raise TypeError(
                            "When dumping a H5Dataclass, make sure the array fields are "
                            f"numpy arrays (the type of {field.name!r} is {typing._type_repr(type(value))}).\n"
                            "Example: h5dataclass.map_arrays(torch.Tensor.numpy)"
                        )
                else:
                    pass

                def write_value(key: str, value: typing.Any):
                    if field.is_array:
                        f.create_dataset(key, data=value, **hdf5plugin.LZ4())
                    else:
                        f.create_dataset(key, data=value)

                if field.is_prefix:
                    for k, v in value.items():
                        write_value(self._prefix + field.name + "_" + k, v)
                else:
                    write_value(self._prefix + field.name, value)

    def map_arrays(self: T, func: typing.Callable[[H5Array], H5Array], do_copy: bool = False) -> T:
        if do_copy: # shallow
            self = self.copy(deep=False)
        for field in type(self)._get_fields():
            if field.is_optional and getattr(self, field.name) is None:
                continue
            if field.is_prefix and field.is_array:
                setattr(self, field.name, {
                    k : func(v)
                    for k, v in getattr(self, field.name).items()
                })
            elif field.is_array:
                setattr(self, field.name, func(getattr(self, field.name)))

        return self

    def astype(self: T, t: type, do_copy: bool = False, convert_nonfloats: bool = False) -> T:
        return self.map_arrays(lambda x: x.astype(t) if convert_nonfloats or not np.issubdtype(x.dtype, int) else x)

    def copy(self: T, *, deep=True) -> T:
        out = super().copy(deep=deep)
        if not deep:
            for field in type(self)._get_fields():
                if field.is_prefix:
                    out[field.name] = copy.copy(field.name)
        return out

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return {
            key: value.shape
            for key, value in self.items()
            if hasattr(value, "shape")
        }

class TransformableDataclassMixin(metaclass=DataclassABCMeta):

    @abstractmethod
    def transform(self: T, mat4: np.ndarray, inplace=False) -> T:
        ...

    def transform_to(self: T, name: str, inverse_name: str = None, *, inplace=False) -> T:
        mtx = self.transforms[name]
        out = self.transform(mtx, inplace=inplace)
        out.transforms.pop(name) # consumed

        inv = np.linalg.inv(mtx)
        for key in list(out.transforms.keys()): # maintain the other transforms
            out.transforms[key] = out.transforms[key] @ inv
        if inverse_name is not None: # store inverse
            out.transforms[inverse_name] = inv

        return out
