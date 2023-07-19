from functools import wraps, reduce, partial
from itertools import zip_longest, groupby
from pathlib import Path
from typing import Iterable, TypeVar, Callable, Union, Optional, Mapping, Hashable
import collections
import operator
import re

Numeric = Union[int, float, complex]
T       = TypeVar("T")
S       = TypeVar("S")

# decorator
def compose(outer_func: Callable[[..., S], T], *outer_a, **outer_kw) -> Callable[..., T]:
    def wrapper(inner_func: Callable[..., S]):
        @wraps(inner_func)
        def wrapped(*a, **kw):
            return outer_func(*outer_a, inner_func(*a, **kw), **outer_kw)
        return wrapped
    return wrapper

def compose_star(outer_func: Callable[[..., S], T], *outer_a, **outer_kw) -> Callable[..., T]:
    def wrapper(inner_func: Callable[..., S]):
        @wraps(inner_func)
        def wrapped(*a, **kw):
            return outer_func(*outer_a, *inner_func(*a, **kw), **outer_kw)
        return wrapped
    return wrapper


# itertools

def elementwise_max(iterable: Iterable[Iterable[T]]) -> Iterable[T]:
    return reduce(lambda xs, ys: [*map(max, zip(xs, ys))], iterable)

def prod(numbers: Iterable[T], initial: Optional[T] = None) -> T:
    if initial is not None:
        return reduce(operator.mul, numbers, initial)
    else:
        return reduce(operator.mul, numbers)

def run_length_encode(data: Iterable[T]) -> Iterable[tuple[T, int]]:
    return (
        (x, len(y))
        for x, y in groupby(data)
    )


# text conversion

def camel_to_snake_case(text: str, sep: str = "_", join_abbreviations: bool = False) -> str:
    parts = (
        part.lower()
        for part in re.split(r'(?=[A-Z])', text)
        if part
    )
    if join_abbreviations:
        parts = list(parts)
        if len(parts) > 1:
            for i, (a, b) in list(enumerate(zip(parts[:-1], parts[1:])))[::-1]:
                if len(a) == len(b) == 1:
                    parts[i] = parts[i] + parts.pop(i+1)
    return sep.join(parts)

def snake_to_camel_case(text: str) -> str:
    return "".join(
        part.captialize()
        for part in text.split("_")
        if part
    )


# textwrap

def columnize_dict(data: dict, n_columns=2, prefix="", sep=" ") -> str:
    sub = (len(data) + 1) // n_columns
    return reduce(partial(columnize, sep=sep),
        (
            columnize(
                "\n".join([f"{'' if n else prefix}{i!r}" for i in data.keys()  ][n*sub : (n+1)*sub]),
                "\n".join([f": {i!r},"                   for i in data.values()][n*sub : (n+1)*sub]),
            )
            for n in range(n_columns)
        )
    )

def columnize(left: str, right: str, prefix="", sep=" ") -> str:
    left  = left .split("\n")
    right = right.split("\n")
    width = max(map(len, left)) if left else 0
    return "\n".join(
        f"{prefix}{a.ljust(width)}{sep}{b}"
        if b else
        f"{prefix}{a}"
        for a, b in zip_longest(left, right, fillvalue="")
    )


# pathlib

def make_relative(path: Union[Path, str], parent: Path = None) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if parent is None:
        parent = Path.cwd()
    try:
        return path.relative_to(parent)
    except ValueError:
        pass
    try:
        return ".." / path.relative_to(parent.parent)
    except ValueError:
        pass
    return path


# dictionaries

def update_recursive(target: dict, source: dict):
    """ Update two config dictionaries recursively. """
    for k, v in source.items():
        if isinstance(v, dict):
            if k not in target:
                target[k] = type(target)()
            update_recursive(target[k], v)
        else:
            target[k] = v

def map_tree(func: Callable[[T], S], val: Union[Mapping[Hashable, T], tuple[T, ...], list[T], T]) -> Union[Mapping[Hashable, S], tuple[S, ...], list[S], S]:
    if isinstance(val, collections.abc.Mapping):
        return {
            k: map_tree(func, subval)
            for k, subval in val.items()
        }
    elif isinstance(val, tuple):
        return tuple(
            map_tree(func, subval)
            for subval in val
        )
    elif isinstance(val, list):
        return [
            map_tree(func, subval)
            for subval in val
        ]
    else:
        return func(val)

def flatten_tree(val, *, sep=".", prefix=None):
    if isinstance(val, collections.abc.Mapping):
        return {
            k: v
            for subkey, subval in val.items()
            for k, v in flatten_tree(subval, sep=sep, prefix=f"{prefix}{sep}{subkey}" if prefix else subkey).items()
        }
    elif isinstance(val, tuple) or isinstance(val, list):
        return {
            k: v
            for index, subval in enumerate(val)
            for k, v in flatten_tree(subval, sep=sep, prefix=f"{prefix}{sep}[{index}]" if prefix else f"[{index}]").items()
        }
    elif prefix:
        return {prefix: val}
    else:
        return val

# conversions

def hex2tuple(data: str) -> tuple[int]:
    data = data.removeprefix("#")
    return (*(
        int(data[i:i+2], 16)
        for i in range(0, len(data), 2)
    ),)


# repr shims

class CustomRepr:
    def __init__(self, repr_str: str):
        self.repr_str = repr_str
    def __str__(self):
        return self.repr_str
    def __repr__(self):
        return self.repr_str


# Meta Params Module proxy

class MetaModuleProxy:
    def __init__(self, module, params):
        self._module = module
        self._params = params

    def __getattr__(self, name):
        params = super().__getattribute__("_params")
        if name in params:
            return params[name]
        else:
            return getattr(super().__getattribute__("_module"), name)

    def __setattr__(self, name, value):
        if name not in ("_params", "_module"):
            super().__getattribute__("_params")[name] = value
        else:
            super().__setattr__(name, value)
