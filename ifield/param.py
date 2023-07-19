from .utils.helpers import compose, elementwise_max
from datetime import datetime
from torch import nn
from typing import Any, Literal, Iterable, Union, Callable, Optional
import inspect
import jinja2
import json
import os
import random
import re
import shlex
import string
import sys
import time
import typing
import warnings
import yaml

_UNDEFINED = " I AM UNDEFINED "

def _yaml_encode_value(val) -> str:
    if isinstance(val, tuple):
        val = list(val)
    elif isinstance(val, set):
        val = list(val)
    if isinstance(val, list):
        return json.dumps(val)
    elif isinstance(val, dict):
        return json.dumps(val)
    else:
        return yaml.dump(val).removesuffix("\n...\n").rstrip("\n")

def _raise(val: Union[Exception, str]):
    if isinstance(val, str):
        val = jinja2.TemplateError(val)
    raise val

def make_jinja_globals(*, enable_require_defined: bool) -> dict:
    import builtins
    import functools
    import itertools
    import operator
    import json

    def require_defined(name, value, *defaults, failed: bool = False, strict: bool=False, exchaustive=False):
        if not defaults:
            raise ValueError("`require_defined` requires at least one valid value provided")
        if jinja2.is_undefined(value):
            assert value._undefined_name == name, \
                f"Name mismatch: {value._undefined_name=}, {name=}"
        if failed or jinja2.is_undefined(value):
            if enable_require_defined or strict:
                raise ValueError(
                    f"Required variable {name!r} "
                    f"is {'incorrect' if failed else 'undefined'}! "
                    f"Try providing:\n" + "\n".join(
                        f"-O{shlex.quote(name)}={shlex.quote(str(default))}"
                        for default in defaults
                    )
                )
            else:
                warnings.warn(
                    f"Required variable {name!r} "
                    f"is {'incorrect' if failed else 'undefined'}! "
                    f"Try providing:\n" + "\n".join(
                        f"-O{shlex.quote(name)}={shlex.quote(str(default))}"
                        for default in defaults
                    )
                )
        if exchaustive and not jinja2.is_undefined(value) and value not in defaults:
            raise ValueError(
                f"Variable {name!r} not in list of allowed values: {defaults!r}"
            )

    def gen_run_uid(n: int, _choice = random.Random(time.time_ns()).choice):
        """
        generates a UID for the experiment run, nice for regexes, grepping and timekeeping.
        """
        # we have _choice, since most likely, pl.seed_everything has been run by this point
        # we store it as a default parameter to reuse it, on the off-chance of two calls to this function being run withion the same ns
        code = ''.join(_choice(string.ascii_lowercase) for _ in range(n))
        return f"{datetime.now():%Y-%m-%d-%H%M}-{code}"
        return f"{datetime.now():%Y%m%d-%H%M}-{code}"

    def cartesian_hparams(_map=None, **kw: dict[str, list]) -> Iterable[jinja2.utils.Namespace]:
        "Use this to bypass the common error 'SyntaxError: too many statically nested blocks'"
        if isinstance(_map, jinja2.utils.Namespace):
            kw = _map._Namespace__attrs | kw
        elif isinstance(_map, dict):
            kw = _map._Namespace__attrs | kw
        keys, vals = zip(*kw.items())
        for i in itertools.product(*vals):
            yield jinja2.utils.Namespace(zip(keys, i))

    def ablation_hparams(_map=None, *, caartesian_keys: list[str] = None, **kw: dict[str, list]) -> Iterable[jinja2.utils.Namespace]:
        "Use this to bypass the common error 'SyntaxError: too many statically nested blocks'"
        if isinstance(_map, jinja2.utils.Namespace):
            kw = _map._Namespace__attrs | kw
        elif isinstance(_map, dict):
            kw = _map._Namespace__attrs | kw
        keys = list(kw.keys())

        caartesian_keys = [k for k in keys if k in caartesian_keys] if caartesian_keys else []
        ablation_keys   = [k for k in keys if k not in caartesian_keys]
        caartesian_vals = list(map(kw.__getitem__, caartesian_keys))
        ablation_vals   = list(map(kw.__getitem__, ablation_keys))

        for base_vals in itertools.product(*caartesian_vals):
            base = list(itertools.chain(zip(caartesian_keys, base_vals), zip(ablation_keys, [i[0] for i in ablation_vals])))
            yield jinja2.utils.Namespace(base)
            for ablation_key, ablation_val in zip(ablation_keys, ablation_vals):
                for val in ablation_val[1:]:
                    yield jinja2.utils.Namespace(base, **{ablation_key: val}) # ablation variation

    return {
        **locals(),
        **vars(builtins),
        "argv": sys.argv,
        "raise": _raise,
    }

def make_jinja_env(globals = make_jinja_globals(enable_require_defined=True), allow_undef=False) -> jinja2.Environment:
    env = jinja2.Environment(
    	loader        = jinja2.FileSystemLoader([os.getcwd(), "/"], followlinks=True),
    	autoescape    = False,
        trim_blocks   = True,
        lstrip_blocks = True,
        undefined     = jinja2.Undefined if allow_undef else jinja2.StrictUndefined,
        extensions    = [
            "jinja2.ext.do", # statements with side-effects
            "jinja2.ext.loopcontrols", # break and continue
        ],
    )
    env.globals.update(globals)
    env.filters.update({
        "defined": lambda x: _raise(f"{x._undefined_name!r} is not defined!") if jinja2.is_undefined(x) else x,
        "repr":    repr,
        "to_json": json.dumps,
        "bool":    lambda x: json.dumps(bool(x)),
        "int":     lambda x: json.dumps(int(x)),
        "float":   lambda x: json.dumps(float(x)),
        "str":     lambda x: json.dumps(str(x)),
    })
    return env

def list_func_params(func: callable, exclude_list: set[str], defaults: dict={}) -> Iterable[tuple[str, Any, str]]:
    signature = inspect.signature(func)
    for i, (k, v) in enumerate(signature.parameters.items()):
        if not i and k in {"self", "cls"}:
            continue
        if k in exclude_list:
            continue
        if k.startswith("_"):
            continue
        if v.kind is v.VAR_POSITIONAL or v.kind is v.VAR_KEYWORD:
            continue
        has_default      = not defaults.get(k, v.default) is v.empty
        has_annotation   = not v.annotation is v.empty
        allowed_literals = f"{{{', '.join(map(_yaml_encode_value, typing.get_args(v.annotation)))}}}" \
            if typing.get_origin(v.annotation) is Literal else None

        assert has_annotation, f"param {k!r} has no type annotation"
        yield (
            k,
            defaults.get(k, v.default) if has_default else _UNDEFINED,
            f"in {allowed_literals}" if allowed_literals else typing._type_repr(v.annotation),
        )

@compose("\n".join)
def make_jinja_template(
        network_cls: nn.Module,
        *,
        exclude_list: set[str] = set(),
        defaults: dict[str, Any]={},
        top_level: bool = True,
        commented: bool = False,
        name=None,
        comment: Optional[str] = None,
        special_encoders: dict[str, Callable[[Any], str]]={},
        ) -> str:
    c = "#" if commented else ""
    if name is None:
        name = network_cls.__name__

    if comment is not None:
        if "\n" in comment:
            raise ValueError("newline in jinja template comment is not allowed")

    hparams = [*list_func_params(network_cls, exclude_list, defaults=defaults)]
    if not hparams:
        if top_level:
            yield f"{name}:"
        else:
            yield f"  # {name}:"
        return


    encoded_hparams = [
        (key, _yaml_encode_value(value) if value is not _UNDEFINED else "", comment)
        if key not in special_encoders else
        (key, special_encoders[key](value) if value is not _UNDEFINED else "", comment)
        for key, value, comment in hparams
    ]

    ml_key, ml_value = elementwise_max(
        (
            len(key),
            len(value),
        )
        for key, value, comment in encoded_hparams
    )

    if top_level:
        yield f"{name}:"     if not comment else f"{name}: # {comment}"
    else:
        yield f"  # {name}:" if not comment else f"  # {name}: # {comment}"

    for key, value, comment in encoded_hparams:
        if key in exclude_list:
            continue
        pad_key     = ml_key   - len(key)
        pad_value   = ml_value - len(value)

        yield f"  {c}{key}{' '*pad_key} : {value}{' '*pad_value} # {comment}"

    yield ""

# helpers:

def squash_newlines(data: str) -> str:
    return re.sub(r'\n\n\n+', '\n\n', data)
