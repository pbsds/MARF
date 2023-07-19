from . import logging, param
from .utils import helpers
from .utils.helpers import camel_to_snake_case
from argparse import ArgumentParser, _SubParsersAction, Namespace
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from munch import Munch, munchify
from pathlib import Path
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from serve_me_once import serve_once_in_background, gen_random_port
from torch import nn
from tqdm import tqdm
from typing import Optional, Callable, TypeVar, Union, Any
import argparse, collections, copy
import inspect, io, os, platform, psutil, pygments, pygments.lexers, pygments.formatters
import pytorch_lightning as pl, re, rich, rich.pretty, shlex, shutil, string, subprocess, sys, textwrap
import traceback, time, torch, torchviz, urllib.parse, warnings, webbrowser, yaml


CONSOLE = rich.console.Console(width=None if os.isatty(1) else 140)
torch.set_printoptions(threshold=200)

# https://gist.github.com/pypt/94d747fe5180851196eb#gistcomment-3595282
#class UniqueKeyYAMLLoader(yaml.SafeLoader):
class UniqueKeyYAMLLoader(yaml.Loader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise KeyError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)

# load scientific notation correctly as floats and not as strings
# basically, support for the to_json filter in jinja
# https://stackoverflow.com/a/30462009
# https://github.com/yaml/pyyaml/issues/173
UniqueKeyYAMLLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class IgnorantActionsContainer(argparse._ActionsContainer):
    """
    Ignores conflicts with
    Must be enabled with ArgumentParser(conflict_handler="ignore")
    """
    # https://stackoverflow.com/a/71782808
    def _handle_conflict_ignore(self, action, conflicting_actions):
        pass
argparse.ArgumentParser.__bases__ = (argparse._AttributeHolder, IgnorantActionsContainer)
argparse._ArgumentGroup.__bases__ = (IgnorantActionsContainer,)

@contextmanager
def ignore_action_container_conflicts(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]):
    old = parser.conflict_handler
    parser.conflict_handler = "ignore"
    yield
    parser.conflict_handler = old

def _print_with_syntax_highlighting(language, string, indent=""):
    if os.isatty(1):
        string = pygments.highlight(string,
            lexer     = pygments.lexers.get_lexer_by_name(language),
            formatter = pygments.formatters.Terminal256Formatter(style="monokai"),
        )
    if indent:
        string = textwrap.indent(string, indent)
    print(string)

def print_column_dict(data: dict, n_columns: int = 2, prefix: str="  "):
    small = {k: v for k, v in data.items() if not isinstance(v, dict) and len(repr(v)) <= 40}
    wide  = {k: v for k, v in data.items() if not isinstance(v, dict) and len(repr(v))  > 40}
    dicts = {k: v for k, v in data.items() if     isinstance(v, dict)}
    kw = dict(
        crop      = False,
        overflow  = "ignore",
    )
    if small:
        CONSOLE.print(helpers.columnize_dict(small, prefix=prefix, n_columns=n_columns, sep="  "), **kw)
    key_len = max(map(len, map(repr, wide.keys()))) if wide else 0
    for key, val in wide.items():
        CONSOLE.print(f"{prefix}{repr(key).ljust(key_len)} : {val!r},", **kw)
    for key, val in dicts.items():
        CONSOLE.print(f"{prefix}{key!r}: {{", **kw)
        print_column_dict(val, n_columns=n_columns, prefix=prefix+"  ")
        CONSOLE.print(f"{prefix}}},", **kw)


M  = TypeVar("M",  bound=nn.Module)
DM = TypeVar("DM", bound=pl.LightningDataModule)
FitHook = Callable[[Namespace, Munch, M, pl.Trainer, DM, logging.Logger], None]

class CliInterface:
    trainer_defaults: dict

    def __init__(self, *, module_cls: type[M], workdir: Path, datamodule_cls: Union[list[type[DM]], type[DM], None] = None, experiment_name_prefix = "experiment"):
        self.module_cls     = module_cls
        self.datamodule_cls = [datamodule_cls] if not isinstance(datamodule_cls, list) and datamodule_cls is not None else datamodule_cls
        self.workdir        = workdir
        self.experiment_name_prefix = experiment_name_prefix

        self.trainer_defaults = dict(
            enable_model_summary = False,
        )

        self.pre_fit_handlers:  list[FitHook] = []
        self.post_fit_handlers: list[FitHook] = []

        self._registered_actions : dict[str, tuple[Callable[M, None], list, dict, Optional[callable]]]  = {}
        self._included_in_config_template : dict[str, tuple[callable, dict]] = {}

        self.register_action(_func=self.repr, help="Print str(module).", args=[])
        self.register_action(_func=self.yaml, help="Print evaluated config.", args=[])
        self.register_action(_func=self.hparams, help="Print hparams, like during training.", args=[])
        self.register_action(_func=self.dot, help="Print graphviz graph of computation graph.", args=[
            ("-e", "--eval",   dict(action="store_true")),
            ("-f", "--filter", dict(action="store_true")),
        ])
        self.register_action(_func=self.jit, help="Print a TorchScript graph of the model", args=[])
        self.register_action(_func=self.trace, help="Dump a TorchScript trace of the model.", args=[
            ("output_file", dict(type=Path,
                help="Path to write the .pt file. Use \"-\" to instead open the trace in Netron.app")),
        ])
        self.register_action(_func=self.onnx, help="Dump a ONNX trace of the model.", args=[
            ("output_file", dict(type=Path,
                help="Path to write the .onnx file. Use \"-\" to instead open the onnx in Netron.app")),
        ])

        if self.datamodule_cls:
            names = [i.__name__ for i in self.datamodule_cls]
            names_snake = [datamodule_name_to_snake_case(i) for i in names]
            assert len(names) == len(set(names)),\
                f"Datamodule names are not unique: {names!r}"
            assert len(names) == len(set(names_snake)),\
                f"Datamodule snake-names are not unique: {names_snake!r}"

            self.register_action(_func=self.test_dataloader,
                help="Benchmark the speed of the dataloader",
                args=[
                    ("datamodule", dict(type=str, default=None, nargs='?', choices=names_snake,
                        help="Which dataloader to test. Defaults to the first one found in config.")),
                    ("--limit-cores", dict(type=int, default=None,
                        help="Limits the cpu affinity to N cores. Perfect to simulate a SLURM environ.")),
                    ("--profile", dict(type=Path, default=None,
                        help="Profile using cProfile, marshaling the result to a .prof or .log file.")),
                    ("-n", "--n-rounds", dict(type=int, default=3,
                        help="Number of times to read the dataloader.")),
                ],
                conflict_handler = "ignore" if len(self.datamodule_cls) > 1 else "error",
                add_argparse_args=[i.add_argparse_args for i in self.datamodule_cls],
            )


    # decorator
    def register_pre_training_callback(self, func: FitHook):
        self.pre_fit_handlers.append(func)
        return func

    # decorator
    def register_post_training_callback(self, func: FitHook):
        self.post_fit_handlers.append(func)
        return func

    # decorator
    def register_action(self, *,
            help              : str,
            args              : list[tuple[Any, ..., dict]]                          = [],
            _func             : Optional[Callable[[Namespace, Munch, M], None]]      = None,
            add_argparse_args : Union[list[Callable[[ArgumentParser], ArgumentParser]], Callable[[ArgumentParser], ArgumentParser], None] = None,
            **kw,
            ):
        def wrapper(action: Callable[[Namespace, Munch, M], None]):
            cli_name = action.__name__.lower().replace("_", "-")
            self._registered_actions[cli_name] = (
                action,
                args,
                kw | {"help": help},
                add_argparse_args,
            )
            return action
        if _func is not None: # shortcut
            return wrapper(_func)
        else:
            return wrapper

    def make_parser(self,
            parser      : ArgumentParser    = None,
            subparsers  : _SubParsersAction = None,
            add_trainer : bool              = False,
            ) -> tuple[ArgumentParser, _SubParsersAction, _SubParsersAction]:
        if parser is None:
            parser = ArgumentParser()
        if subparsers is None:
            subparsers = parser.add_subparsers(dest="mode", required=True)

        parser.add_argument("-pm", "--post-mortem", action="store_true",
            help="Start a debugger if a uncaught exception is thrown.")

        # Template generation and exploration
        parser_template = subparsers.add_parser("template",
            help="Generate or evaluate a config template")
        if 1: # fold me
            parser_mode_mutex = parser_template.add_mutually_exclusive_group()#(required=True)
            parser_mode_mutex.add_argument("-e", "--evaluate", metavar="TEMPLATE", type=Path,
                help="Read jinja2 yaml config template file, then evaluate and print it.")
            parser_mode_mutex.add_argument("-p", "--parse", metavar="TEMPLATE", type=Path,
                help="Read jinja2 yaml config template file, then evaluate, parse and print it.")

            def pair(data: str) -> tuple[str, str]:
                key, sep, value = data.partition("=")
                if not sep:
                    if key in os.environ:
                        value = os.environ[key]
                    else:
                        raise ValueError(f"the variable {key!r} was not given any value, and none was found in the environment.")
                elif "$" in value:
                    value = string.Template(value).substitute(os.environ)
                return (key, value)
            parser_template.add_argument("-O", dest="jinja2_variables", action="append", type=pair,
                help="Variable available as string in the jinja2. (a=b). b will be expanded as an"
                    " env var if prefixed with $, or set equal to the env var a if =b is omitted.")

            parser_template.add_argument("-s", "--strict", action="store_true",
                help="Enable {% do require_defined(\"var\",var) %}".replace("%", "%%"))
            parser_template.add_argument("-d", "--defined-only", action="store_true",
                help="Disallow any use of undefined variables")


        # Load a module
        parser_module = subparsers.add_parser("module", aliases=["model"],
            help="Load a config template, evaluate it and use the resulting module")
        if 1: # fold me
            parser_module.add_argument("module_file", type=Path,
                help="Jinja2 yaml config template or pytorch-lightning .ckpt file.")
            parser_module.add_argument("-O", dest="jinja2_variables", action="append", type=pair,
                help="Variable available as string in the jinja2. (a=b). b will be expanded as an"
                    " env var if prefixed with $, or set equal to the env var a if =b is omitted.")
            parser_module.add_argument("--last", action="store_true",
                help="if multiple ckpt match, prefer the last one")
            parser_module.add_argument("--best", action="store_true",
                help="if multiple ckpt match, prefer the best one")

            parser_module.add_argument("--add-shape-prehook", action="store_true",
                help="Add a forward hook which prints the tensor shapes of all inputs, but not the outputs.")
            parser_module.add_argument("--add-shape-hook", action="store_true",
                help="Add a forward hook which prints the tensor shapes of all inputs AND outputs.")
            parser_module.add_argument("--add-oob-hook", action="store_true",
                help="Add a forward hook checking for INF and NaN values in inputs or outputs.")
            parser_module.add_argument("--add-oob-hook-input", action="store_true",
                help="Add a forward hook checking for INF and NaN values in inputs.")
            parser_module.add_argument("--add-oob-hook-output", action="store_true",
                help="Add a forward hook checking for INF and NaN values in outputs.")


        module_actions_subparser = parser_module.add_subparsers(dest="action", required=True)

        # add pluggables
        for name, (action, args, kw, add_argparse_args) in self._registered_actions.items():
            action_parser = module_actions_subparser.add_parser(name, **kw)
            if add_argparse_args is not None and add_argparse_args:
                for func in add_argparse_args if isinstance(add_argparse_args, list) else [add_argparse_args]:
                    action_parser = func(action_parser)
            for *a, kw in args:
                action_parser.add_argument(*a, **kw)

        # Module: train or test
        if self.datamodule_cls:
            parser_trainer = module_actions_subparser.add_parser("fit", aliases=["test"],
                help="Train/fit or evaluate the module with train/val or test data.")

            # pl.Trainer
            parser_trainer = pl.Trainer.add_argparse_args(parser_trainer)

            # datamodule
            parser_trainer.add_argument("datamodule", type=str, default=None, nargs='?',
                choices=[datamodule_name_to_snake_case(i) for i in self.datamodule_cls],
                help="Which dataloader to test. Defaults to the first one found in config.")
            if len(self.datamodule_cls) > 1:
                # check that none of the datamodules conflict with trainer or module
                for datamodule_cls in self.datamodule_cls:
                    datamodule_cls.add_argparse_args(copy.deepcopy(parser_trainer)) # will raise on conflict
            # Merge the datamodule options, the above sanity check makes it "okay"
            with ignore_action_container_conflicts(parser_trainer):
                for datamodule_cls in self.datamodule_cls:
                    parser_trainer = datamodule_cls.add_argparse_args(parser_trainer)

            # defaults and jinja template
            self._included_in_config_template.clear()
            remove_options_from_parser(parser_trainer, "--logger")
            parser_trainer.set_defaults(**self.trainer_defaults)
            self.add_to_jinja_template("trainer", pl.Trainer, defaults=self.trainer_defaults, exclude_list={
                # not yaml friendly, already covered anyway:
                "logger",
                "plugins",
                "callbacks",
                # deprecated or covered by callbacks:
                "stochastic_weight_avg",
                "enable_model_summary",
                "track_grad_norm",
                "log_gpu_memory",
            })
            for datamodule_cls in self.datamodule_cls:
                self.add_to_jinja_template(datamodule_cls.__name__, datamodule_cls,
                    comment=f"select with {datamodule_name_to_snake_case(datamodule_cls)!r}")#, commented=False)
            self.add_to_jinja_template("logging", logging, save_dir = "logdir", commented=False)

        return parser, subparsers, module_actions_subparser

    def add_to_jinja_template(self, name: str, func: callable, **kwargs):
        """
        Basically a call to `make_jinja_template`.
        Will ensure the keys are present in the output from `from_argparse_args`.
        """
        self._included_in_config_template[name] = (func, dict(commented=True) | kwargs)

    def make_jinja_template(self) -> str:
        return "\n".join([
            f'#!/usr/bin/env -S python {sys.argv[0]} module',
            r'{% do require_defined("select", select, 0, "$SLURM_ARRAY_TASK_ID") %}{# requires jinja2.ext.do #}',
            r"{% set counter = itertools.count(start=0, step=1) %}",
            r"",
            r"{% set hp_matrix = namespace() %}{# hyper parameter matrix #}",
            r"{% set hp_matrix.my_hparam = [0] %}{##}",
            r"",
            r"{% for hp in cartesian_hparams(hp_matrix) %}{##}",
            r"{#% for hp in ablation_hparams(hp_matrix, caartesian_keys=[]) %}{##}",
            r"",
            r"{% set index = next(counter) %}",
            r"{% if select is not defined and index > 0 %}---{% endif %}",
            r"{% if select is not defined or int(select) == index %}",
            r"",
            *[
                func.make_jinja_template(name=name, **kwargs)
                if hasattr(func, "make_jinja_template") else
                param.make_jinja_template(func, name=name, **kwargs)
                for name, (func, kwargs) in self._included_in_config_template.items()
            ],
            r"{% autoescape false %}",
            r'{% do require_defined("experiment_name", experiment_name, "test", strict=true) %}',
            f"experiment_name: { self.experiment_name_prefix }-{{{{ experiment_name }}}}",
            r'{#--#}-{{ hp.my_hparam }}',
            r'{#--#}-{{ gen_run_uid(4) }} # select with -Oselect={{ index }}',
            r"{% endautoescape %}",
            self.module_cls.make_jinja_template(),
            r"{% endif %}{# -Oselect #}",
            r"",
            r"{% endfor %}",
            r"",
            r"{% set index = next(counter) %}",
            r"# number of possible 'select': {{ index }}, from 0 to {{ index-1 }}",
            r"# local: for select in {0..{{ index-1 }}}; do python ... -Oselect=$select ... ; done",
            r"# local: for select in {0..{{ index-1 }}}; do python -O {{ argv[0] }} model marf.yaml.j2 -Oselect=$select -Oexperiment_name='{{ experiment_name }}' fit --accelerator gpu ; done",
            r"# slurm: sbatch --array=0-{{ index-1 }} runcommand.slurm python ... -Oselect=\$SLURM_ARRAY_TASK_ID ...",
            r"# slurm: sbatch --array=0-{{ index-1 }} runcommand.slurm python -O {{ argv[0] }} model this-file.yaml.j2 -Oselect=\$SLURM_ARRAY_TASK_ID -Oexperiment_name='{{ experiment_name }}' fit --accelerator gpu --devices -1 --strategy ddp"
        ])

    def run(self, args=None, args_hook: Optional[Callable[[ArgumentParser, _SubParsersAction, _SubParsersAction], None]] = None):
        parser, mode_subparser, action_subparser = self.make_parser()
        if args_hook is not None:
            args_hook(parser, mode_subparser, action_subparser)
        args = parser.parse_args(args) # may exit
        if os.isatty(0) and args.post_mortem:
            warnings.warn("post-mortem debugging is enabled without any TTY attached. Will be ignored.")
        if args.post_mortem and os.isatty(0):
            try:
                self.handle_args(args)
            except Exception:
                # print exception
                sys.excepthook(*sys.exc_info())
                # debug
                *debug_module, debug_func = os.environ.get("PYTHONBREAKPOINT", "pdb.set_trace").split(".")
                __import__(".".join(debug_module)).post_mortem()
                exit(1)
        else:
            self.handle_args(args)

    def handle_args(self, args: Namespace):
        """
        May call exit()
        """
        if args.mode == "template":

            if args.evaluate or args.parse:
                template_file = args.evaluate or args.parse
                env = param.make_jinja_env(globals=param.make_jinja_globals(enable_require_defined=args.strict), allow_undef=not args.defined_only)
                if str(template_file) == "-":
                    template = env.from_string(sys.stdin.read(), globals=dict(args.jinja2_variables or []))
                else:
                    template = env.get_template(str(template_file.absolute()), globals=dict(args.jinja2_variables or []))
                config_yaml = param.squash_newlines(template.render())#.lstrip("\n").rstrip()
                if args.evaluate:
                    _print_with_syntax_highlighting("yaml+jinja", config_yaml)
                else:
                    config = yaml.load(config_yaml, UniqueKeyYAMLLoader)
                    CONSOLE.print(config)

            else:
                _print_with_syntax_highlighting("yaml+jinja", self.make_jinja_template())

        elif args.mode in ("module", "model"):

            module: nn.Module

            if not args.module_file.is_file():
                matches = [*Path("logdir/tensorboard").rglob(f"*-{args.module_file}/checkpoints/*.ckpt")]
                if len(matches) == 1:
                    args.module_file, = matches
                elif len(matches) > 1:
                    if (args.last or args.best) and len(set(match.parent.parent.name for match in matches)) == 1:
                        if args.last:
                            args.module_file, = (match for match in matches if match.name == "last.ckpt")
                        elif args.best:
                            args.module_file, = (match for match in matches if match.name.startswith("epoch="))
                        else:
                            assert False
                    else:
                        raise ValueError("uid matches multiple paths:\n"+"\n".join(map(str, matches)))
                else:
                    raise ValueError("path does not exist, and is not a uid")

            # load module from cli args
            if args.module_file.suffix == ".ckpt": # from checkpoint
                # load from checkpoint
                rich.print(f"Loading module from {str(args.module_file)!r}...", file=sys.stderr)
                module = self.module_cls.load_from_checkpoint(args.module_file)

                if (args.module_file.parent.parent / "hparams.yaml").is_file():
                    with (args.module_file.parent.parent / "hparams.yaml").open() as f:
                        config_yaml = yaml.load(f.read(), UniqueKeyYAMLLoader)["_pickled_cli_args"]["_raw_yaml"]
                else:
                    with (args.module_file.parent.parent / "config.yaml").open() as f:
                        config_yaml = f.read()

                config = munchify(yaml.load(config_yaml, UniqueKeyYAMLLoader) | {"_raw_yaml": config_yaml})

            else: # from yaml

                # read, evaluate and parse config
                if args.module_file.suffix == ".j2" or str(args.module_file) == "-":
                    env = param.make_jinja_env()
                    if str(args.module_file) == "-":
                        template = env.from_string(sys.stdin.read(), globals=dict(args.jinja2_variables or []))
                    else: # jinja+yaml file
                        template = env.get_template(str(args.module_file.absolute()), globals=dict(args.jinja2_variables or []))
                    config_yaml = param.squash_newlines(template.render()).lstrip("\n").rstrip()
                else: # yaml file (the git diffs in _pickled_cli_args may trigger jinja's escape sequences)
                    with args.module_file.open() as f:
                        config_yaml = f.read().lstrip("\n").rstrip()

                config = yaml.load(config_yaml, UniqueKeyYAMLLoader)

                if "_pickled_cli_args" in config: # hparams.yaml in tensorboard logdir
                    config_yaml = config["_pickled_cli_args"]["_raw_yaml"]
                    config = yaml.load(config_yaml, UniqueKeyYAMLLoader)

                from_checkpoint: Optional[Path] = None
                if (args.module_file.parent / "checkpoints").glob("*.ckpt"):
                    checkpoints_fnames = list((args.module_file.parent / "checkpoints").glob("*.ckpt"))
                    if len(checkpoints_fnames) == 1:
                        from_checkpoint = checkpoints_fnames[0]
                    elif args.last:
                        from_checkpoint, = (i for i in checkpoints_fnames if i.name == "last.ckpt")
                    elif args.best:
                        from_checkpoint, = (i for i in checkpoints_fnames if i.name.startswith("epoch="))
                    elif len(checkpoints_fnames) > 1:
                        rich.print(f"[yellow]WARNING:[/] {str(args.module_file.parent / 'checkpoints')!r} contains more than one checkpoint, unable to automatically load one.", file=sys.stderr)

                config = munchify(config | {"_raw_yaml": config_yaml})

                # Ensure date and uid to experiment name, allowing for reruns and organization
                assert config.experiment_name
                assert re.match(r'^.*-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{4}-[a-z]{4}$', config.experiment_name),\
                    config.experiment_name

                # init the module
                if from_checkpoint:
                    rich.print(f"Loading module from {str(from_checkpoint)!r}...", file=sys.stderr)
                    module = self.module_cls.load_from_checkpoint(from_checkpoint)
                else:
                    module = self.module_cls(**{k:v for k, v in config[self.module_cls.__name__].items() if k != "_extra"})

            # optional debugging forward hooks

            if args.add_shape_hook or args.add_shape_prehook:
                def shape_forward_hook(is_prehook: bool, name: str, module: nn.Module, input, output=None):
                    def tensor_to_shape(val):
                        if isinstance(val, torch.Tensor):
                            return tuple(val.shape)
                        elif isinstance(val, (str, float, int)) or val is None:
                            return 1
                        else:
                            assert 0, (val, name)
                    with torch.no_grad():
                        rich.print(
                            f"{name}.forward({helpers.map_tree(tensor_to_shape, input)})"
                            if is_prehook else
                            f"{name}.forward({helpers.map_tree(tensor_to_shape, input)})"
                            f" -> {helpers.map_tree(tensor_to_shape, output)}"
                            , file=sys.stderr)

                for submodule_name, submodule in module.named_modules():
                    if submodule_name:
                        submodule_name = f"{module.__class__.__qualname__}.{submodule_name}"
                    else:
                        submodule_name = f"{module.__class__.__qualname__}"
                    if args.add_shape_prehook:
                        submodule.register_forward_pre_hook(partial(shape_forward_hook, True, submodule_name))
                    if args.add_shape_hook:
                        submodule.register_forward_hook(partial(shape_forward_hook, False, submodule_name))

            if args.add_oob_hook or args.add_oob_hook_input or args.add_oob_hook_output:
                def oob_forward_hook(name: str, module: nn.Module, input, output):
                    def raise_if_oob(key, val):
                        if isinstance(val, collections.abc.Mapping):
                            for k, subval in val.items():
                                raise_if_oob(f"{key}[{k!r}]", subval)
                        elif isinstance(val, (tuple, list)):
                            for i, subval in enumerate(val):
                                raise_if_oob(f"{key}[{i}]", subval)
                        elif isinstance(val, torch.Tensor):
                            assert not torch.isinf(val).any(), \
                                f"INFs found in {key}"
                            assert not val.isnan().any(), \
                                f"NaNs found in {key}"
                        elif isinstance(val, (str, float, int)):
                            pass
                        elif val is None:
                            warnings.warn(f"None found in {key}")
                        else:
                            assert False, val
                    with torch.no_grad():
                        if args.add_oob_hook or args.add_oob_hook_input:
                            raise_if_oob(f"{name}.forward input",  input)
                        if args.add_oob_hook or args.add_oob_hook_output:
                            raise_if_oob(f"{name}.forward output", output)

                for submodule_name, submodule in module.named_modules():
                    submodule.register_forward_hook(partial(oob_forward_hook,
                        f"{module.__class__.__qualname__}.{submodule_name}"
                        if submodule_name else
                        f"{module.__class__.__qualname__}"
                    ))

            # Ensure all the top-level config keys are there
            for key in self._included_in_config_template.keys():
                if key in (i.__name__ for i in self.datamodule_cls):
                    continue
                if key not in config or config[key] is None:
                    config[key] = {}

            # Run registered action
            if args.action in self._registered_actions:
                action, *_ = self._registered_actions[args.action]
                action(args, config, module)
            elif args.action in ("fit", "test") and self.datamodule_cls is not None:
                self.fit(args, config, module)
            else:
                raise ValueError(f"{args.mode=}, {args.action=}")

        else:
            raise ValueError(f"{args.mode=}")

    def get_datamodule_cls_from_config(self, args: Namespace, config: Munch) -> DM:
        assert self.datamodule_cls
        cli = getattr(args, "datamodule", None)
        datamodule_cls: pl.LightningDataModule
        if cli is not None:
            datamodule_cls, = (i for i in self.datamodule_cls if datamodule_name_to_snake_case(i) == cli)
        else:
            datamodules = {
                cls.__name__: cls
                for cls in self.datamodule_cls
            }
            for key in config.keys():
                if key in datamodules:
                    datamodule_cls = datamodules[key]
                    break
            else:
                datamodule_cls = self.datamodule_cls[0]
                warnings.warn(f"None of the following datamodules were found in config: {set(datamodules.keys())!r}. {datamodule_cls.__name__!r} was chosen as the default.")

        return datamodule_cls

    def init_datamodule_cls_from_config(self, args: Namespace, config: Munch) -> DM:
        datamodule_cls = self.get_datamodule_cls_from_config(args, config)
        return datamodule_cls.from_argparse_args(args, **(config.get(datamodule_cls.__name__) or {}))


    # Module actions

    def repr(self, args: Namespace, config: Munch, module: M):
        rich.print(module)

    def yaml(self, args: Namespace, config: Munch, module: M):
        _print_with_syntax_highlighting("yaml+jinja", config["_raw_yaml"])

    def dot(self, args: Namespace, config: Munch, module: M):
        module.train(not args.eval)
        assert not args.filter, "not implemented! pipe it through examples/scripts/filter_dot.py in the meanwhile"

        example_input_array = module.example_input_array
        assert example_input_array is not None, f"{module.__class__.__qualname__}.example_input_array=None"
        assert isinstance(example_input_array, (tuple, dict, torch.Tensor)), type(example_input_array)

        def set_requires_grad(val):
            if isinstance(val, torch.Tensor):
                val.requires_grad = True
            return val

        with torch.enable_grad():
            outputs = module(*helpers.map_tree(set_requires_grad, example_input_array))

        dot = torchviz.make_dot(outputs, params=dict(module.named_parameters()), show_attrs=False, show_saved=False)
        _print_with_syntax_highlighting("dot", str(dot))

    def jit(self, args: Namespace, config: Munch, module: M):
        example_input_array = module.example_input_array
        assert example_input_array is not None, f"{module.__class__.__qualname__}.example_input_array=None"
        assert isinstance(example_input_array, (tuple, dict, torch.Tensor)), type(example_input_array)
        trace = torch.jit.trace_module(module, {"forward": example_input_array})
        _print_with_syntax_highlighting("python", str(trace.inlined_graph))

    def trace(self, args: Namespace, config: Munch, module: M):
        if isinstance(module, pl.LightningModule):
            trace = module.to_torchscript(method="trace")
        else:
            example_input_array = module.example_input_array
            assert example_input_array is not None, f"{module.__class__.__qualname__}.example_input_array is None"
            assert isinstance(module, torch.Module)
            trace = torch.jit.trace_module(module, {"forward": example_input_array})

        use_netron = str(args.output_file) == "-"
        trace_f = io.BytesIO() if use_netron else args.output_file

        torch.jit.save(trace, trace_f)

        if use_netron:
            open_in_netron(f"{self.module_cls.__name__}.pt", trace_f.getvalue())

    def onnx(self, args: Namespace, config: Munch, module: M):
        example_input_array = module.example_input_array
        assert example_input_array is not None, f"{module.__class__.__qualname__}.example_input_array=None"
        assert isinstance(example_input_array, (tuple, dict, torch.Tensor)), type(example_input_array)

        use_netron = str(args.output_file) == "-"
        onnx_f = io.BytesIO() if use_netron else args.output_file

        torch.onnx.export(module,
            tuple(example_input_array),
            onnx_f,
            export_params       = True,
            opset_version       = 17,
            do_constant_folding = True,
            input_names         = ["input"],
            output_names        = ["output"],
            dynamic_axes        = {
                "input"  : {0 : "batch_size"},
                "output" : {0 : "batch_size"},
            },
        )

        if use_netron:
            open_in_netron(f"{self.module_cls.__name__}.onnx", onnx_f.getvalue())

    def hparams(self, args: Namespace, config: Munch, module: M):
        assert isinstance(module, self.module_cls)
        print(f"{self.module_cls.__qualname__} hparams:")
        print_column_dict(map_type_to_repr(module.hparams, nn.Module, lambda t: f"{t.__class__.__qualname__}"), 3)


    def fit(self, args: Namespace, config: Munch, module: M):
        is_rank_zero = pl.utilities.rank_zero_only.rank == 0

        metric_prefix = f"{module.__class__.__name__}.validation_step/"

        pl_callbacks = [
            pl.callbacks.LearningRateMonitor(log_momentum=True),
            pl.callbacks.EarlyStopping(monitor=metric_prefix+getattr(module, "metric_early_stop", "loss"), patience=200, check_on_train_epoch_end=False, verbose=True),
            pl.callbacks.ModelCheckpoint(monitor=metric_prefix+getattr(module, "metric_best_model", "loss"), mode="min", save_top_k=1, save_last=True),
            logging.ModelOutputMonitor(),
            logging.EpochTimeMonitor(),
            (pl.callbacks.RichModelSummary if os.isatty(1) else pl.callbacks.ModelSummary)(max_depth=30),
            logging.PsutilMonitor(),
        ]
        if os.isatty(1):
            pl_callbacks.append( pl.callbacks.RichProgressBar() )

        trainer: pl.Trainer
        logger     = logging.make_logger(config.experiment_name, config.trainer.get("default_root_dir", args.default_root_dir or self.workdir), **config.logging)
        trainer    = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=pl_callbacks, **config.trainer)

        datamodule = self.init_datamodule_cls_from_config(args, config)

        for f in self.pre_fit_handlers:
            print(f"pre-train hook {f.__name__!r}...")
            f(args, config, module, trainer, datamodule, logger)

        # print and log hparams/config
        if 1: # fold me
            if is_rank_zero:
                CONSOLE.print(f"Experiment name: {config.experiment_name!r}", soft_wrap=False, crop=False, no_wrap=False, overflow="ignore")

            # parser.args and sys.argv
            pickled_cli_args = dict(
                sys_argv    = sys.argv,
                parser_args = args.__dict__,
                config      = config.copy(),
                _raw_yaml   = config["_raw_yaml"],
            )
            del pickled_cli_args["config"]["_raw_yaml"]
            for k,v in pickled_cli_args["parser_args"].items():
                if isinstance(v, Path):
                    pickled_cli_args["parser_args"][k] = str(v)

            # trainer
            params_trainer  = inspect.signature(pl.Trainer.__init__).parameters
            trainer_hparams = vars(pl.Trainer.parse_argparser(args))
            trainer_hparams = { name: trainer_hparams[name] for name in params_trainer if name in trainer_hparams }
            if is_rank_zero:
                print("pl.Trainer hparams:")
                print_column_dict(trainer_hparams, 3)
            pickled_cli_args.update(trainer_hparams=trainer_hparams)

            # module
            assert isinstance(module, self.module_cls)
            if is_rank_zero:
                print(f"{self.module_cls.__qualname__} hparams:")
                print_column_dict(map_type_to_repr(module.hparams, nn.Module, lambda t: f"{t.__class__.__qualname__}"), 3)
            pickled_cli_args.update(module_hparams={
                k : v
                for k, v in module.hparams.items()
                if k != "_raw_yaml"
            })

            # module extra state, like autodecoder uids
            for submodule_name, submodule in module.named_modules():
                if not submodule_name:
                    submodule_name = module.__class__.__qualname__
                else:
                    submodule_name = module.__class__.__qualname__ + "." + submodule_name
                try:
                    state = submodule.get_extra_state()
                except RuntimeError:
                    continue
                if "extra_state" not in pickled_cli_args:
                    pickled_cli_args["extra_state"] = {}
                pickled_cli_args["extra_state"][submodule_name] = state

            # datamodule
            if self.datamodule_cls:
                assert datamodule is not None and any(isinstance(datamodule, i) for i in self.datamodule_cls), datamodule
                for datamodule_cls in self.datamodule_cls:
                    params_d = inspect.signature(datamodule_cls.__init__).parameters
                    assert {"self"} == set(params_trainer).intersection(params_d), \
                        f"trainer and datamodule has overlapping params: {set(params_trainer).intersection(params_d) - {'self'}}"

                if is_rank_zero:
                    print(f"{datamodule.__class__.__qualname__} hparams:")
                    print_column_dict(datamodule.hparams)
                pickled_cli_args.update(datamodule_hparams=dict(datamodule.hparams))

            # logger
            if logger is not None:
                print(f"{logger.__class__.__qualname__} hparams:")
                print_column_dict(config.logging)
                pickled_cli_args.update(logger_hparams = {"_class": logger.__class__.__name__} | config.logging)

            # host info
            def cmd(cmd: Union[str, list[str]]) -> str:
                if isinstance(cmd, str):
                    cmd = shlex.split(cmd)
                if shutil.which(cmd[0]):
                    try:
                        return subprocess.run(cmd,
                            capture_output=True,
                            check=True,
                            text=True,
                        ).stdout.strip()
                    except subprocess.CalledProcessError as e:
                        warnings.warn(f"{e.__class__.__name__}: {e}")
                        return f"{e.__class__.__name__}: {e}\n{e.output = }\n{e.stderr = }"
                else:
                    warnings.warn(f"command {cmd[0]!r} not found")
                    return f"*command {cmd[0]!r} not found*"

            pickled_cli_args.update(host = dict(
                platform = textwrap.dedent(f"""
                    {platform.architecture()          = }
                    {platform.java_ver()              = }
                    {platform.libc_ver()              = }
                    {platform.mac_ver()               = }
                    {platform.machine()               = }
                    {platform.node()                  = }
                    {platform.platform()              = }
                    {platform.processor()             = }
                    {platform.python_branch()         = }
                    {platform.python_build()          = }
                    {platform.python_compiler()       = }
                    {platform.python_implementation() = }
                    {platform.python_revision()       = }
                    {platform.python_version()        = }
                    {platform.release()               = }
                    {platform.system()                = }
                    {platform.uname()                 = }
                    {platform.version()               = }
                """.rstrip()).lstrip(),
                cuda = dict(
                    gpus      = [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ],
                    available = torch.cuda.is_available(),
                    version   = torch.version.cuda,
                ),
                hostname    = cmd("hostname --fqdn"),
                cwd         = os.getcwd(),
                date        = datetime.now().astimezone().isoformat(),
                date_utc    = datetime.utcnow().isoformat(),
                ifconfig    = cmd("ifconfig"),
                lspci       = cmd("lspci"),
                lsusb       = cmd("lsusb"),
                lsblk       = cmd("lsblk"),
                mount       = cmd("mount"),
                environ     = os.environ.copy(),
                vcs         = f"commit {cmd('git rev-parse HEAD')}\n{cmd('git status')}\n{cmd('git diff --stat --patch HEAD')}",
                venv_pip    = cmd("pip list --format=freeze"),
                venv_conda  = cmd("conda list"),
                venv_poetry = cmd("poetry show -t"),
                gpus        = [i.split(", ") for i in cmd("nvidia-smi --query-gpu=index,name,memory.total,driver_version,uuid --format=csv").splitlines()],
            ))

            if logger is not None:
                logging.log_config(logger, _pickled_cli_args=pickled_cli_args)

        warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)

        if __debug__ and is_rank_zero:
            warnings.warn("You're running python with assertions active. Enable optimizations with `python -O` for improved performance.")

        # train

        t_begin = datetime.now()
        if args.action == "fit":
            trainer.fit(module, datamodule)
        elif args.action == "test":
            trainer.test(module, datamodule)
        else:
            raise ValueError(f"{args.mode=}, {args.action=}")

        if not is_rank_zero:
            return

        t_end = datetime.now()
        print(f"Training time: {t_end - t_begin}")

        for f in self.post_fit_handlers:
            print(f"post-train hook {f.__name__!r}...")
            try:
                f(args, config, module, trainer, datamodule, logger)
            except Exception:
                traceback.print_exc()

        rich.print(f"Experiment name: {config.experiment_name!r}")
        rich.print(f"Best model path: {helpers.make_relative(trainer.checkpoint_callback.best_model_path).__str__()!r}")
        rich.print(f"Last model path: {helpers.make_relative(trainer.checkpoint_callback.last_model_path).__str__()!r}")

    def test_dataloader(self, args: Namespace, config: Munch, module: M):
        # limit CPU affinity
        if args.limit_cores is not None:
            # https://stackoverflow.com/a/40856471
            p = psutil.Process()
            assert len(p.cpu_affinity()) >= args.limit_cores
            cpus = list(range(args.limit_cores))
            p.cpu_affinity(cpus)
            print("Process limited to CPUs", cpus)

        datamodule = self.init_datamodule_cls_from_config(args, config)

        # setup
        rich.print(f"Setup {datamodule.__class__.__qualname__}...")
        datamodule.prepare_data()
        datamodule.setup("fit")
        try:
            train = datamodule.train_dataloader()
        except (MisconfigurationException, NotImplementedError):
            train = None
        try:
            val = datamodule.val_dataloader()
        except (MisconfigurationException, NotImplementedError):
            val = None
        try:
            test = datamodule.test_dataloader()
        except (MisconfigurationException, NotImplementedError):
            test = None

        # inspect
        rich.print("batch[0] = ", end="")
        rich.pretty.pprint(
            map_type_to_repr(
                next(iter(train)),
                torch.Tensor,
                lambda x: f"Tensor(..., shape={x.shape}, dtype={x.dtype}, device={x.device})",
            ),
            indent_guides = False,
        )

        if args.profile is not None:
            import cProfile
            profile = cProfile.Profile()
            profile.enable()

        # measure
        n_train, td_train = 0, 0
        n_val,   td_val   = 0, 0
        n_test,  td_test  = 0, 0
        try:
            for i in range(args.n_rounds):
                print(f"Round {i+1} of {args.n_rounds}")
                if train is not None:
                    epoch     = time.perf_counter_ns()
                    n_train  += sum(1 for _ in tqdm(train, desc=f"train {i+1}/{args.n_rounds}"))
                    td_train += time.perf_counter_ns() - epoch
                if val is not None:
                    epoch   = time.perf_counter_ns()
                    n_val  += sum(1 for _ in tqdm(val, desc=f"val {i+1}/{args.n_rounds}"))
                    td_val += time.perf_counter_ns() - epoch
                if test is not None:
                    epoch    = time.perf_counter_ns()
                    n_test  += sum(1 for _ in tqdm(test, desc=f"train {i+1}/{args.n_rounds}"))
                    td_test += time.perf_counter_ns() - epoch
        except KeyboardInterrupt:
            rich.print("Recieved a `KeyboardInterrupt`...")

        if args.profile is not None:
            profile.disable()
            if args.profile != "-":
                profile.dump_stats(args.profile)
            profile.print_stats("tottime")

        # summary
        for label, data, n, td in [
            ("train", train, n_train, td_train),
            ("val",   val,   n_val,   td_val),
            ("test",  test,  n_test,  td_test),
        ]:
            if not n: continue
            if data is not None:
                print(f"{label}:",
                    f" - per epoch: {td / args.n_rounds * 1e-9        :11.6f} s",
                    f" - per batch: {td / n * 1e-9  :11.6f} s",
                    f" - batches/s: {n / (td * 1e-9):11.6f}",
                sep="\n")

        datamodule.teardown("fit")



# helpers:

def open_in_netron(filename: str, data: bytes, *, timeout: float = 10):
    # filename is only used to determine the filetype
    url = serve_once_in_background(
        data,
        mime_type = "application/octet-stream",
        timeout   = timeout,
        port      = gen_random_port(),
    )
    url = f"https://netron.app/?url={urllib.parse.quote(url)}{filename}"
    print("Open in Netron:", url)
    webbrowser.get("firefox").open_new_tab(url)
    if timeout:
        time.sleep(timeout)

def remove_options_from_parser(parser: ArgumentParser, *options: str):
    options = set(options)
    # https://stackoverflow.com/questions/32807319/disable-remove-argument-in-argparse/36863647#36863647
    for action in parser._actions:
        if action.option_strings:
            option = action.option_strings[0]
            if option in options:
                parser._handle_conflict_resolve(None, [(option, action)])

def map_type_to_repr(batch, type_match: type, repr_func: callable):
    def mapper(value):
        if isinstance(value, type_match):
            return helpers.CustomRepr(repr_func(value))
        else:
            return value
    return helpers.map_tree(mapper, batch)

def datamodule_name_to_snake_case(datamodule: Union[str, type[DM]]) -> str:
    if not isinstance(datamodule, str):
        datamodule = datamodule.__name__
    datamodule = datamodule.replace("DataModule", "Datamodule")
    if datamodule != "Datamodule":
        datamodule = datamodule.removesuffix("Datamodule")
    return camel_to_snake_case(datamodule, sep="-", join_abbreviations=True)
