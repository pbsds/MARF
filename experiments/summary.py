#!/usr/bin/env python
from concurrent.futures import ThreadPoolExecutor, Future, ProcessPoolExecutor
from functools import partial
from more_itertools import first, last, tail
from munch import Munch, DefaultMunch, munchify, unmunchify
from pathlib import Path
from statistics import mean, StatisticsError
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Iterable, Optional, Literal
from math import isnan
import json
import stat
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os, os.path
import re
import shlex
import time
import itertools
import shutil
import subprocess
import sys
import traceback
import typer
import warnings
import yaml
import tempfile

EXPERIMENTS = Path(__file__).resolve()
LOGDIR      = EXPERIMENTS / "logdir"
TENSORBOARD = LOGDIR / "tensorboard"
SLURM_LOGS  = LOGDIR / "slurm_logs"
CACHED_SUMMARIES = LOGDIR / "cached_summaries"
COMPUTED_SCORES = LOGDIR / "computed_scores"

MISSING = object()

class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None
SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

def camel_to_snake_case(text: str, sep: str = "_", join_abbreviations: bool = False) -> str:
    parts = (
        part.lower()
        for part in re.split(r'(?=[A-Z])', text)
        if part
    )
    if join_abbreviations: # this operation is not reversible
        parts = list(parts)
        if len(parts) > 1:
            for i, (a, b) in list(enumerate(zip(parts[:-1], parts[1:])))[::-1]:
                if len(a) == len(b) == 1:
                    parts[i] = parts[i] + parts.pop(i+1)
    return sep.join(parts)

def flatten_dict(data: dict, key_mapper: callable = lambda x: x) -> dict:
    if not any(isinstance(val, dict) for val in data.values()):
        return data
    else:
        return {
            k: v
            for k, v in data.items()
            if not isinstance(v, dict)
        } | {
            f"{key_mapper(p)}/{k}":v
            for p,d in data.items()
            if isinstance(d, dict)
            for k,v in d.items()
        }

def parse_jsonl(data: str) -> Iterable[dict]:
    yield from map(json.loads, (line for line in data.splitlines() if line.strip()))

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        data = f.read()
    yield from parse_jsonl(data)

def get_experiment_paths(filter: str | None, assert_dumped = False) -> Iterable[Path]:
    for path in TENSORBOARD.iterdir():
        if filter is not None and not re.search(filter, path.name): continue
        if not path.is_dir(): continue

        if not (path / "hparams.yaml").is_file():
            warnings.warn(f"Missing hparams: {path}")
            continue
        if not any(path.glob("events.out.tfevents.*")):
            warnings.warn(f"Missing tfevents: {path}")
            continue

        if __debug__ and assert_dumped:
            assert (path / "scalars/epoch.json").is_file(), path
            assert (path / "scalars/IntersectionFieldAutoDecoderModel.validation_step/loss.json").is_file(), path
            assert (path / "scalars/IntersectionFieldAutoDecoderModel.training_step/loss.json").is_file(), path

        yield path

def dump_pl_tensorboard_hparams(experiment: Path):
    with (experiment / "hparams.yaml").open() as f:
        hparams = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    shebang = None
    with (experiment / "config.yaml").open("w") as f:
        raw_yaml = hparams.get('_pickled_cli_args', {}).get('_raw_yaml', "").replace("\n\r", "\n")
        if raw_yaml.startswith("#!"): # preserve shebang
            shebang, _, raw_yaml = raw_yaml.partition("\n")
            f.write(f"{shebang}\n")
        f.write(f"# {' '.join(map(shlex.quote, hparams.get('_pickled_cli_args', {}).get('sys_argv', ['None'])))}\n\n")
        f.write(raw_yaml)
    if shebang is not None:
        os.chmod(experiment / "config.yaml", (experiment / "config.yaml").stat().st_mode | stat.S_IXUSR)
    print(experiment / "config.yaml", "written!", file=sys.stderr)

    with (experiment / "environ.yaml").open("w") as f:
        yaml.safe_dump(hparams.get('_pickled_cli_args', {}).get('host', {}).get('environ'), f)
    print(experiment / "environ.yaml", "written!", file=sys.stderr)

    with (experiment / "repo.patch").open("w") as f:
        f.write(hparams.get('_pickled_cli_args', {}).get('host', {}).get('vcs', "None"))
    print(experiment / "repo.patch", "written!", file=sys.stderr)

def dump_simple_tf_events_to_jsonl(output_dir: Path, *tf_files: Path):
    from google.protobuf.json_format import MessageToDict
    import tensorboard.backend.event_processing.event_accumulator
    s, l = {}, [] # reused sentinels

    #resource.setrlimit(resource.RLIMIT_NOFILE, (2**16,-1))
    file_handles = {}
    try:
        for tffile in tf_files:
            loader = tensorboard.backend.event_processing.event_file_loader.LegacyEventFileLoader(str(tffile))
            for event in loader.Load():
                for summary in MessageToDict(event).get("summary", s).get("value", l):
                    if "simpleValue" in summary:
                        tag = summary["tag"]
                        if tag not in file_handles:
                            fname = output_dir / f"{tag}.json"
                            print(f"Opening {str(fname)!r}...", file=sys.stderr)
                            fname.parent.mkdir(parents=True, exist_ok=True)
                            file_handles[tag] = fname.open("w") # ("a")
                        val = summary["simpleValue"]
                        data = json.dumps({
                            "step"      : event.step,
                            "value"     : float(val) if isinstance(val, str) else val,
                            "wall_time" : event.wall_time,
                        })
                        file_handles[tag].write(f"{data}\n")
    finally:
        if file_handles:
            print("Closing json files...", file=sys.stderr)
        for k, v in file_handles.items():
            v.close()


NO_FILTER = {
    "__uid",
    "_minutes",
    "_epochs",
    "_hp_nonlinearity",
    "_val_uloss_intersection",
    "_val_uloss_normal_cossim",
    "_val_uloss_intersection",
}
def filter_jsonl_columns(data: Iterable[dict | None], no_filter=NO_FILTER) -> list[dict]:
    def merge_siren_omega(data: dict) -> dict:
        return {
            key: (
                f"{val}-{data.get('hp_omega_0', 'ERROR')}"
                if (key.removeprefix("_"), val) == ("hp_nonlinearity", "sine") else
                val
            )
            for key, val in data.items()
            if key != "hp_omega_0"
        }

    def remove_uninteresting_cols(rows: list[dict]) -> Iterable[dict]:
        unique_vals = {}
        def register_val(key, val):
            unique_vals.setdefault(key, set()).add(repr(val))
            return val

        whitelisted = {
            key
            for row in rows
            for key, val in row.items()
            if register_val(key, val) and val not in ("None", "0", "0.0")
        }
        for key in unique_vals:
            for row in rows:
                if key not in row:
                    unique_vals[key].add(MISSING)
        for key, vals in unique_vals.items():
            if key not in whitelisted: continue
            if len(vals) == 1:
                whitelisted.remove(key)

        whitelisted.update(no_filter)

        yield from (
            {
                key: val
                for key, val in row.items()
                if key in whitelisted
            }
            for row in rows
        )

    def pessemize_types(rows: list[dict]) -> Iterable[dict]:
        types = {}
        order = (str, float, int, bool, tuple, type(None))
        for row in rows:
            for key, val in row.items():
                if isinstance(val, list): val = tuple(val)
                assert type(val) in order, (type(val), val)
                index = order.index(type(val))
                types[key] = min(types.get(key, 999), index)

        yield from (
            {
                key: order[types[key]](val) if val is not None else None
                for key, val in row.items()
            }
            for row in rows
        )

    data = (row for row in data if row is not None)
    data = map(partial(flatten_dict, key_mapper=camel_to_snake_case), data)
    data = map(merge_siren_omega, data)
    data = remove_uninteresting_cols(list(data))
    data = pessemize_types(list(data))

    return data

PlotMode = Literal["stackplot", "lineplot"]

def plot_losses(experiments: list[Path], mode: PlotMode, write: bool = False, dump: bool = False, training: bool = False, unscaled: bool = False, force=True):
    def get_losses(experiment: Path, training: bool = True, unscaled: bool = False) -> Iterable[Path]:
        if not training and unscaled:
            return experiment.glob("scalars/*.validation_step/unscaled_loss_*.json")
        elif not training and not unscaled:
            return experiment.glob("scalars/*.validation_step/loss_*.json")
        elif training and unscaled:
            return experiment.glob("scalars/*.training_step/unscaled_loss_*.json")
        elif training and not unscaled:
            return experiment.glob("scalars/*.training_step/loss_*.json")

    print("Mapping colors...")
    configurations = [
        dict(unscaled=unscaled, training=training),
    ] if not write else [
        dict(unscaled=False, training=False),
        dict(unscaled=False, training=True),
        dict(unscaled=True,  training=False),
        dict(unscaled=True,  training=True),
    ]
    legends = set(
        f"""{
            loss.parent.name.split(".", 1)[0]
        }.{
            loss.name.removesuffix(loss.suffix).removeprefix("unscaled_")
        }"""
        for experiment in experiments
        for kw in configurations
        for loss in get_losses(experiment, **kw)
    )
    colormap = dict(zip(
        sorted(legends),
        itertools.cycle(mcolors.TABLEAU_COLORS),
    ))

    def mkplot(experiment: Path, training: bool = True, unscaled: bool = False) -> tuple[bool, str]:
        label = f"{'unscaled' if unscaled else 'scaled'} {'training' if training else 'validation'}"
        if write:
            old_savefig_fname = experiment       / f"{label.replace(' ', '-')}-{mode}.png"
            savefig_fname = experiment / "plots" / f"{label.replace(' ', '-')}-{mode}.png"
            savefig_fname.parent.mkdir(exist_ok=True, parents=True)
            if old_savefig_fname.is_file():
                old_savefig_fname.rename(savefig_fname)
            if savefig_fname.is_file() and not force:
                return True, "savefig_fname already exists"

        # Get and sort data
        losses = {}
        for loss in get_losses(experiment, training=training, unscaled=unscaled):
            model = loss.parent.name.split(".", 1)[0]
            name = loss.name.removesuffix(loss.suffix).removeprefix("unscaled_")
            losses[f"{model}.{name}"] = (loss, list(read_jsonl(loss)))
        losses = dict(sorted(losses.items())) # sort keys
        if not losses:
            return True, "no losses"

        # unwrap
        steps = [i["step"] for i in first(losses.values())[1]]
        values = [
            [i["value"] if not isnan(i["value"]) else 0 for i in data]
            for name, (scalar, data) in losses.items()
        ]

        # normalize
        if mode == "stackplot":
            totals = list(map(sum, zip(*values)))
            values = [
                [i / t for i, t in zip(data, totals)]
                for data in values
            ]

        print(experiment.name, label)
        fig, ax = plt.subplots(figsize=(16, 12))

        if mode == "stackplot":
            ax.stackplot(steps, values,
                colors = list(map(colormap.__getitem__, losses.keys())),
                labels = list(
                    label.split(".", 1)[1].removeprefix("loss_")
                    for label in losses.keys()
                ),
            )
            ax.set_xlim(0, steps[-1])
            ax.set_ylim(0, 1)
            ax.invert_yaxis()

        elif mode == "lineplot":
            for data, color, label in zip(
                values,
                map(colormap.__getitem__, losses.keys()),
                list(losses.keys()),
            ):
                ax.plot(steps, data,
                    color = color,
                    label = label,
                )
            ax.set_xlim(0, steps[-1])

        else:
            raise ValueError(f"{mode=}")

        ax.legend()
        ax.set_title(f"{label} loss\n{experiment.name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("loss%")

        if mode == "stackplot":
            ax2 = make_axes_locatable(ax).append_axes("bottom", 0.8, pad=0.05, sharex=ax)
            ax2.stackplot( steps, totals )

        for tl in ax.get_xticklabels(): tl.set_visible(False)

        fig.tight_layout()

        if write:
            fig.savefig(savefig_fname, dpi=300)
            print(savefig_fname)
            plt.close(fig)

        return False, None

    print("Plotting...")
    if write:
        matplotlib.use('agg') # fixes "WARNING: QApplication was not created in the main() thread."
    any_error = False
    if write:
        with ThreadPoolExecutor(max_workers=None) as pool:
            futures = [
                (experiment, pool.submit(mkplot, experiment, **kw))
                for experiment in experiments
                for kw in configurations
            ]
    else:
        def mkfuture(item):
            f = Future()
            f.set_result(item)
            return f
        futures = [
            (experiment, mkfuture(mkplot(experiment, **kw)))
            for experiment in experiments
            for kw in configurations
        ]

    for experiment, future in futures:
        try:
            err, msg = future.result()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            any_error = True
            continue
        if err:
            print(f"{msg}: {experiment.name}")
            any_error = True
            continue

    if not any_error and not write: # show in main thread
        plt.show()
    elif not write:
        print("There were errors, will not show figure...", file=sys.stderr)



# =========

app = typer.Typer(no_args_is_help=True, add_completion=False)

@app.command(help="Dump simple tensorboard events to json and extract some pytorch lightning hparams")
def tf_dump(tfevent_files: list[Path], j: int = typer.Option(1, "-j"), force: bool = False):
    # expand to all tfevents files (there may be more than one)
    tfevent_files = sorted(set([
        tffile
        for tffile in tfevent_files
        if tffile.name.startswith("events.out.tfevents.")
    ] + [
        tffile
        for experiment_dir in tfevent_files
        if experiment_dir.is_dir()
        for tffile in experiment_dir.glob("events.out.tfevents.*")
    ] + [
        tffile
        for hparam_file in tfevent_files
        if hparam_file.name in ("hparams.yaml", "config.yaml")
        for tffile in hparam_file.parent.glob("events.out.tfevents.*")
    ]))

    # filter already dumped
    if not force:
        tfevent_files = [
            tffile
            for tffile in tfevent_files
            if not (
                (tffile.parent / "scalars/epoch.json").is_file()
                and
                tffile.stat().st_mtime < (tffile.parent / "scalars/epoch.json").stat().st_mtime
            )
        ]

    if not tfevent_files:
        raise typer.BadParameter("Nothing to be done, consider --force")

    jobs = {}
    for tffile in tfevent_files:
        if not tffile.is_file():
            print("ERROR: file not found:", tffile, file=sys.stderr)
            continue
        output_dir = tffile.parent / "scalars"
        jobs.setdefault(output_dir, []).append(tffile)
    with ProcessPoolExecutor() as p:
        for experiment in set(tffile.parent for tffile in tfevent_files):
            p.submit(dump_pl_tensorboard_hparams, experiment)
        for output_dir, tffiles in jobs.items():
            p.submit(dump_simple_tf_events_to_jsonl, output_dir, *tffiles)

@app.command(help="Propose experiment regexes")
def propose(cmd: str = typer.Argument("summary"), null: bool = False):
    def get():
        for i in TENSORBOARD.iterdir():
            if not i.is_dir(): continue
            if not (i / "hparams.yaml").is_file(): continue
            prefix, name, *hparams, year, month, day, hhmm, uid = i.name.split("-")
            yield f"{name}.*-{year}-{month}-{day}"
    proposals = sorted(set(get()), key=lambda x: x.split(".*-", 1)[1])
    print("\n".join(
        f"{'>/dev/null ' if null else ''}{sys.argv[0]} {cmd or 'summary'} {shlex.quote(i)}"
        for i in proposals
    ))

@app.command("list", help="List used experiment regexes")
def list_cached_summaries(cmd: str = typer.Argument("summary")):
    if not CACHED_SUMMARIES.is_dir():
        cached = []
    else:
        cached = [
            i.name.removesuffix(".jsonl")
            for i in CACHED_SUMMARIES.iterdir()
            if i.suffix == ".jsonl"
            if i.is_file() and i.stat().st_size
        ]
    def order(key: str) -> list[str]:
        return re.sub(r'[^0-9\-]', '', key.split(".*")[-1]).strip("-").split("-") + [key]

    print("\n".join(
        f"{sys.argv[0]} {cmd or 'summary'} {shlex.quote(i)}"
        for i in sorted(cached, key=order)
    ))

@app.command(help="Precompute the summary of a experiment regex")
def compute_summary(filter: str, force: bool = False, dump: bool = False, no_cache: bool = False):
    cache = CACHED_SUMMARIES / f"{filter}.jsonl"
    if cache.is_file() and cache.stat().st_size:
        if not force:
            raise FileExistsError(cache)

    def mk_summary(path: Path) -> dict | None:
        cache = path / "train_summary.json"
        if cache.is_file() and cache.stat().st_size and cache.stat().st_mtime > (path/"scalars/epoch.json").stat().st_mtime:
            with cache.open() as f:
                return json.load(f)
        else:
            with (path / "hparams.yaml").open() as f:
                hparams = munchify(yaml.load(f, Loader=SafeLoaderIgnoreUnknown), factory=partial(DefaultMunch, None))
            config = hparams._pickled_cli_args._raw_yaml
            config = munchify(yaml.load(config, Loader=SafeLoaderIgnoreUnknown), factory=partial(DefaultMunch, None))

            try:
                train_loss = list(read_jsonl(path / "scalars/IntersectionFieldAutoDecoderModel.training_step/loss.json"))
                val_loss   = list(read_jsonl(path / "scalars/IntersectionFieldAutoDecoderModel.validation_step/loss.json"))
            except:
                traceback.print_exc(file=sys.stderr)
                return None

            out = Munch()
            out.uid     = path.name.rsplit("-", 1)[-1]
            out.name    = path.name
            out.date    = "-".join(path.name.split("-")[-5:-1])
            out.epochs  = int(last(read_jsonl(path / "scalars/epoch.json"))["value"])
            out.steps   = val_loss[-1]["step"]
            out.gpu     = hparams._pickled_cli_args.host.gpus[1][1]

            if val_loss[-1]["wall_time"] - val_loss[0]["wall_time"] > 0:
                out.batches_per_second = val_loss[-1]["step"] / (val_loss[-1]["wall_time"] - val_loss[0]["wall_time"])
            else:
                out.batches_per_second = 0

            out.minutes = (val_loss[-1]["wall_time"] - train_loss[0]["wall_time"]) / 60

            if (path / "scalars/PsutilMonitor/gpu.00.memory.used.json").is_file():
                max(i["value"] for i in read_jsonl(path / "scalars/PsutilMonitor/gpu.00.memory.used.json"))

            for metric_path in (path / "scalars/IntersectionFieldAutoDecoderModel.validation_step").glob("*.json"):
                if not metric_path.is_file() or not metric_path.stat().st_size: continue

                metric_name = metric_path.name.removesuffix(".json")
                metric_data = read_jsonl(metric_path)
                try:
                    out[f"val_{metric_name}"] = mean(i["value"] for i in tail(5, metric_data))
                except StatisticsError:
                    out[f"val_{metric_name}"] = float('nan')

            for metric_path in (path / "scalars/IntersectionFieldAutoDecoderModel.training_step").glob("*.json"):
                if not any(i in metric_path.name for i in ("miss_radius_grad", "sphere_center_grad", "loss_tangential_reg", "multi_view")): continue
                if not metric_path.is_file() or not metric_path.stat().st_size: continue

                metric_name = metric_path.name.removesuffix(".json")
                metric_data = read_jsonl(metric_path)
                try:
                    out[f"train_{metric_name}"] = mean(i["value"] for i in tail(5, metric_data))
                except StatisticsError:
                    out[f"train_{metric_name}"] = float('nan')

            out.hostname = hparams._pickled_cli_args.host.hostname

            for key, val in config.IntersectionFieldAutoDecoderModel.items():
                if isinstance(val, dict):
                    out.update({f"hp_{key}_{k}": v for k, v in val.items()})
                elif isinstance(val, float | int | str | bool | None):
                    out[f"hp_{key}"] = val

            with cache.open("w") as f:
                json.dump(unmunchify(out), f)

            return dict(out)

    experiments = list(get_experiment_paths(filter, assert_dumped=not dump))
    if not experiments:
        raise typer.BadParameter("No matching experiment")
    if dump:
        try:
            tf_dump(experiments) # force=force_dump)
        except typer.BadParameter:
            pass

    # does literally nothing, thanks GIL
    with ThreadPoolExecutor() as p:
        results = list(p.map(mk_summary, experiments))

    if any(result is None for result in results):
        if all(result is None for result in results):
            print("No summary succeeded", file=sys.stderr)
            raise typer.Exit(exit_code=1)
        warnings.warn("Some summaries failed:\n" + "\n".join(
            str(experiment)
            for result, experiment in zip(results, experiments)
            if result is None
        ))

    summaries = "\n".join( map(json.dumps, results) )
    if not no_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w") as f:
            f.write(summaries)
    return summaries

@app.command(help="Show the summary of a experiment regex, precompute it if needed")
def summary(filter: Optional[str] = typer.Argument(None), force: bool = False, dump: bool = False, all: bool = False):
    if filter is None:
        return list_cached_summaries("summary")

    def key_mangler(key: str) -> str:
        for pattern, sub in (
            (r'^val_unscaled_loss_',   r'val_uloss_'),
            (r'^train_unscaled_loss_', r'train_uloss_'),
            (r'^val_loss_',            r'val_sloss_'),
            (r'^train_loss_',          r'train_sloss_'),
        ):
            key = re.sub(pattern, sub, key)

        return key

    cache = CACHED_SUMMARIES / f"{filter}.jsonl"
    if force or not (cache.is_file() and cache.stat().st_size):
        compute_summary(filter, force=force, dump=dump)
        assert cache.is_file() and cache.stat().st_size, (cache, cache.stat())

    if os.isatty(0) and os.isatty(1) and shutil.which("vd"):
        rows = read_jsonl(cache)
        rows = ({key_mangler(k): v for k, v in row.items()} if row is not None else None for row in rows)
        if not all:
            rows = filter_jsonl_columns(rows)
        rows = ({k: v for k, v in row.items() if not k.startswith(("val_sloss_", "train_sloss_"))} for row in rows)
        data = "\n".join(map(json.dumps, rows))
        subprocess.run(["vd",
            #"--play", EXPERIMENTS / "set-key-columns.vd",
            "-f", "jsonl"
        ], input=data, text=True, check=True)
    else:
        with cache.open() as f:
            print(f.read())

@app.command(help="Filter uninteresting keys from jsonl stdin")
def filter_cols():
    rows = map(json.loads, (line for line in sys.stdin.readlines() if line.strip()))
    rows = filter_jsonl_columns(rows)
    print(*map(json.dumps, rows), sep="\n")

@app.command(help="Run a command for each experiment matched by experiment regex")
def exec(filter: str, cmd: list[str], j: int = typer.Option(1, "-j"), dumped: bool = False, undumped: bool = False):
    # inspired by fd / gnu parallel
    def populate_cmd(experiment: Path, cmd: Iterable[str]) -> Iterable[str]:
        any = False
        for i in cmd:
            if i == "{}":
                any = True
                yield str(experiment / "hparams.yaml")
            elif i == "{//}":
                any = True
                yield str(experiment)
            else:
                yield i
        if not any:
            yield str(experiment / "hparams.yaml")

    with ThreadPoolExecutor(max_workers=j or None) as p:
        results = p.map(subprocess.run, (
            list(populate_cmd(experiment, cmd))
            for experiment in get_experiment_paths(filter)
            if not dumped   or     (experiment / "scalars/epoch.json").is_file()
            if not undumped or not (experiment / "scalars/epoch.json").is_file()
        ))

    if any(i.returncode for i in results):
        return typer.Exit(1)

@app.command(help="Show stackplot of experiment loss")
def stackplot(filter: str, write: bool = False, dump: bool = False, training: bool = False, unscaled: bool = False, force: bool = False):
    experiments = list(get_experiment_paths(filter, assert_dumped=not dump))
    if not experiments:
        raise typer.BadParameter("No match")
    if dump:
        try:
            tf_dump(experiments)
        except typer.BadParameter:
            pass

    plot_losses(experiments,
        mode     = "stackplot",
        write    = write,
        dump     = dump,
        training = training,
        unscaled = unscaled,
        force    = force,
    )

@app.command(help="Show stackplot of experiment loss")
def lineplot(filter: str, write: bool = False, dump: bool = False, training: bool = False, unscaled: bool = False, force: bool = False):
    experiments = list(get_experiment_paths(filter, assert_dumped=not dump))
    if not experiments:
        raise typer.BadParameter("No match")
    if dump:
        try:
            tf_dump(experiments)
        except typer.BadParameter:
            pass

    plot_losses(experiments,
        mode     = "lineplot",
        write    = write,
        dump     = dump,
        training = training,
        unscaled = unscaled,
        force    = force,
    )

@app.command(help="Open tensorboard for the experiments matching the regex")
def tensorboard(filter: Optional[str] = typer.Argument(None), watch: bool = False):
    if filter is None:
        return list_cached_summaries("tensorboard")
    experiments = list(get_experiment_paths(filter, assert_dumped=False))
    if not experiments:
        raise typer.BadParameter("No match")

    with tempfile.TemporaryDirectory(suffix=f"ifield-{filter}") as d:
        treefarm = Path(d)
        with ThreadPoolExecutor(max_workers=2) as p:
            for experiment in experiments:
                (treefarm / experiment.name).symlink_to(experiment)

            cmd = ["tensorboard", "--logdir", d]
            print("+", *map(shlex.quote, cmd), file=sys.stderr)
            tensorboard = p.submit(subprocess.run, cmd, check=True)
            if not watch:
                tensorboard.result()

            else:
                all_experiments = set(get_experiment_paths(None, assert_dumped=False))
                while not tensorboard.done():
                    time.sleep(10)
                    new_experiments = set(get_experiment_paths(None, assert_dumped=False)) - all_experiments
                    if new_experiments:
                        for experiment in new_experiments:
                            print(f"Adding {experiment.name!r}...", file=sys.stderr)
                            (treefarm / experiment.name).symlink_to(experiment)
                        all_experiments.update(new_experiments)

@app.command(help="Compute evaluation metrics")
def metrics(filter: Optional[str] = typer.Argument(None), dump: bool = False, dry: bool = False, prefix: Optional[str] = typer.Option(None), derive: bool = False, each: bool = False, no_total: bool = False):
    if filter is None:
        return list_cached_summaries("metrics --derive")
    experiments = list(get_experiment_paths(filter, assert_dumped=False))
    if not experiments:
        raise typer.BadParameter("No match")
    if dump:
        try:
            tf_dump(experiments)
        except typer.BadParameter:
            pass

    def run(*cmd):
        if prefix is not None:
            cmd = [*shlex.split(prefix), *cmd]
        if dry:
            print(*map(shlex.quote, map(str, cmd)))
        else:
            print("+", *map(shlex.quote, map(str, cmd)))
            subprocess.run(cmd)

    for experiment in experiments:
        if no_total: continue
        if not (experiment / "compute-scores/metrics.json").is_file():
            run(
                "python", "./marf.py", "module", "--best", experiment / "hparams.yaml",
                "compute-scores", experiment / "compute-scores/metrics.json",
                "--transpose",
            )
        if not (experiment / "compute-scores/metrics-last.json").is_file():
            run(
                "python", "./marf.py", "module", "--last", experiment / "hparams.yaml",
                "compute-scores", experiment / "compute-scores/metrics-last.json",
                "--transpose",
            )
        if "2prif-" not in experiment.name: continue
        if not (experiment / "compute-scores/metrics-sans_outliers.json").is_file():
            run(
                "python", "./marf.py", "module", "--best", experiment / "hparams.yaml",
                "compute-scores", experiment / "compute-scores/metrics-sans_outliers.json",
                "--transpose", "--filter-outliers"
            )
        if not (experiment / "compute-scores/metrics-last-sans_outliers.json").is_file():
            run(
                "python", "./marf.py", "module", "--last", experiment / "hparams.yaml",
                "compute-scores", experiment / "compute-scores/metrics-last-sans_outliers.json",
                "--transpose", "--filter-outliers"
            )

    if dry: return
    if prefix is not None:
        print("prefix was used, assuming a job scheduler was used, will not print scores.", file=sys.stderr)
        return

    metrics = [
        *(experiment / "compute-scores/metrics.json"                    for experiment in experiments),
        *(experiment / "compute-scores/metrics-last.json"               for experiment in experiments),
        *(experiment / "compute-scores/metrics-sans_outliers.json"      for experiment in experiments if "2prif-" in experiment.name),
        *(experiment / "compute-scores/metrics-last-sans_outliers.json" for experiment in experiments if "2prif-" in experiment.name),
    ]
    if not no_total:
        assert all(metric.exists() for metric in metrics)
    else:
        metrics = (metric for metric in metrics if metric.exists())

    out = []
    for metric in metrics:
        experiment = metric.parent.parent.name
        is_last = metric.name in ("metrics-last.json", "metrics-last-sans_outliers.json")
        with metric.open() as f:
            data = json.load(f)

        if derive:
            derived = {}
            objs = [i for i in data.keys() if i != "_hparams"]
            for obj in (objs if each else []) + [None]:
                if obj is None:
                    d = DefaultMunch(0)
                    for obj in objs:
                        for k, v in data[obj].items():
                            d[k] += v
                    obj = "_all_"
                    n_cd  = data["_hparams"]["n_cd"]  * len(objs)
                    n_emd = data["_hparams"]["n_emd"] * len(objs)
                else:
                    d = munchify(data[obj])
                    n_cd  = data["_hparams"]["n_cd"]
                    n_emd = data["_hparams"]["n_emd"]

                precision = d.TP / (d.TP + d.FP)
                recall    = d.TP / (d.TP + d.FN)
                derived[obj] = dict(
                    filtered  = d.n_outliers / d.n if "n_outliers" in d else None,
                    iou       = d.TP / (d.TP + d.FN + d.FP),
                    precision = precision,
                    recall    = recall,
                    f_score   = 2 * (precision * recall) / (precision + recall),
                    cd        = d.cd_dist / n_cd,
                    emd       = d.emd     / n_emd,
                    cos_med   = 1 - (d.cd_cos_med / n_cd) if "cd_cos_med" in d else None,
                    cos_jac   = 1 - (d.cd_cos_jac / n_cd),
                )
            data = derived if each else derived["_all_"]

        data["uid"]             = experiment.rsplit("-", 1)[-1]
        data["experiment_name"] = experiment
        data["is_last"]         = is_last

        out.append(json.dumps(data))

    if derive and not each and os.isatty(0) and os.isatty(1) and shutil.which("vd"):
        subprocess.run(["vd", "-f", "jsonl"], input="\n".join(out), text=True, check=True)
    else:
        print("\n".join(out))

if __name__ == "__main__":
    app()
