from . import param
from dataclasses import dataclass
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import Union, Literal, Optional, TypeVar
import concurrent.futures
import psutil
import pytorch_lightning as pl
import statistics
import threading
import time
import torch
import yaml

# from https://github.com/yaml/pyyaml/issues/240#issuecomment-1018712495
def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.add_representer(str, str_presenter)


LoggerStr = Literal[
    #"csv",
    "tensorboard",
    #"mlflow",
    #"comet",
    #"neptune",
    #"wandb",
    None]
try:
    Logger    = TypeVar("L", bound=pl.loggers.Logger)
except AttributeError:
    Logger    = TypeVar("L", bound=pl.loggers.base.LightningLoggerBase)

def make_logger(
        experiment_name  : str,
        default_root_dir : Union[str, Path], # from pl.Trainer
        save_dir         : Union[str, Path],
        type             : LoggerStr = "tensorboard",
        project          : str       = "ifield",
        ) -> Optional[Logger]:
    if type is None:
        return None
    elif type == "tensorboard":
        return pl.loggers.TensorBoardLogger(
            name      = "tensorboard",
            save_dir  = Path(default_root_dir) / save_dir,
            version   = experiment_name,
            log_graph = True,
        )
    raise ValueError(f"make_logger({type=})")

def make_jinja_template(*, save_dir: Union[None, str, Path], **kw) -> str:
    return param.make_jinja_template(make_logger,
        defaults = dict(
            save_dir = save_dir,
        ),
        exclude_list = {
            "experiment_name",
            "default_root_dir",
        },
        **({"name": "logging"} | kw),
    )

def get_checkpoints(experiment_name, default_root_dir, save_dir, type, project) -> list[Path]:
    if type is None:
        return None
    if type == "tensorboard":
        folder = Path(default_root_dir) / save_dir / "tensorboard" / experiment_name
        return folder.glob("*.ckpt")
    if type == "mlflow":
        raise NotImplementedError(f"{type=}")
    if type == "wandb":
        raise NotImplementedError(f"{type=}")
    raise ValueError(f"get_checkpoint({type=})")


def log_config(_logger: Logger, **kwargs: Union[str, dict, list, int, float]):
    assert isinstance(_logger, pl.loggers.Logger) \
        or isinstance(_logger, pl.loggers.base.LightningLoggerBase), _logger

    _logger: pl.loggers.TensorBoardLogger
    _logger.log_hyperparams(params=kwargs)

@dataclass
class ModelOutputMonitor(pl.callbacks.Callback):
    log_training   : bool = True
    log_validation : bool = True

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(f"Cannot use {self._class__.__name__} callback with Trainer that has no logger.")

    @staticmethod
    def _log_outputs(trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, fname: str):
        if outputs is None:
            return
        elif isinstance(outputs, list) or isinstance(outputs, tuple):
            outputs = {
                f"loss[{i}]": v
                for i, v in enumerate(outputs)
            }
        elif isinstance(outputs, torch.Tensor):
            outputs = {
                "loss": outputs,
            }
        elif isinstance(outputs, dict):
            pass
        else:
            raise ValueError
        sep = trainer.logger.group_separator
        pl_module.log_dict({
            f"{pl_module.__class__.__qualname__}.{fname}{sep}{k}":
                float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in outputs.items()
        }, sync_dist=True)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, unused=0):
        if self.log_training:
            self._log_outputs(trainer, pl_module, outputs, "training_step")

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        if self.log_validation:
            self._log_outputs(trainer, pl_module, outputs, "validation_step")

class EpochTimeMonitor(pl.callbacks.Callback):
    __slots__ = [
        "epoch_start",
        "epoch_start_train",
        "epoch_start_validation",
        "epoch_start_test",
        "epoch_start_predict",
    ]

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(f"Cannot use {self._class__.__name__} callback with Trainer that has no logger.")


    @rank_zero_only
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epoch_start_train = time.time()

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epoch_start_validation = time.time()

    @rank_zero_only
    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epoch_start_test = time.time()

    @rank_zero_only
    def on_predict_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epoch_start_predict = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        t = time.time() - self.epoch_start_train
        del self.epoch_start_train
        sep = trainer.logger.group_separator
        trainer.logger.log_metrics({f"{self.__class__.__qualname__}{sep}epoch_train_time" : t}, step=trainer.global_step)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        t = time.time() - self.epoch_start_validation
        del self.epoch_start_validation
        sep = trainer.logger.group_separator
        trainer.logger.log_metrics({f"{self.__class__.__qualname__}{sep}epoch_validation_time" : t}, step=trainer.global_step)

    @rank_zero_only
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        t = time.time() - self.epoch_start_test
        del self.epoch_start_validation
        sep = trainer.logger.group_separator
        trainer.logger.log_metrics({f"{self.__class__.__qualname__}{sep}epoch_test_time" : t}, step=trainer.global_step)

    @rank_zero_only
    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        t = time.time() - self.epoch_start_predict
        del self.epoch_start_validation
        sep = trainer.logger.group_separator
        trainer.logger.log_metrics({f"{self.__class__.__qualname__}{sep}epoch_predict_time" : t}, step=trainer.global_step)

@dataclass
class PsutilMonitor(pl.callbacks.Callback):
    sample_rate : float = 0.2 # times per second

    _should_stop = False

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.loggers:
            raise MisconfigurationException(f"Cannot use {self._class__.__name__} callback with Trainer that has no logger.")
        assert not hasattr(self, "_thread")

        self._should_stop = False
        self._thread = threading.Thread(
            target = self.thread_target,
            name = self.thread_target.__qualname__,
            args = [trainer],
            daemon=True,
        )
        self._thread.start()

    @rank_zero_only
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        assert getattr(self, "_thread", None) is not None
        self._should_stop = True
        del self._thread

    def thread_target(self, trainer: pl.Trainer):
        uses_gpu = isinstance(trainer.accelerator, (pl.accelerators.GPUAccelerator, pl.accelerators.CUDAAccelerator))
        gpu_ids = trainer.device_ids

        prefix = f"{self.__class__.__qualname__}{trainer.logger.group_separator}"

        while not self._should_stop:
            step = trainer.global_step
            p = psutil.Process()

            meminfo      = p.memory_info()
            rss_ram      = meminfo.rss / 1024**2 # MB
            vms_ram      = meminfo.vms / 1024**2 # MB

            util_per_cpu = psutil.cpu_percent(percpu=True)

            util_per_cpu = [util_per_cpu[i] for i in p.cpu_affinity()]
            util_total   = statistics.mean(util_per_cpu)

            if uses_gpu:
                with concurrent.futures.ThreadPoolExecutor() as e:
                    if hasattr(pl.accelerators, "cuda"):
                        gpu_stats = e.map(pl.accelerators.cuda.get_nvidia_gpu_stats, gpu_ids)
                    else:
                        gpu_stats = e.map(pl.accelerators.gpu.get_nvidia_gpu_stats, gpu_ids)
                trainer.logger.log_metrics({
                    f"{prefix}ram.rss" : rss_ram,
                    f"{prefix}ram.vms" : vms_ram,
                    f"{prefix}cpu.total"   : util_total,
                    **{ f"{prefix}cpu.{i:03}.utilization" : stat for i, stat in enumerate(util_per_cpu) },
                    **{
                        f"{prefix}gpu.{gpu_idx:02}.{key.split(' ',1)[0]}" : stat
                        for gpu_idx, stats in zip(gpu_ids, gpu_stats)
                        for key, stat in stats.items()
                    },
                }, step = step)
            else:
                trainer.logger.log_metrics({
                    f"{prefix}cpu.total" : util_total,
                    **{ f"{prefix}cpu.{i:03}.utilization" : stat for i, stat in enumerate(util_per_cpu) },
                }, step = step)

            time.sleep(1 / self.sample_rate)
        print("DAEMON END")
