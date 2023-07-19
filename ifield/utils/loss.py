from abc import abstractmethod, ABC
from dataclasses import dataclass, field, fields, MISSING
from functools import wraps
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from tabulate import tabulate
from torch import nn
from typing import Optional, TypeVar, Union
import inspect
import math
import pytorch_lightning as pl
import typing
import warnings


HParamSchedule = TypeVar("HParamSchedule", bound="HParamScheduleBase")
Schedulable    = Union[HParamSchedule, int, float, str]

class HParamScheduleBase(ABC):
    _subclasses = {} # shared reference intended
    def __init_subclass__(cls):
        if not cls.__name__.startswith("_"):
            cls._subclasses[cls.__name__] = cls

    _infix      : Optional[str]  = field(init=False, repr=False, default=None)
    _param_name : Optional[str]  = field(init=False, repr=False, default=None)
    _expr       : Optional[str]  = field(init=False, repr=False, default=None)

    def get(self, module: nn.Module, *, trainer: Optional[pl.Trainer] = None) -> float:
        if module.training:
            if trainer is None:
                trainer = module.trainer # this assumes `module` is a pl.LightningModule
            value = self.get_train_value(
                epoch = trainer.current_epoch + (trainer.fit_loop.epoch_loop.batch_progress.current.processed / trainer.num_training_batches),
            )
            if trainer.logger is not None and self._param_name is not None and self.__class__ is not Const and trainer.global_step % 15 == 0:
                trainer.logger.log_metrics({
                    f"HParamSchedule/{self._param_name}": value,
                }, step=trainer.global_step)
            return value
        else:
            return self.get_eval_value()

    def _gen_data(self, n_epochs, steps_per_epoch=1000):
        global_steps = 0
        for epoch in range(n_epochs):
            for step in range(steps_per_epoch):
                yield (
                    epoch + step/steps_per_epoch,
                    self.get_train_value(epoch + step/steps_per_epoch),
                )
            global_steps += steps_per_epoch

    def plot(self, *a, ax: Optional[plt.Axes] = None, **kw) -> Artist:
        if ax is None: ax = plt.gca()
        out = ax.plot(*zip(*self._gen_data(*a, **kw)), label=self._expr)
        ax.set_title(self._param_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        return out

    def assert_positive(self, *a, **kw):
        for epoch, val in self._gen_data(*a, **kw):
            assert val >= 0, f"{epoch=}, {val=}"

    @abstractmethod
    def get_eval_value(self) -> float:
        ...

    @abstractmethod
    def get_train_value(self, epoch: float) -> float:
        ...

    def __add__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "+":
                return cls(self, rhs)
        return NotImplemented

    def __radd__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "+":
                return cls(lhs, self)
        return NotImplemented

    def __sub__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "-":
                return cls(self, rhs)
        return NotImplemented

    def __rsub__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "-":
                return cls(lhs, self)
        return NotImplemented

    def __mul__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "*":
                return cls(self, rhs)
        return NotImplemented

    def __rmul__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "*":
                return cls(lhs, self)
        return NotImplemented

    def __matmul__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "@":
                return cls(self, rhs)
        return NotImplemented

    def __rmatmul__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "@":
                return cls(lhs, self)
        return NotImplemented

    def __truediv__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "/":
                return cls(self, rhs)
        return NotImplemented

    def __rtruediv__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "/":
                return cls(lhs, self)
        return NotImplemented

    def __floordiv__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "//":
                return cls(self, rhs)
        return NotImplemented

    def __rfloordiv__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "//":
                return cls(lhs, self)
        return NotImplemented

    def __mod__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "%":
                return cls(self, rhs)
        return NotImplemented

    def __rmod__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "%":
                return cls(lhs, self)
        return NotImplemented

    def __pow__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "**":
                return cls(self, rhs)
        return NotImplemented

    def __rpow__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "**":
                return cls(lhs, self)
        return NotImplemented

    def __lshift__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "<<":
                return cls(self, rhs)
        return NotImplemented

    def __rlshift__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "<<":
                return cls(lhs, self)
        return NotImplemented

    def __rshift__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == ">>":
                return cls(self, rhs)
        return NotImplemented

    def __rrshift__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == ">>":
                return cls(lhs, self)
        return NotImplemented

    def __and__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "&":
                return cls(self, rhs)
        return NotImplemented

    def __rand__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "&":
                return cls(lhs, self)
        return NotImplemented

    def __xor__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "^":
                return cls(self, rhs)
        return NotImplemented

    def __rxor__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "^":
                return cls(lhs, self)
        return NotImplemented

    def __or__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "|":
                return cls(self, rhs)
        return NotImplemented

    def __ror__(self, lhs):
        for cls in self._subclasses.values():
            if cls._infix == "|":
                return cls(lhs, self)
        return NotImplemented

    def __ge__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == ">=":
                return cls(self, rhs)
        return NotImplemented

    def __gt__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == ">":
                return cls(self, rhs)
        return NotImplemented

    def __le__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "<=":
                return cls(self, rhs)
        return NotImplemented

    def __lt__(self, rhs):
        for cls in self._subclasses.values():
            if cls._infix == "<":
                return cls(self, rhs)
        return NotImplemented

    def __bool__(self):
        return True

    def __neg__(self):
        for cls in self._subclasses.values():
            if cls._infix == "-":
                return cls(0, self)
        return NotImplemented

    @property
    def is_const(self) -> bool:
        return False



def parse_dsl(config: Schedulable, name=None) -> HParamSchedule:
    if isinstance(config, HParamScheduleBase):
        return config
    elif isinstance(config, str):
        out = eval(config, {"__builtins__": {}, "lg": math.log10}, HParamScheduleBase._subclasses)
        if not isinstance(out, HParamScheduleBase):
            out = Const(out)
    else:
        out = Const(config)
    out._expr = config
    out._param_name = name
    return out


# decorator
def ensure_schedulables(func):
    signature = inspect.signature(func)
    module_name = func.__qualname__.removesuffix(".__init__")

    @wraps(func)
    def wrapper(*a, **kw):
        bound_args = signature.bind(*a, **kw)

        for param_name, param in signature.parameters.items():
            type_origin = typing.get_origin(param.annotation)
            type_args   = typing.get_args  (param.annotation)

            if type_origin is HParamSchedule or (type_origin is Union and (HParamSchedule in type_args or HParamScheduleBase in type_args)):
                if param_name in bound_args.arguments:
                    bound_args.arguments[param_name] = parse_dsl(bound_args.arguments[param_name], name=f"{module_name}.{param_name}")
                elif param.default is not param.empty:
                    bound_args.arguments[param_name] = parse_dsl(param.default, name=f"{module_name}.{param_name}")

        return func(
            *bound_args.args,
            **bound_args.kwargs,
        )
    return wrapper

# https://easings.net/

@dataclass
class _InfixBase(HParamScheduleBase):
    l : Union[HParamSchedule, int, float]
    r : Union[HParamSchedule, int, float]

    def _operation(self, l: float, r: float) -> float:
        raise NotImplementedError

    def get_eval_value(self) -> float:
        return self._operation(
            self.l.get_eval_value() if isinstance(self.l, HParamScheduleBase) else self.l,
            self.r.get_eval_value() if isinstance(self.r, HParamScheduleBase) else self.r,
        )

    def get_train_value(self, epoch: float) -> float:
        return self._operation(
            self.l.get_train_value(epoch) if isinstance(self.l, HParamScheduleBase) else self.l,
            self.r.get_train_value(epoch) if isinstance(self.r, HParamScheduleBase) else self.r,
        )

    def __bool__(self):
        if self.is_const:
            return bool(self.get_eval_value())
        else:
            return True

    @property
    def is_const(self) -> bool:
        return (self.l.is_const if isinstance(self.l, HParamScheduleBase) else True) \
           and (self.r.is_const if isinstance(self.r, HParamScheduleBase) else True)

@dataclass
class Add(_InfixBase):
    """ adds the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default="+")
    def _operation(self, l: float, r: float) -> float:
        return l + r


@dataclass
class Sub(_InfixBase):
    """ subtracts the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default="-")
    def _operation(self, l: float, r: float) -> float:
        return l - r


@dataclass
class Prod(_InfixBase):
    """ multiplies the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default="*")
    def _operation(self, l: float, r: float) -> float:
        return l * r
    @property
    def is_const(self) -> bool: # propagate identity
        l = self.l.get_eval_value() if isinstance(self.l, HParamScheduleBase) and self.l.is_const else self.l
        r = self.r.get_eval_value() if isinstance(self.r, HParamScheduleBase) and self.r.is_const else self.r
        return l == 0 or r == 0 or super().is_const


@dataclass
class Div(_InfixBase):
    """ divides the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default="/")
    def _operation(self, l: float, r: float) -> float:
        return l / r


@dataclass
class Pow(_InfixBase):
    """ raises the results of one schedule to the other """
    _infix : Optional[str]  = field(init=False, repr=False, default="**")
    def _operation(self, l: float, r: float) -> float:
        return l ** r


@dataclass
class Gt(_InfixBase):
    """ compares the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default=">")
    def _operation(self, l: float, r: float) -> float:
        return l > r


@dataclass
class Lt(_InfixBase):
    """ compares the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default="<")
    def _operation(self, l: float, r: float) -> float:
        return l < r


@dataclass
class Ge(_InfixBase):
    """ compares the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default=">=")
    def _operation(self, l: float, r: float) -> float:
        return l >= r


@dataclass
class Le(_InfixBase):
    """ compares the results of two schedules """
    _infix : Optional[str]  = field(init=False, repr=False, default="<=")
    def _operation(self, l: float, r: float) -> float:
        return l <= r


@dataclass
class Const(HParamScheduleBase):
    """ A way to ensure .get(...) exists """

    c : Union[int, float]

    def get_eval_value(self) -> float:
        return self.c

    def get_train_value(self, epoch: float) -> float:
        return self.c

    def __bool__(self):
        return bool(self.get_eval_value())

    @property
    def is_const(self) -> bool:
        return True

@dataclass
class Step(HParamScheduleBase):
    """ steps from 0 to 1 at `epoch` """

    epoch : float

    def get_eval_value(self) -> float:
        return 1

    def get_train_value(self, epoch: float) -> float:
         return 1 if epoch >= self.epoch else 0

@dataclass
class Linear(HParamScheduleBase):
    """ linear from 0 to 1 over `n_epochs`, delayed by `offset` """

    n_epochs : float
    offset : float = 0

    def get_eval_value(self) -> float:
        return 1

    def get_train_value(self, epoch: float) -> float:
        if self.n_epochs <= 0: return 1
        return min(max(epoch - self.offset, 0) / self.n_epochs, 1)

@dataclass
class EaseSin(HParamScheduleBase): # effectively 1-CosineAnnealing
    """ sinusoidal ease in-out from 0 to 1 over `n_epochs`, delayed by `offset` """

    n_epochs : float
    offset : float = 0

    def get_eval_value(self) -> float:
        return 1

    def get_train_value(self, epoch: float) -> float:
        x = min(max(epoch - self.offset, 0) / self.n_epochs, 1)
        return -(math.cos(math.pi * x) - 1) / 2

@dataclass
class EaseExp(HParamScheduleBase):
    """ exponential ease in-out from 0 to 1 over `n_epochs`, delayed by `offset` """

    n_epochs : float
    offset : float = 0

    def get_eval_value(self) -> float:
        return 1

    def get_train_value(self, epoch: float) -> float:
        if (epoch-self.offset) < 0:
            return 0
        if (epoch-self.offset) > self.n_epochs:
            return 1
        x = min(max(epoch - self.offset, 0) / self.n_epochs, 1)
        return (
            2**(20*x-10) / 2
            if x < 0.5 else
            (2 - 2**(-20*x+10)) / 2
        )

@dataclass
class Steps(HParamScheduleBase):
    """ Starts at 1, multiply by gamma every n epochs. Models StepLR in pytorch """
    step_size: float
    gamma: float = 0.1

    def get_eval_value(self) -> float:
        return 1
    def get_train_value(self, epoch: float) -> float:
        return self.gamma**int(epoch / self.step_size)

@dataclass
class MultiStep(HParamScheduleBase):
    """ Starts at 1, multiply by gamma every milstone epoch. Models MultiStepLR in pytorch """
    milestones: list[float]
    gamma: float = 0.1

    def get_eval_value(self) -> float:
        return 1
    def get_train_value(self, epoch: float) -> float:
        for i, m in list(enumerate(self.milestones))[::-1]:
            if epoch >= m:
                return self.gamma**(i+1)
        return 1

@dataclass
class Epoch(HParamScheduleBase):
    """ The current epoch, starting at 0 """

    def get_eval_value(self) -> float:
        return 0
    def get_train_value(self, epoch: float) -> float:
        return epoch

@dataclass
class Offset(HParamScheduleBase):
    """ Offsets the epoch for the subexpression, clamped above 0. Positive offsets makes it happen later """
    expr : Union[HParamSchedule, int, float]
    offset : float

    def get_eval_value(self) -> float:
        return self.expr.get_eval_value() if isinstance(self.expr, HParamScheduleBase) else self.expr
    def get_train_value(self, epoch: float) -> float:
        return self.expr.get_train_value(max(epoch - self.offset, 0)) if isinstance(self.expr, HParamScheduleBase) else self.expr

@dataclass
class Mod(HParamScheduleBase):
    """ The epoch in the subexptression is subject to a modulus. Use for warm restarts """

    modulus : float
    expr    : Union[HParamSchedule, int, float]

    def get_eval_value(self) -> float:
        return self.expr.get_eval_value() if isinstance(self.expr, HParamScheduleBase) else self.expr
    def get_train_value(self, epoch: float) -> float:
        return self.expr.get_train_value(epoch % self.modulus) if isinstance(self.expr, HParamScheduleBase) else self.expr


def main():
    import sys, rich.pretty
    if not sys.argv[2:]:
        print(f"Usage: {sys.argv[0]} n_epochs 'expression'")
        print("Available operations:")
        def mk_ops():
            for name, cls in HParamScheduleBase._subclasses.items():
                if isinstance(cls._infix, str):
                    yield (cls._infix, f"(infix) {cls.__doc__.strip()}")
                else:
                    yield (
                        f"""{name}({', '.join(
                            i.name
                            if i.default is MISSING else
                            f"{i.name}={i.default!r}"
                            for i in fields(cls)
                        )})""",
                        cls.__doc__.strip(),
                    )
        rich.print(tabulate(sorted(mk_ops()), tablefmt="plain"))
    else:
        n_epochs  = int(sys.argv[1])
        schedules = [parse_dsl(arg, name="cli arg") for arg in sys.argv[2:]]
        ax = plt.gca()
        print("[")
        for schedule in schedules:
            rich.print(f"  {schedule}, #  {schedule.is_const = }")
            schedule.plot(n_epochs, ax=ax)
        print("]")
        plt.show()

if __name__ == "__main__":
    main()
