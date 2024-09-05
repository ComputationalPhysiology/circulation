from __future__ import annotations
from typing import Callable, Any, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import json
from collections import defaultdict
import numpy as np
import time
import logging
from rich.table import Table

from . import units
from . import time_varying_elastance
from . import log


logger = logging.getLogger(__name__)


def smooth_heavyside(x):
    return np.arctan(np.pi / 2 * x * 200) * 1 / np.pi + 0.5


def remove_units(parameters: dict[str, Any]) -> dict[str, Any]:
    d = {}
    for k, v in parameters.items():
        if isinstance(v, units.pint.Quantity):
            d[k] = v.magnitude
        elif isinstance(v, dict):
            d[k] = remove_units(v)
        else:
            d[k] = v
    return d


class CallBack(Protocol):
    def __call__(self, model: "CirculationModel", t: float = 0, save: bool = True) -> None: ...


def dummy_callback(model: "CirculationModel", t: float = 0, save: bool = True) -> None:
    pass


class CirculationModel(ABC):
    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        outdir: Path = Path("results"),
        add_units: bool = False,
        callback: CallBack | None = None,
        verbose: bool = False,
        comm=None,
        callback_save_state: CallBack | None = None,
    ):
        self.parameters = type(self).default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)
        if not add_units:
            self.parameters = remove_units(self.parameters)
        self._add_units = add_units
        self.outdir = outdir
        outdir.mkdir(exist_ok=True, parents=True)

        if callback is not None:
            assert callable(callback), "callback must be callable"

            self.callback = callback
        else:
            self.callback = dummy_callback

        if callback_save_state is not None:
            assert callable(callback_save_state), "callback_save_state must be callable"

            self.callback_save_state = callback_save_state
        else:
            self.callback_save_state = dummy_callback

        self._verbose = verbose
        loglevel = logging.DEBUG if verbose else logging.INFO
        log.setup_logging(level=loglevel)
        self._comm = comm

    def _initialize(self):
        self.var = {}
        self.state = type(self).default_initial_conditions()
        self.update_state()
        self.update_static_variables(0.0)

        if self._comm is None or (self._comm is not None and self._comm.rank == 0):
            # Dump parameters to file
            (self.outdir / "parameters.json").write_text(json.dumps(self.parameters, indent=2))
            # Dump initial conditions to file
            (self.outdir / "initial_conditions.json").write_text(json.dumps(self.state, indent=2))

    @property
    def THB(self):
        if self._add_units:
            return (1 / self.parameters["BPM"]).to(units.ureg("s"))

        return 60.0 / self.parameters["BPM"]

    @staticmethod
    @abstractmethod
    def default_parameters() -> dict[str, Any]: ...

    @abstractmethod
    def update_static_variables(self, t: float):
        pass

    def update_state(self, state: dict[str, float] | None = None):
        if state is not None:
            self.state.update(state)

        if not self._add_units:
            self.state = remove_units(self.state)

    @staticmethod
    @abstractmethod
    def default_initial_conditions() -> dict[str, float]: ...

    def time_varying_elastance(self, EA, EB, tC, TC, TR, **kwargs):
        return time_varying_elastance.blanco_ventricle(
            EA=EA,
            EB=EB,
            tC=tC,
            TC=TC,
            TR=TR,
            THB=self.THB,
        )

    def flux_through_valve(self, p1, p2, R):
        return (p1 - p2) / R(p1, p2)

    def _R(
        self,
        Rmin: float,
        Rmax: float,
        unit_R: float = 1.0,
        unit_p: float = 1.0,
    ) -> Callable[[float, float], float]:
        return lambda w, v: unit_R * 10.0 ** (
            np.log10(Rmin / unit_R)
            + (np.log10(Rmax / unit_R) - np.log10(Rmin / unit_R))
            * smooth_heavyside((v - w) / unit_p)
        )

    @abstractmethod
    def step(self, t: float, dt: float) -> None: ...

    def solve(
        self,
        T: float | None = None,
        num_cycles: int | None = None,
        initial_state: dict[str, float] | None = None,
        dt: float = 1e-3,
        dt_eval: float | None = None,
        checkpoint: int = 0,
    ):
        logger.info("Running circulation model")
        if T is None:
            assert num_cycles is not None, "Please provide num_cycles or T"
            T = self.THB * num_cycles

        initial_state = initial_state or dict()

        if dt_eval is None:
            output_every_n_steps = 1
        else:
            output_every_n_steps = np.round(dt_eval / dt)

        if checkpoint > 0:
            checkoint_every_n_steps = np.round(checkpoint / dt)
        else:
            checkoint_every_n_steps = np.inf

        self.update_state(state=initial_state)
        self.initialize_output()
        t = 0.0
        if self._add_units:
            t *= units.ureg("s")
            dt *= units.ureg("s")

        self.store(t)

        time_start = time.time()

        i = 0
        while t < T:
            self.callback(self, t, i % output_every_n_steps == 0)
            self.step(t, dt)
            if i % output_every_n_steps == 0:
                self.store(t)
            if self._verbose:
                self.print_info()

            if i % checkoint_every_n_steps == 0:
                self.save_state()
            t += dt
            i += 1

        duration = time.time() - time_start

        logger.info(f"Done running circulation model in {duration:.2f} s")
        self.save_state()
        return self.results

    def initialize_output(self):
        self.results = defaultdict(list)

    def store(self, t):
        get = lambda x: float(np.copy(x)) if not self._add_units else x.magnitude

        self.results["time"].append(get(t))
        for k, v in self.state.items():
            self.results[k].append(get(v))
        for k, v in self.var.items():
            self.results[k].append(get(v))

    def save_state(self):
        self.callback_save_state(self)
        (self.outdir / "state.json").write_text(json.dumps(self.state, indent=2))
        (self.outdir / "results.json").write_text(json.dumps(self.results, indent=2))

    @property
    def volumes(self) -> dict[str, float]:
        return {}

    @property
    def pressures(self) -> dict[str, float]:
        return {}

    @property
    def flows(self) -> dict[str, float]:
        return {}

    def print_info(self):
        msg = []
        for attr, title in [
            (self.volumes, "Volumes"),
            (self.pressures, "Pressures"),
            (self.flows, "Flows"),
        ]:
            table = Table(title=title)
            row = []
            for k, v in attr.items():
                table.add_column(k)
                row.append(f"{v:.3f}")
            table.add_row(*row)
            msg.append(f"\n{log.log_table(table)}")
        logger.info("".join(msg))
