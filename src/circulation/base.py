from __future__ import annotations
from typing import Callable, Any, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import json

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


def external_blood(
    t: float,
    start_withdrawal: float,
    end_withdrawal: float,
    start_infusion: float,
    end_infusion: float,
    flow_withdrawal: float,
    flow_infusion: float,
    **kwargs,
) -> float:
    if start_withdrawal <= t < end_withdrawal:
        return flow_withdrawal
    elif start_infusion <= t < end_infusion:
        return flow_infusion
    else:
        return 0.0


class CallBack(Protocol):
    def __call__(self, model: "CirculationModel", t: float = 0, save: bool = True) -> None: ...


def dummy_callback(model: "CirculationModel", t: float = 0, save: bool = True) -> None:
    pass


def recuursive_table(d: dict[str, Any], table: Table, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            recuursive_table(v, table, prefix=f"{prefix}.{k}")
        else:
            table.add_row(f"{prefix}.{k}".lstrip("."), str(v))


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class CirculationModel(ABC):
    """Base class for circulation models

    Parameters
    ----------
    parameters : dict[str, Any] | None, optional
        Parameters used in the model, by default None which uses the default parameters
    add_units : bool, optional
        Add units to the parameters, by default False. Note that adding units
        will drastically slow down the simulation, so it is recommended to
        use this only for testing purposes.
    callback : base.CallBack | None, optional
        Optional callback function which is called at every time step, by default None.
        The callback function take three arguments: the model, the current time,
        and a boolean flag `save` which indicates if the current state should be saved.
    verbose : bool, optional
        Print additional information, by default False
    comm : mpi4py.MPI_InterComm optional
        MPI communicator, by default None
    callback_save_state : base.CallBack | None, optional
        Optional callback function called every time the state should be saved, by default None.
        The function should take three arguments: the model, the current time, and a boolean
        flag `save` which indicates if the current state should be saved.
    initial_state : dict[str, float] | None, optional
        Initial state of the model, by default None which uses the default initial state
    """

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        outdir: Path = Path("results"),
        add_units: bool = False,
        callback: CallBack | None = None,
        verbose: bool = False,
        comm=None,
        callback_save_state: CallBack | None = None,
        initial_state: dict[str, float] | None = None,
        theta: float = 0.5,
    ):
        self.parameters = type(self).default_parameters()
        if parameters is not None:
            self.parameters = deep_update(self.parameters, parameters)

        self._add_units = add_units
        self._theta = theta

        self._initial_state = type(self).default_initial_conditions()
        self.update_inital_state(initial_state)

        table = Table(title=f"Circulation model parameters ({type(self).__name__})")
        table.add_column("Parameter")
        table.add_column("Value")
        recuursive_table(self.parameters, table)
        logger.info(f"\n{log.log_table(table)}")

        table = Table(title=f"Circulation model initial states ({type(self).__name__})")
        table.add_column("State")
        table.add_column("Value")
        recuursive_table(self._initial_state, table)
        logger.info(f"\n{log.log_table(table)}")

        if not add_units:
            self.parameters = remove_units(self.parameters)
        self.dy = np.zeros_like(self.state)

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

    def update_static_variables_external(self, t: float, y: np.ndarray) -> None: ...

    def update_inital_state(self, state: dict[str, float] | None = None):
        if state is not None:
            self._initial_state.update(state)

        self.state = np.array(list(remove_units(self._initial_state).values()), dtype=np.float64)
        self.state_old = np.copy(self.state)

    @property
    def states_names(self):
        return list(self._initial_state.keys())

    @property
    def num_states(self):
        return len(self.state)

    @staticmethod
    @abstractmethod
    def var_names():
        return []

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray) -> np.ndarray: ...

    @property
    def num_vars(self):
        return len(type(self).var_names())

    @property
    def state_theta(self):
        return self._theta * self.state + (1 - self._theta) * self.state_old

    def _initialize(self):
        self.var = np.zeros(self.num_vars, dtype=np.float64)
        if self._comm is None or (self._comm is not None and self._comm.rank == 0):
            # Dump parameters to file
            (self.outdir / "parameters.json").write_text(json.dumps(self.parameters, indent=2))
            # Dump initial conditions to file
            (self.outdir / "initial_conditions.json").write_text(
                json.dumps(remove_units(self._initial_state), indent=2)
            )

    def initialize_results(self, num_beats: int, dt_eval: float):
        self.times = np.arange(0, num_beats / self.HR + dt_eval, dt_eval)
        N = len(self.times)
        self.results_state = np.zeros((self.num_states, N))
        self.results_state[:, 0] = self.state

        self.rhs(0.0, self.state)

        self.results_var = np.zeros((self.num_vars, N))
        self.results_var[:, 0] = self.var

        self._index = 1

    @property
    @abstractmethod
    def HR(self) -> float:
        """Heart rate"""
        ...

    @staticmethod
    @abstractmethod
    def default_parameters() -> dict[str, Any]: ...

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
            RR=1 / self.HR,
        )

    def flux_through_valve(self, p1: float, p2: float, R: Callable[[float, float], float]) -> float:
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

    def times_one_beat(self, dt: float) -> np.ndarray:
        return np.arange(0, 1 / self.HR, dt)

    @abstractmethod
    def step(self, t: float, dt: float) -> None: ...

    def solve(
        self,
        num_beats: int = 1,
        initial_state: dict[str, float] | None = None,
        dt: float = 1e-3,
        dt_eval: float | None = None,
        checkpoint: int = 0,
    ):
        logger.info("Running circulation model")
        initial_state = initial_state or dict()

        if dt_eval is None:
            dt_eval = dt

        output_every_n_steps = np.round(dt_eval / dt)
        self.initialize_results(num_beats, dt_eval)

        if checkpoint > 0:
            checkoint_every_n_steps = np.round(checkpoint / dt)
        else:
            checkoint_every_n_steps = np.inf

        if initial_state is not None:
            self.update_inital_state(initial_state)

        t = 0.0
        if self._add_units:
            t *= units.ureg("s")
            dt *= units.ureg("s")

        time_start = time.time()

        for beat in range(num_beats):
            logger.info(f"Solving beat {beat}")
            for i, t in enumerate(self.times_one_beat(dt)):
                self.callback(self, t, False)
                self.step(t, dt)
                if i % output_every_n_steps == 0:
                    self.store()
                if self._verbose:
                    self.print_info()

                if i % checkoint_every_n_steps == 0:
                    self.save_state()

        duration = time.time() - time_start

        logger.info(f"Done running circulation model in {duration:.2f} s")
        self.save_state()

        return self.history

    @property
    def history(self):
        history = {}
        for i, name in enumerate(self.states_names):
            history[name] = self.results_state[i, :]
        for i, name in enumerate(type(self).var_names()):
            history[name] = self.results_var[i, :]
        history["time"] = self.times
        return history

    def store(self):
        self.results_state[:, self._index] = self.state[:]
        self.results_var[:, self._index] = self.var[:]
        self._index += 1

    def save_state(self):
        self.callback_save_state(self)

        np.savetxt(self.outdir / "state.txt", self.state)
        np.savetxt(self.outdir / "results_state.txt", self.results_state)
        np.savetxt(self.outdir / "results_var.txt", self.results_var)
        np.savetxt(self.outdir / "time.txt", self.times)
        np.savetxt(self.outdir / "var_names.txt", self.var_names(), fmt="%s")
        np.savetxt(self.outdir / "state_names.txt", self.states_names, fmt="%s")

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
