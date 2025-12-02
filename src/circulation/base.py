from __future__ import annotations
from typing import Callable, Any, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import json

import numpy as np
import numpy.typing as npt
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
    return np.where(
        np.logical_and(start_withdrawal <= t, t < end_withdrawal),
        flow_withdrawal,
        np.where(np.logical_and(start_infusion <= t, t < end_infusion), flow_infusion, 0.0),
    )


class CallBack(Protocol):
    def __call__(self, model: "CirculationModel", i: int = 0, t: float = 0, **kwargs) -> None: ...


def dummy_callback(model: "CirculationModel", i: int = 0, t: float = 0, **kwargs) -> None:
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
        reset_callback: CallBack | None = None,
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

        if reset_callback is not None:
            assert callable(reset_callback), "reset_callback must be callable"

            self.reset_callback = reset_callback
        else:
            self.reset_callback = dummy_callback

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
    def rhs(self, t: float, y: npt.NDArray) -> npt.NDArray: ...

    @property
    def num_vars(self):
        return len(type(self).var_names())

    @property
    def state_theta(self):
        return self._theta * self.state + (1 - self._theta) * self.state_old

    def _initialize(self):
        self.var = np.zeros(self.num_vars, dtype=np.float64)
        self.var_old = np.copy(self.var)
        if self._comm is None or (self._comm is not None and self._comm.rank == 0):
            # Dump parameters to file
            (self.outdir / "parameters.json").write_text(
                json.dumps(remove_units(self.parameters), indent=2)
            )
            # Dump initial conditions to file
            (self.outdir / "initial_conditions.json").write_text(
                json.dumps(remove_units(self._initial_state), indent=2)
            )

    def initialize_results(self):
        N = len(self.times_eval)
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

    def times_n_beats(self, dt: float, n: int = 1) -> np.ndarray:
        return np.arange(0, n / self.HR, dt)

    # def step(self, t, dt):
    #     dy = self.rhs(t, self.state)
    #     self.state += dt * dy

    def _get_var(self, t):
        try:
            float(t)  # type: ignore[arg-type]
        except TypeError:
            var = np.zeros((len(self.var), len(t)), dtype=float)  # type: ignore[arg-type]
        else:
            var = self.var
        return var

    def solve(
        self,
        num_beats: int | None = None,
        T: float | None = None,
        initial_state: dict[str, float] | None = None,
        dt: float = 1e-3,
        dt_eval: float | None = None,
        checkpoint: int = 0,
        method: str = "forward_euler",
        max_failures: int = 10,
    ):
        if dt_eval is None:
            dt_eval = dt
        output_every_n_steps = int(np.round(dt_eval / dt))

        if num_beats is None and T is None:
            raise ValueError("Need to specify either number of beats or total time")

        elif T is not None:
            # Choose T first, then num_beats
            num_beats = 1
            times_one_beat = np.arange(0, T, dt)

        else:
            assert num_beats is not None, "If T is None, num_beats must be specified"
            if T is not None:
                logger.warning("Ignoring T, using num_beats instead")
            times_one_beat = self.times_n_beats(dt, n=1)
            T = (times_one_beat[-1] + dt) * num_beats

        N = (
            sum(
                1
                for _ in range(num_beats)
                for i in range(len(times_one_beat))
                if i % output_every_n_steps == 0
            )
            + 1
        )

        self.times_eval = np.linspace(0, T, N)
        logger.info("Running circulation model")

        if initial_state is not None:
            if isinstance(initial_state, (list, np.ndarray, tuple)):
                initial_state = dict(zip(self.states_names, initial_state))
            else:
                assert isinstance(initial_state, dict), (
                    "initial_state must be a dict or convertible to one"
                )
            self.update_inital_state(initial_state)

        self.initialize_results()
        if checkpoint > 0:
            checkoint_every_n_steps = np.round(checkpoint / dt)
        else:
            checkoint_every_n_steps = np.inf

        t = 0.0
        if self._add_units:
            t *= units.ureg("s")
            dt *= units.ureg("s")

        time_start = time.time()

        for beat in range(num_beats):
            logger.info(f"Solving beat {beat}")
            for i, t in enumerate(times_one_beat):
                converged = False
                num_failures = 0
                ti = t
                dti = dt

                while not converged:
                    self.state_old[:] = self.state.copy()

                    try:
                        if method == "forward_euler":
                            dy = self.rhs(ti, self.state)
                            self.state += dti * dy

                        elif method == "backward_euler":
                            from scipy.optimize import root

                            old_state = np.copy(self.state)

                            def f(s):
                                return s - old_state - dt * self.rhs(t, s)

                            res = root(f, old_state)
                            self.state[:] = res.x
                        else:
                            from scipy.integrate import solve_ivp

                            jac = None
                            if hasattr(self, "jac"):
                                jac = self.jac

                            res = solve_ivp(
                                self.rhs,
                                [t, t + dt],
                                self.state,
                                t_eval=[t + dt],
                                method=method,
                                jac=jac,
                            )
                            self.state[:] = res.y[:, -1]
                    except RuntimeError:
                        logger.warning(
                            f"RuntimeError at beat {beat}, step {i}, time {ti:.3f} s, "
                            "trying to reduce time step"
                        )
                        num_failures += 1
                        self.state[:] = self.state_old.copy()
                        self.reset_callback(self, i, ti)
                        dti *= 0.5
                        ti = t - dti

                        if num_failures > max_failures:
                            logger.error(
                                f"Failed to converge after {num_failures} attempts at "
                                f"beat {beat}, step {i}, time {ti:.3f} s, "
                                "reducing time step too much, aborting"
                            )
                            raise RuntimeError(
                                "Failed to converge after too many attempts, "
                                "reducing time step too much"
                            )
                    else:
                        converged = True
                        ti += dti

                        # if self._comm is not None:
                        #     sol = self._comm.bcast(sol, root=0)
                        # self.state[:] = sol

                self.callback(
                    self,
                    beat * len(times_one_beat) + i,
                    beat * times_one_beat[-1] + t,
                )
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
        history["time"] = self.times_eval
        return history

    def store(self):
        try:
            self.results_state[:, self._index] = self.state[:]
        except IndexError:
            logger.warning(
                "IndexError when storing results, this is likely due that the "
                "HR and times for evaluations are not divisible by dt_eval. "
                "This will result in a loss of data. Please check your "
                "parameters."
            )
        else:
            self.results_var[:, self._index] = self.var[:]
        finally:
            self._index += 1

    def save_state(self):
        self.callback_save_state(self)

        np.savetxt(self.outdir / "state.txt", self.state)
        np.savetxt(self.outdir / "results_state.txt", self.results_state)
        np.savetxt(self.outdir / "results_var.txt", self.results_var)
        np.savetxt(self.outdir / "time.txt", self.times_eval)
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
