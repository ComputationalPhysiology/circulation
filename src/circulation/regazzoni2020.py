from __future__ import annotations

from typing import Callable, Any
from pathlib import Path

import numpy as np

from . import base
from . import units

mL = units.ureg("mL")
mmHg = units.ureg("mmHg")
s = units.ureg("s")


def list2array(lst):
    if hasattr(lst, "__len__"):
        # Turn the object into a numpy array
        return np.array(lst)
    else:
        return lst


class Regazzoni2020(base.CirculationModel):
    """
    Closed loop circulation model fom Regazzoni et al. [2]_.

    .. [2] F. Regazzoni, M. Salvador, P. C. Africa, M. Fedele, L. Dede', A. Quarteroni,
        "A cardiac electromechanics model coupled with a lumped parameters model for
        closed-loop blood circulation. Part I: model derivation", arXiv (2020)
        https://arxiv.org/abs/2011.15040

    Parameters
    ----------
    parameters : dict[str, Any] | None, optional
        Parameters used in the model, by default None which uses the default parameters
    p_LV_func : Callable[[float, float], float] | None, optional
        Optional function to calculate the pressure in the LV, by default None.
        The function should take the volume in the LV as the first argument and
        the time as the second argument, and return the pressure in the LV
    p_BiV_func : Callable[[float, float, float], float] | None, optional
        Optional function to calculate the pressure in the LV and RV, by default None.
        The function should take the volume in the LV as the first argument, the volume
        in the RV as the second argument, and the time as the third argument, and return
        a tuple (plv, prv) with the pressures in the LV and RV.
    add_units : bool, optional
        Add units to the parameters, by default False. Note that adding units
        will drastically slow down the simulation, so it is recommended to
        use this only for testing purposes.
    callback : base.CallBack | None, optional
        Optional callback function, by default None. The callback function takes
        three arguments: the model, the current time, and a boolean flag `save`
        which indicates if the current state should be saved.
    verbose : bool, optional
        Print additional information, by default False
    comm : mpi4py.MPI_InterComm optional
        MPI communicator, by default None
    outdir : Path, optional
        Output directory, by default Path("results-regazzoni")
    initial_state : dict[str, float] | None, optional
        Initial state of the model, by default None which uses the default initial state
    """

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        p_LV: Callable[[float, float], float] | None = None,
        p_RV: Callable[[float, float], float] | None = None,
        p_BiV: Callable[[float, float, float], tuple[float, float]] | None = None,
        add_units=False,
        callback: base.CallBack | None = None,
        verbose: bool = False,
        comm=None,
        outdir: Path = Path("results-regazzoni"),
        initial_state: dict[str, float] | None = None,
    ):
        super().__init__(
            parameters,
            add_units=add_units,
            callback=callback,
            verbose=verbose,
            comm=comm,
            outdir=outdir,
            initial_state=initial_state,
        )
        chambers = self.parameters["chambers"]
        valves = self.parameters["valves"]

        if self._add_units:
            unit_R = 1 * mmHg * s / mL
            unit_p = 1 * mmHg
        else:
            unit_R = 1
            unit_p = 1

        self.R_MV = self._R(
            valves["MV"]["Rmin"], valves["MV"]["Rmax"], unit_R=unit_R, unit_p=unit_p
        )
        self.R_AV = self._R(
            valves["AV"]["Rmin"], valves["AV"]["Rmax"], unit_R=unit_R, unit_p=unit_p
        )
        self.R_TV = self._R(
            valves["TV"]["Rmin"], valves["TV"]["Rmax"], unit_R=unit_R, unit_p=unit_p
        )
        self.R_PV = self._R(
            valves["PV"]["Rmin"], valves["PV"]["Rmax"], unit_R=unit_R, unit_p=unit_p
        )

        self._E_LA = self.time_varying_elastance(**chambers["LA"])
        self.p_LA = lambda V, t: self._E_LA(t) * (V - chambers["LA"]["V0"])

        # If p_BiV is provided, we use it to calculate the pressures in both LV and RV
        if p_BiV is not None:
            self.p_BiV = p_BiV
            self._E_LV = lambda t: 1.0
            self._E_RV = lambda t: 1.0

        else:
            if p_LV is not None:
                self.p_LV = p_LV
                self._E_LV = lambda t: 1.0  # Dummy function, not used
            else:
                # Use default time varying elastance model
                self._E_LV = self.time_varying_elastance(**chambers["LV"])
                self.p_LV = lambda V, t: self._E_LV(t) * (V - chambers["LV"]["V0"])

        self._E_RA = self.time_varying_elastance(**chambers["RA"])
        self.p_RA = lambda V, t: self._E_RA(t) * (V - chambers["RA"]["V0"])

        if p_RV is not None:
            self.p_RV = p_RV
            self._E_RV = lambda t: 1.0  # Dummy function, not used
        else:
            self._E_RV = self.time_varying_elastance(**chambers["RV"])
            self.p_RV = lambda V, t: self._E_RV(t) * (V - chambers["RV"]["V0"])

        self._initialize()

    @property
    def HR(self) -> float:
        return self.parameters["HR"]

    @staticmethod
    def default_parameters() -> dict[str, Any]:
        return {
            "HR": 1.0 * units.ureg("Hz"),
            "chambers": {
                "LA": {
                    "EA": 0.07 * mmHg / mL,
                    "EB": 0.09 * mmHg / mL,
                    "TC": 0.17 * s,
                    "TR": 0.17 * s,
                    "tC": 0.80 * s,
                    "V0": 4.0 * mL,
                },
                "LV": {
                    "EA": 2.75 * mmHg / mL,
                    "EB": 0.08 * mmHg / mL,
                    "TC": 0.34 * s,
                    "TR": 0.17 * s,
                    "tC": 0.00 * s,
                    "V0": 5.0 * mL,
                },
                "RA": {
                    "EA": 0.06 * mmHg / mL,
                    "EB": 0.07 * mmHg / mL,
                    "TC": 0.17 * s,
                    "TR": 0.17 * s,
                    "tC": 0.80 * s,
                    "V0": 4.0 * mL,
                },
                "RV": {
                    "EA": 0.55 * mmHg / mL,
                    "EB": 0.05 * mmHg / mL,
                    "TC": 0.34 * s,
                    "TR": 0.17 * s,
                    "tC": 0.00 * s,
                    "V0": 10.0 * mL,
                },
            },
            "valves": {
                "MV": {
                    "Rmin": 0.0075 * mmHg * s / mL,
                    "Rmax": 75006.2 * mmHg * s / mL,
                },
                "AV": {
                    "Rmin": 0.0075 * mmHg * s / mL,
                    "Rmax": 75006.2 * mmHg * s / mL,
                },
                "TV": {
                    "Rmin": 0.0075 * mmHg * s / mL,
                    "Rmax": 75006.2 * mmHg * s / mL,
                },
                "PV": {
                    "Rmin": 0.0075 * mmHg * s / mL,
                    "Rmax": 75006.2 * mmHg * s / mL,
                },
            },
            "circulation": {
                "SYS": {
                    "R_AR": 0.8 * mmHg * s / mL,
                    "C_AR": 1.2 * mL / mmHg,
                    "R_VEN": 0.26 * mmHg * s / mL,
                    "C_VEN": 130 * mL / mmHg,
                    "L_AR": 5e-3 * mmHg * s**2 / mL,
                    "L_VEN": 5e-4 * mmHg * s**2 / mL,
                },
                "PUL": {
                    "R_AR": 0.1625 * mmHg * s / mL,
                    "C_AR": 10.0 * mL / mmHg,
                    "R_VEN": 0.1625 * mmHg * s / mL,
                    "C_VEN": 16.0 * mL / mmHg,
                    "L_AR": 5e-4 * mmHg * s**2 / mL,
                    "L_VEN": 5e-4 * mmHg * s**2 / mL,
                },
                "external": {
                    "start_withdrawal": 0.0 * s,
                    "end_withdrawal": 0.0 * s,
                    "start_infusion": 0.0 * s,
                    "end_infusion": 0.0 * s,
                    "flow_withdrawal": 0.0 * mL / s,
                    "flow_infusion": 0.0 * mL / s,
                },
            },
        }

    @staticmethod
    def default_initial_conditions() -> dict[str, float]:
        return {
            "V_LA": 65.0 * mL,
            "V_LV": 120.0 * mL,
            "V_RA": 65.0 * mL,
            "V_RV": 145.0 * mL,
            "p_AR_SYS": 80.0 * mmHg,
            "p_VEN_SYS": 30.0 * mmHg,
            "p_AR_PUL": 35.0 * mmHg,
            "p_VEN_PUL": 24.0 * mmHg,
            "Q_AR_SYS": 0.0 * mL / s,
            "Q_VEN_SYS": 0.0 * mL / s,
            "Q_AR_PUL": 0.0 * mL / s,
            "Q_VEN_PUL": 0.0 * mL / s,
        }

    @staticmethod
    def var_names() -> list[str]:
        return ["p_LA", "p_LV", "p_RA", "p_RV", "Q_MV", "Q_AV", "Q_TV", "Q_PV", "I_ext"]

    @staticmethod
    def state_names():
        return [
            "V_LA",
            "V_LV",
            "V_RA",
            "V_RV",
            "p_AR_SYS",
            "p_VEN_SYS",
            "p_AR_PUL",
            "p_VEN_PUL",
            "Q_AR_SYS",
            "Q_VEN_SYS",
            "Q_AR_PUL",
            "Q_VEN_PUL",
        ]

    def update_static_variables(self, t, y):
        V_LA = y[0]
        V_LV = y[1]
        V_RA = y[2]
        V_RV = y[3]
        p_AR_SYS = y[4]
        # p_VEN_SYS = y[5]
        p_AR_PUL = y[6]
        # p_VEN_PUL = y[7]
        # Q_AR_SYS = y[8]
        # Q_VEN_SYS = y[9]
        # Q_AR_PUL = y[10]
        # Q_VEN_PUL = y[11]

        var = self._get_var(t)
        if hasattr(self, "p_BiV"):
            p_LV, p_RV = self.p_BiV(V_LV, V_RV, t)
        else:
            p_LV = self.p_LV(V_LV, t)
            p_RV = self.p_RV(V_RV, t)

        var[0] = self.p_LA(V_LA, t)
        var[1] = p_LV
        var[2] = self.p_RA(V_RA, t)
        var[3] = p_RV
        var[4] = self.flux_through_valve(var[0], var[1], self.R_MV)
        var[5] = self.flux_through_valve(var[1], p_AR_SYS, self.R_AV)
        var[6] = self.flux_through_valve(var[2], var[3], self.R_TV)
        var[7] = self.flux_through_valve(var[3], p_AR_PUL, self.R_PV)
        var[8] = base.external_blood(**self.parameters["circulation"]["external"], t=t)
        return var

    def jac(self, t, y):
        """
        Returns the Jacobian of the system of equations.
        The Jacobian is a 2D numpy array with shape (12, 12).
        """

        C_VEN_SYS = self.parameters["circulation"]["SYS"]["C_VEN"]
        C_AR_SYS = self.parameters["circulation"]["SYS"]["C_AR"]
        C_VEN_PUL = self.parameters["circulation"]["PUL"]["C_VEN"]
        C_AR_PUL = self.parameters["circulation"]["PUL"]["C_AR"]
        R_AR_SYS = self.parameters["circulation"]["SYS"]["R_AR"]
        R_VEN_SYS = self.parameters["circulation"]["SYS"]["R_VEN"]
        R_AR_PUL = self.parameters["circulation"]["PUL"]["R_AR"]
        R_VEN_PUL = self.parameters["circulation"]["PUL"]["R_VEN"]
        L_AR_SYS = self.parameters["circulation"]["SYS"]["L_AR"]
        L_VEN_SYS = self.parameters["circulation"]["SYS"]["L_VEN"]
        L_AR_PUL = self.parameters["circulation"]["PUL"]["L_AR"]
        L_VEN_PUL = self.parameters["circulation"]["PUL"]["L_VEN"]
        E_LA = self._E_LA(t)
        E_LV = self._E_LV(t)
        E_RA = self._E_RA(t)
        E_RV = self._E_RV(t)
        var = self._get_var(t)
        R_MV = self.R_MV(var[0], var[1])
        R_AV = self.R_AV(var[1], y[4])
        R_TV = self.R_TV(var[2], var[3])
        R_PV = self.R_PV(var[3], y[6])

        return np.array(
            [
                [-E_LA / R_MV, E_LV / R_MV, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [E_LA / R_MV, -E_LV / R_MV - E_LV / R_AV, 0, 0, 1 / R_AV, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -E_RA / R_TV, E_RV / R_TV, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, E_RA / R_TV, -E_RV / R_TV - E_RV / R_PV, 0, 0, 1 / R_PV, 0, 0, 0, 0, 0],
                [
                    0,
                    E_LV / (C_AR_SYS * R_AV),
                    0,
                    0,
                    -1 / (C_AR_SYS * R_AV),
                    0,
                    0,
                    0,
                    -1 / C_AR_SYS,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 0, 0, 1 / C_VEN_SYS, -1 / C_VEN_SYS, 0, 0],
                [
                    0,
                    0,
                    0,
                    E_RV / (C_AR_PUL * R_PV),
                    0,
                    0,
                    -1 / (C_AR_PUL * R_PV),
                    0,
                    0,
                    0,
                    -1 / C_AR_PUL,
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / C_VEN_PUL, -1 / C_VEN_PUL],
                [0, 0, 0, 0, 1 / L_AR_SYS, -1 / L_AR_SYS, 0, 0, -R_AR_SYS / L_AR_SYS, 0, 0, 0],
                [
                    0,
                    0,
                    -E_RA / L_VEN_SYS,
                    0,
                    0,
                    1 / L_VEN_SYS,
                    0,
                    0,
                    0,
                    -R_VEN_SYS / L_VEN_SYS,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 1 / L_AR_PUL, -1 / L_AR_PUL, 0, 0, -R_AR_PUL / L_AR_PUL, 0],
                [
                    -E_LA / L_VEN_PUL,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1 / L_VEN_PUL,
                    0,
                    0,
                    0,
                    -R_VEN_PUL / L_VEN_PUL,
                ],
            ]
        )

    def rhs(self, t, y):
        # V_LA = y[0]
        # V_LV = y[1]
        # V_RA = y[2]
        # V_RV = y[3]
        p_AR_SYS = y[4]
        p_VEN_SYS = y[5]
        p_AR_PUL = y[6]
        p_VEN_PUL = y[7]
        Q_AR_SYS = y[8]
        Q_VEN_SYS = y[9]
        Q_AR_PUL = y[10]
        Q_VEN_PUL = y[11]

        var = self.update_static_variables(t, y)

        p_LA = var[0]
        # p_LV = var[1]
        p_RA = var[2]
        # p_RV = var[3]
        Q_MV = var[4]
        Q_AV = var[5]
        Q_TV = var[6]
        Q_PV = var[7]
        I_ext = var[8]

        C_VEN_SYS = self.parameters["circulation"]["SYS"]["C_VEN"]
        C_AR_SYS = self.parameters["circulation"]["SYS"]["C_AR"]
        C_VEN_PUL = self.parameters["circulation"]["PUL"]["C_VEN"]
        C_AR_PUL = self.parameters["circulation"]["PUL"]["C_AR"]
        R_AR_SYS = self.parameters["circulation"]["SYS"]["R_AR"]
        R_VEN_SYS = self.parameters["circulation"]["SYS"]["R_VEN"]
        R_AR_PUL = self.parameters["circulation"]["PUL"]["R_AR"]
        R_VEN_PUL = self.parameters["circulation"]["PUL"]["R_VEN"]
        L_AR_SYS = self.parameters["circulation"]["SYS"]["L_AR"]
        L_VEN_SYS = self.parameters["circulation"]["SYS"]["L_VEN"]
        L_AR_PUL = self.parameters["circulation"]["PUL"]["L_AR"]
        L_VEN_PUL = self.parameters["circulation"]["PUL"]["L_VEN"]

        self.dy[0] = Q_VEN_PUL - Q_MV
        self.dy[1] = Q_MV - Q_AV
        self.dy[2] = Q_VEN_SYS - Q_TV
        self.dy[3] = Q_TV - Q_PV
        self.dy[4] = (Q_AV - Q_AR_SYS) / C_AR_SYS
        self.dy[5] = (Q_AR_SYS - Q_VEN_SYS + I_ext) / C_VEN_SYS
        self.dy[6] = (Q_PV - Q_AR_PUL) / C_AR_PUL
        self.dy[7] = (Q_AR_PUL - Q_VEN_PUL) / C_VEN_PUL
        self.dy[8] = -((R_AR_SYS * Q_AR_SYS + p_VEN_SYS - p_AR_SYS) / L_AR_SYS)
        self.dy[9] = -((R_VEN_SYS * Q_VEN_SYS + p_RA - p_VEN_SYS) / L_VEN_SYS)
        self.dy[10] = -(R_AR_PUL * Q_AR_PUL + p_VEN_PUL - p_AR_PUL) / L_AR_PUL
        self.dy[11] = -(R_VEN_PUL * Q_VEN_PUL + p_LA - p_VEN_PUL) / L_VEN_PUL
        return self.dy

    @property
    def volumes(self):
        return type(self).compute_volumes(self.parameters, self.state)

    @staticmethod
    def compute_volumes(parameters, state):
        C_VEN_SYS = parameters["circulation"]["SYS"]["C_VEN"]
        C_AR_SYS = parameters["circulation"]["SYS"]["C_AR"]
        C_VEN_PUL = parameters["circulation"]["PUL"]["C_VEN"]
        C_AR_PUL = parameters["circulation"]["PUL"]["C_AR"]

        volumes = {
            "V_LA": state[0],
            "V_LV": state[1],
            "V_RA": state[2],
            "V_RV": state[3],
            "V_AR_SYS": C_AR_SYS * state[4],
            "V_VEN_SYS": C_VEN_SYS * state[5],
            "V_AR_PUL": C_AR_PUL * state[6],
            "V_VEN_PUL": C_VEN_PUL * state[7],
        }

        volumes["Heart"] = volumes["V_LA"] + volumes["V_LV"] + volumes["V_RA"] + volumes["V_RV"]
        volumes["SYS"] = volumes["V_AR_SYS"] + volumes["V_VEN_SYS"]
        volumes["PUL"] = volumes["V_AR_PUL"] + volumes["V_VEN_PUL"]
        volumes["Total"] = volumes["Heart"] + volumes["SYS"] + volumes["PUL"]
        return volumes

    @property
    def pressures(self) -> dict[str, float]:
        return {
            "p_LA": self.var[0],
            "p_LV": self.var[1],
            "p_RA": self.var[2],
            "p_RV": self.var[3],
            "p_AR_SYS": self.state[4],
            "p_VEN_SYS": self.state[5],
            "p_AR_PUL": self.state[6],
            "p_VEN_PUL": self.state[7],
        }

    @property
    def flows(self) -> dict[str, float]:
        return {
            "Q_MV": self.var[4],
            "Q_AV": self.var[5],
            "Q_TV": self.var[6],
            "Q_PV": self.var[7],
            "Q_AR_SYS": self.state[8],
            "Q_VEN_SYS": self.state[9],
            "Q_AR_PUL": self.state[10],
            "Q_VEN_PUL": self.state[11],
        }
