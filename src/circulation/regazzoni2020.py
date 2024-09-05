from __future__ import annotations

from typing import Callable, Any

from . import base
from . import units

mL = units.ureg("mL")
mmHg = units.ureg("mmHg")
s = units.ureg("s")


class Regazzoni2020(base.CirculationModel):
    """
    Closed loop circulation model.

    References
    ----------
    F. Regazzoni, M. Salvador, P. C. Africa, M. Fedele, L. Dede', A. Quarteroni,
    "A cardiac electromechanics model coupled with a lumped parameters model for
    closed-loop blood circulation. Part I: model derivation", arXiv (2020)
    https://arxiv.org/abs/2011.15040

    """

    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        add_units=False,
        p_LV_func: Callable[[float, float], float] | None = None,
        leak: Callable[[float], float] | None = None,
        callback: Callable[[float], None] | None = None,
        verbose: bool = False,
    ):
        super().__init__(parameters, add_units=add_units, callback=callback)
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

        E_LA = self.time_varying_elastance(**chambers["LA"])
        self.p_LA_func = lambda V, t: E_LA(t) * (V - chambers["LA"]["V0"])

        if p_LV_func is not None:
            self.p_LV_func = p_LV_func
        else:
            # Use default time varying elastance model
            E_LV = self.time_varying_elastance(**chambers["LV"])
            self.p_LV_func = lambda V, t: E_LV(t) * (V - chambers["LV"]["V0"])

        E_RA = self.time_varying_elastance(**chambers["RA"])
        self.p_RA_func = lambda V, t: E_RA(t) * (V - chambers["RA"]["V0"])

        E_RV = self.time_varying_elastance(**chambers["RV"])
        self.p_RV_func = lambda V, t: E_RV(t) * (V - chambers["RV"]["V0"])

        if leak is not None:
            self.leak = leak
        else:
            self.leak = lambda t: 0.0

        self._initialize()

    @staticmethod
    def default_parameters() -> dict[str, Any]:
        return {
            "BPM": 75.0 * units.ureg("1/minutes"),
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
                    "C_VEN": 60.0 * mL / mmHg,
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

    def update_static_variables(self, t):
        self.var["p_LA"] = self.p_LA_func(self.state["V_LA"], t)
        self.var["p_LV"] = self.p_LV_func(self.state["V_LV"], t)
        self.var["p_RA"] = self.p_RA_func(self.state["V_RA"], t)
        self.var["p_RV"] = self.p_RV_func(self.state["V_RV"], t)
        self.var["Q_MV"] = self.flux_through_valve(self.var["p_LA"], self.var["p_LV"], self.R_MV)
        self.var["Q_AV"] = self.flux_through_valve(
            self.var["p_LV"], self.state["p_AR_SYS"], self.R_AV
        )
        self.var["Q_TV"] = self.flux_through_valve(self.var["p_RA"], self.var["p_RV"], self.R_TV)
        self.var["Q_PV"] = self.flux_through_valve(
            self.var["p_RV"], self.state["p_AR_PUL"], self.R_PV
        )

    def step(self, t, dt):
        self.update_static_variables(t)

        Q_VEN_PUL = self.state["Q_VEN_PUL"]
        Q_AR_PUL = self.state["Q_AR_PUL"]
        Q_VEN_SYS = self.state["Q_VEN_SYS"]
        Q_AR_SYS = self.state["Q_AR_SYS"]

        Q_MV = self.var["Q_MV"]
        Q_AV = self.var["Q_AV"]
        Q_TV = self.var["Q_TV"]
        Q_PV = self.var["Q_PV"]

        p_AR_SYS = self.state["p_AR_SYS"]
        p_VEN_SYS = self.state["p_VEN_SYS"]
        p_AR_PUL = self.state["p_AR_PUL"]
        p_VEN_PUL = self.state["p_VEN_PUL"]
        p_RA = self.var["p_RA"]
        p_LA = self.var["p_LA"]

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

        self.state["V_LA"] += dt * (Q_VEN_PUL - Q_MV - self.leak(t))
        self.state["V_LV"] += dt * (Q_MV - Q_AV)
        self.state["V_RA"] += dt * (Q_VEN_SYS - Q_TV)
        self.state["V_RV"] += dt * (Q_TV - Q_PV)
        self.state["p_AR_SYS"] += dt * (Q_AV - Q_AR_SYS) / C_AR_SYS
        self.state["p_VEN_SYS"] += dt * (Q_AR_SYS - Q_VEN_SYS) / C_VEN_SYS
        self.state["p_AR_PUL"] += dt * (Q_PV - Q_AR_PUL) / C_AR_PUL
        self.state["p_VEN_PUL"] += dt * (Q_AR_PUL - Q_VEN_PUL) / C_VEN_PUL
        self.state["Q_AR_SYS"] += -dt * ((R_AR_SYS * Q_AR_SYS + p_VEN_SYS - p_AR_SYS) / L_AR_SYS)
        self.state["Q_VEN_SYS"] += -dt * (R_VEN_SYS * Q_VEN_SYS + p_RA - p_VEN_SYS) / L_VEN_SYS
        self.state["Q_AR_PUL"] += -dt * (R_AR_PUL * Q_AR_PUL + p_VEN_PUL - p_AR_PUL) / L_AR_PUL
        self.state["Q_VEN_PUL"] += -dt * (R_VEN_PUL * Q_VEN_PUL + p_LA - p_VEN_PUL) / L_VEN_PUL

    @property
    def volumes(self):
        C_VEN_SYS = self.parameters["circulation"]["SYS"]["C_VEN"]
        C_AR_SYS = self.parameters["circulation"]["SYS"]["C_AR"]
        C_VEN_PUL = self.parameters["circulation"]["PUL"]["C_VEN"]
        C_AR_PUL = self.parameters["circulation"]["PUL"]["C_AR"]

        volumes = {
            "V_LA": self.state["V_LA"],
            "V_LV": self.state["V_LV"],
            "V_RA": self.state["V_RA"],
            "V_RV": self.state["V_RV"],
            "V_AR_SYS": C_AR_SYS * self.state["p_AR_SYS"],
            "V_VEN_SYS": C_VEN_SYS * self.state["p_VEN_SYS"],
            "V_AR_PUL": C_AR_PUL * self.state["p_AR_PUL"],
            "V_VEN_PUL": C_VEN_PUL * self.state["p_VEN_PUL"],
        }

        volumes["Heart"] = volumes["V_LA"] + volumes["V_LV"] + volumes["V_RA"] + volumes["V_RV"]
        volumes["SYS"] = volumes["V_AR_SYS"] + volumes["V_VEN_SYS"]
        volumes["PUL"] = volumes["V_AR_PUL"] + volumes["V_VEN_PUL"]
        volumes["Total"] = volumes["Heart"] + volumes["SYS"] + volumes["PUL"]

        return volumes

    @property
    def pressures(self) -> dict[str, float]:
        return {
            "p_LA": self.var["p_LA"],
            "p_LV": self.var["p_LV"],
            "p_RA": self.var["p_RA"],
            "p_RV": self.var["p_RV"],
            "p_AR_SYS": self.state["p_AR_SYS"],
            "p_VEN_SYS": self.state["p_VEN_SYS"],
            "p_AR_PUL": self.state["p_AR_PUL"],
            "p_VEN_PUL": self.state["p_VEN_PUL"],
        }

    @property
    def flows(self) -> dict[str, float]:
        return {
            "Q_MV": self.var["Q_MV"],
            "Q_AV": self.var["Q_AV"],
            "Q_TV": self.var["Q_TV"],
            "Q_PV": self.var["Q_PV"],
            "Q_AR_SYS": self.state["Q_AR_SYS"],
            "Q_VEN_SYS": self.state["Q_VEN_SYS"],
            "Q_AR_PUL": self.state["Q_AR_PUL"],
            "Q_VEN_PUL": self.state["Q_VEN_PUL"],
        }
