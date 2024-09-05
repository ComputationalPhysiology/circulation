from __future__ import annotations
import numpy as np
from . import base
from . import units

mL = units.ureg("mL")
mmHg = units.ureg("mmHg")
s = units.ureg("s")


class Zenkur(base.CirculationModel):
    """
    0D model of the left ventricle only.

    """

    def __init__(self, parameters: dict[str, float] | None = None, add_units=False):
        super().__init__(parameters, add_units=add_units)
        self._initialize()

    @staticmethod
    def default_parameters():
        return {
            "kE_LV": 0.066 * mL**-1,
            "V_ED0": 7.14 * mL,
            "P0_LV": 2.03 * mmHg,
            "tau_Baro": 20.0 * s,
            "k_width": 0.1838 * mmHg**-1,
            "Pa_set": 70.0 * mmHg,
            "Ca": 4.0 * mL / mmHg,
            "Cv": 111.11 * mL / mmHg,
            "Va0": 700 * mL,
            "Vv0_min": 2_700 * mL,
            "Vv0_max": 3_100 * mL,
            "R_TPR_min": 0.5335 * mmHg * s / mL,
            "R_TPR_max": 2.134 * mmHg * s / mL,
            "T_sys": 4 / 15 * s,
            "f_HR_min": 2 / 3 * 1 / s,
            "f_HR_max": 3.0 * 1 / s,
            "R_valve": 0.0025 * mmHg * s / mL,
            "C_PRSW_min": 25.9 * mmHg,
            "C_PRSW_max": 103.8 * mmHg,
            "BPM": 75.0 * units.ureg("1/minutes"),
            "start_withdrawal": 200.0 * s,
            "end_withdrawal": 400.0 * s,
            "start_infusion": 500.0 * s,
            "end_infusion": 700.0 * s,
            "flow_withdrawal": -1.0 * mL / s,
            "flow_infusion": 1.0 * mL / s,
        }

    @staticmethod
    def default_initial_conditions() -> dict[str, float]:
        V0lv = 7.144 * mL
        totalVolume = 4825 * mL

        meanVvU = (2700 + 3100) / 2
        Va = totalVolume / (700 + meanVvU) * 700
        Vv = totalVolume / (700 + meanVvU) * meanVvU

        return {
            "V_ES": 2 * V0lv,
            "V_ED": 3 * V0lv,
            "S": 0.5 * units.ureg("dimensionless"),
            "Va": Va,
            "Vv": Vv,
        }

    def fHR(self, S):
        fHR_min = self.parameters["f_HR_min"]
        fHR_max = self.parameters["f_HR_max"]
        return fHR_min + (fHR_max - fHR_min) * S

    def R_TPR(self, S):
        R_TPR_min = self.parameters["R_TPR_min"]
        R_TPR_max = self.parameters["R_TPR_max"]
        return R_TPR_min + (R_TPR_max - R_TPR_min) * S

    def C_PRSW(self, S):
        C_PRSW_min = self.parameters["C_PRSW_min"]
        C_PRSW_max = self.parameters["C_PRSW_max"]
        return C_PRSW_min + (C_PRSW_max - C_PRSW_min) * S

    def Vv0(self, S):
        Vv0_min = self.parameters["Vv0_min"]
        Vv0_max = self.parameters["Vv0_max"]
        return Vv0_min + (Vv0_max - Vv0_min) * (1 - S)

    def update_static_variables(self, t):
        V_ES = self.state["V_ES"]
        V_ED = self.state["V_ED"]
        S = self.state["S"]
        Va = self.state["Va"]
        Vv = self.state["Vv"]

        Ca = self.parameters["Ca"]
        Cv = self.parameters["Cv"]
        Va0 = self.parameters["Va0"]
        Vv0 = self.Vv0(S)
        fHR = self.fHR(S)
        R_TPR = self.R_TPR(S)
        C_PRSW = self.C_PRSW(S)
        Pa = (Va - Va0) / Ca  # Eq 17
        Pcvp = (Vv - Vv0) / Cv  # Eq 17
        IC = (Pa - Pcvp) / R_TPR  # Eq 18
        ICO = fHR * (V_ED - V_ES)  # Eq 19

        self.var["fHR"] = fHR
        self.var["R_TPR"] = R_TPR
        self.var["C_PRSW"] = C_PRSW
        self.var["Vv0"] = Vv0

        self.var["Pa"] = Pa  # Eq 17
        self.var["Pcvp"] = Pcvp  # Eq 17
        self.var["IC"] = IC  # Eq 18
        self.var["ICO"] = ICO  # Eq 19

        if t > self.parameters["start_withdrawal"] and t < self.parameters["end_withdrawal"]:
            self.var["I_ext"] = self.parameters["flow_withdrawal"]
        elif t > self.parameters["start_infusion"] and t < self.parameters["end_infusion"]:
            self.var["I_ext"] = self.parameters["flow_infusion"]
        else:
            self.var["I_ext"] = 0.0

    def p_LV_func(self, V_LV, t=0.0):
        P0_LV = self.parameters["P0_LV"]
        kE_LV = self.parameters["kE_LV"]
        V_ED0 = self.parameters["V_ED0"]

        # Eq 7
        return P0_LV * (np.exp(kE_LV * (V_LV - V_ED0)) - 1)

    def V_ES(self, V_ED, C_PRSW, Pa, t):
        P_ED = self.p_LV_func(V_ED, t)
        V_ED0 = self.parameters["V_ED0"]

        if Pa > P_ED:
            return V_ED0
        # Eq 5
        return V_ED - ((C_PRSW * (V_ED - V_ED0)) / (Pa - P_ED))

    def V_ED(self, V_ES, fHR, Pcvp, t):
        # Eq 13
        p_LV_ES = self.p_LV_func(V_ES, t)
        T_sys = self.parameters["T_sys"]

        if Pcvp > p_LV_ES:
            # Eq 12
            return self.V((1 / fHR) - T_sys, Pcvp, V_ES=V_ES)
        else:
            return V_ES

    def V(self, t, Pcvp, V_ES):
        R_Valve = self.parameters["R_valve"]
        P0_LV = self.parameters["P0_LV"]
        kE_LV = self.parameters["kE_LV"]
        V_ED0 = self.parameters["V_ED0"]

        # Eq 9
        k1 = -(P0_LV / R_Valve) * np.exp(-kE_LV * V_ED0)
        k2 = kE_LV
        # REMOVE MINUS SIGN
        k3 = (Pcvp + P0_LV) / R_Valve

        # breakpoint()

        return -(1 / k2) * np.log(
            (k1 / k3) * (np.exp(-k2 * k3 * t) - 1) + np.exp(-k2 * (V_ES + k3 * t))
        )

    def step(self, t, dt):
        self.update_static_variables(t)

        V_ES = self.state["V_ES"]
        V_ED = self.state["V_ED"]
        S = self.state["S"]
        tau_Baro = self.parameters["tau_Baro"]
        k_width = self.parameters["k_width"]
        Pa_set = self.parameters["Pa_set"]

        fHR = self.var["fHR"]
        C_PRSW = self.var["C_PRSW"]
        Pa = self.var["Pa"]
        Pcvp = self.var["Pcvp"]
        IC = self.var["IC"]
        ICO = self.var["ICO"]
        I_ext = self.var["I_ext"]

        # Added minus sign to match the other code
        dVa_dt = -(IC - ICO)  # Eq 20

        dS_dt = (1 / tau_Baro) * (1 - (1 / (1 + np.exp(-k_width * (Pa - Pa_set)))) - S)

        self.state["Va"] += dt * dVa_dt
        self.state["Vv"] += dt * (-dVa_dt + I_ext)  # Eq 20
        self.state["V_ES"] += dt * (self.V_ES(V_ED, C_PRSW, Pa, t) - V_ES) * fHR
        self.state["V_ED"] += dt * (self.V_ED(V_ES, fHR, Pcvp, t) - V_ED) * fHR
        self.state["S"] += dt * dS_dt
