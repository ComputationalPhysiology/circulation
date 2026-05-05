from . import base
from . import units

mL = units.ureg("mL")
mmHg = units.ureg("mmHg")
s = units.ureg("s")


class ThreeElementWindkessel(base.CirculationModel):
    """
    A 0D model of the Left Ventricle coupled to a 3-Element Windkessel system.

    The 3-Element Windkessel (WK3) models the systemic arterial tree using:
    - R_c: Characteristic resistance (proximal aortic impedance)
    - C: Arterial compliance (vessel elasticity)
    - R_p: Peripheral resistance (microcirculation)
    """

    def __init__(self, parameters=None, p_LV_func=None, **kwargs):
        super().__init__(parameters, **kwargs)

        if p_LV_func is not None:
            self.p_LV_func = p_LV_func
            self._E_LV = lambda t: 1.0  # Dummy function, not used
        else:
            # Use default time-varying elastance model if no custom mechanics are provided
            self._E_LV = self.time_varying_elastance(**self.parameters["chambers"]["LV"])
            self.p_LV_func = lambda V, t: (
                self._E_LV(t) * (V - self.parameters["chambers"]["LV"]["V0"])
            )

        self._initialize()

    @property
    def HR(self) -> float:
        return self.parameters["HR"]

    @staticmethod
    def default_parameters():
        return {
            "HR": 1.25 * units.ureg("Hz"),  # Heart rate in Hz (75 BPM)
            "R_c": 0.05 * mmHg * s / mL,  # Characteristic resistance (mmHg*s/mL)
            "R_p": 1.05 * mmHg * s / mL,  # Peripheral resistance (mmHg*s/mL)
            "C": 1.3 * mL / mmHg,  # Arterial compliance (mL/mmHg)
            "P_venous": 8.0 * mmHg,  # Preload / Left Atrial Pressure (mmHg)
            "R_mitral": 0.01 * mmHg * s / mL,  # Mitral valve resistance
            "R_aortic": 0.01 * mmHg * s / mL,  # Aortic valve resistance
            "chambers": {
                "LV": {
                    "EA": 2.5 * mmHg / mL,  # Max elastance (mmHg/mL)
                    "EB": 0.08 * mmHg / mL,  # Min elastance (mmHg/mL)
                    "TC": 0.25 * s,  # Contraction duration (s)
                    "TR": 0.15 * s,  # Relaxation duration (s)
                    "tC": 0.0 * s,  # Time of contraction onset (s)
                    "V0": 15.0 * mL,  # Unstressed volume (mL)
                }
            },
        }

    @staticmethod
    def default_initial_conditions() -> dict:
        return {
            "V_LV": 140.0 * mL,  # Initial left ventricular volume
            "p_c": 80.0 * mmHg,  # Pressure in the arterial compliance capacitor
        }

    @staticmethod
    def state_names() -> list:
        return ["V_LV", "p_c"]

    @staticmethod
    def var_names() -> list:
        return ["p_LV", "p_ao", "Q_in", "Q_out", "Q_p"]

    def update_static_variables(self, t, y):
        V_LV = y[0]
        p_c = y[1]

        p_LV = self.p_LV_func(V_LV, t)

        P_venous = self.parameters["P_venous"]
        R_mitral = self.parameters["R_mitral"]
        R_aortic = self.parameters["R_aortic"]
        R_c = self.parameters["R_c"]
        R_p = self.parameters["R_p"]

        # 1. Mitral Flow (Diastolic Filling)
        Q_in = max(0.0, (P_venous - p_LV) / R_mitral)

        # 2. Aortic Flow (Systolic Ejection)
        # In a WK3 model, aortic pressure P_ao = p_c + Q_out * R_c
        # If the valve is open (p_LV > p_ao), we substitute P_ao and solve for Q_out
        if p_LV > p_c:
            Q_out = (p_LV - p_c) / (R_aortic + R_c)
        else:
            Q_out = 0.0

        # 3. Pressures and Peripheral Flow
        p_ao = p_c + Q_out * R_c
        Q_p = p_c / R_p  # Discharge to venous pool (assumed 0 mmHg pressure drop reference)

        var = self._get_var(t)
        var[0] = p_LV
        var[1] = p_ao
        var[2] = Q_in
        var[3] = Q_out
        var[4] = Q_p
        return var

    def rhs(self, t, y):
        var = self.update_static_variables(t, y)
        Q_in = var[2]
        Q_out = var[3]
        Q_p = var[4]
        C = self.parameters["C"]

        self.dy[0] = Q_in - Q_out
        self.dy[1] = (Q_out - Q_p) / C

        return self.dy
