"""
Bestel model for active stress and pressure {cite}`bestel2001biomechanical
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from rich.table import Table

from . import log


logger = logging.getLogger(__name__)


@dataclass
class BestelActivation:
    r"""Active stress model from the Bestel model [3]_.

    Parameters
    ----------
    t_span : tuple[float, float]
        A tuple representing start and end of time
    parameters : dict[str, float]
        Parameters used in the model, see :func:`default_parameters`
    t_eval : np.ndarray, optional
        Time points to evaluate the solution, by default None.
        If not provided, the default points from `scipy.integrate.solve_ivp`
        will be used

    Returns
    -------
    np.ndarray
        An array of activation points

    Notes
    -----
    The active stress is taken from Bestel et al. [3]_, characterized through
    a time-dependent stress function :math:`\tau` solution to the evolution equation

    .. math::
        \dot{\tau}(t) = -|a(t)|\tau(t) + \sigma_0|a(t)|_+

    with :math:`a(\cdot)` being the activation function and \sigma_0 contractility,
    where each remaining term is described below:

    .. math::
        |a(t)|_+ =& \mathrm{max}\{a(t), 0\} \\
        a(t) :=& \alpha_{\mathrm{max}} \cdot f(t)
        + \alpha_{\mathrm{min}} \cdot (1 - f(t)) \\
        f(t) =& S^+(t - t_{\mathrm{sys}}) \cdot S^-(t - t_{\mathrm{dias}}) \\
        S^{\pm}(\Delta t) =& \frac{1}{2}(1 \pm \mathrm{tanh}(\frac{\Delta t}{\gamma}))

    .. [3] J. Bestel, F. Clement, and M. Sorine. "A Biomechanical Model of Muscle Contraction.
        In: Medical Image Computing and Computer-Assisted Intervention - MICCAI 2001. Springer
        Berlin Heidelberg, 2001, pp. 1159{1161.

    """

    parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters
        table = Table(title="Bestel activation model parameters")
        table.add_column("Parameter")
        table.add_column("Value")
        for k, v in parameters.items():
            table.add_row(k, str(v))
        logger.info(f"\n{log.log_table(table)}")

    @staticmethod
    def default_parameters() -> dict[str, float]:
        r"""Default parameters for the activation model

        Returns
        -------
        dict[str, float]
            Default parameters

        Notes
        -----
        The default parameters are

        .. math::
            t_{\mathrm{sys}} &= 0.16 \\
            t_{\mathrm{dias}} &= 0.484 \\
            \gamma &= 0.005 \\
            a_{\mathrm{max}} &= 5.0 \\
            a_{\mathrm{min}} &= -30.0 \\
            \sigma_0 &= 150e3 \\
        """
        return dict(
            t_sys=0.16,
            t_dias=0.484,
            gamma=0.005,
            a_max=5.0,
            a_min=-30.0,
            sigma_0=150e3,
        )

    def __call__(self, t, tau):
        ps = self.parameters

        # print(f"Solving active stress model with parameters: {pprint.pformat(params)}")

        f = (
            lambda t: 0.25
            * (1 + math.tanh((t - ps["t_sys"]) / ps["gamma"]))
            * (1 - math.tanh((t - ps["t_dias"]) / ps["gamma"]))
        )
        a = lambda t: ps["a_max"] * f(t) + ps["a_min"] * (1 - f(t))

        return -abs(a(t)) * tau + ps["sigma_0"] * max(a(t), 0)


@dataclass
class BestelPressure:
    r"""Time-dependent pressure derived from the Bestel model [3]_.

    Parameters
    ----------
    t_span : tuple[float, float]
        A tuple representing start and end of time
    parameters : dict[str, float]
        Parameters used in the model, see :func:`default_parameters`
    t_eval : np.ndarray, optional
        Time points to evaluate the solution, by default None.
        If not provided, the default points from `scipy.integrate.solve_ivp`
        will be used

    Returns
    -------
    np.ndarray
        An array of pressure points

    Notes
    -----
    We consider a time-dependent pressure derived from the Bestel model.
    The solution :math:`p = p(t)` is characterized as solution to the evolution equation

    .. math::
        \dot{p}(t) = -|b(t)|p(t) + \sigma_{\mathrm{mid}}|b(t)|_+
        + \sigma_{\mathrm{pre}}|g_{\mathrm{pre}}(t)|

    with :math:`b(\cdot)` being the activation function described below:

    .. math::
        b(t) =& a_{\mathrm{pre}}(t) + \alpha_{\mathrm{pre}}g_{\mathrm{pre}}(t)
        + \alpha_{\mathrm{mid}} \\
        a_{\mathrm{pre}}(t) :=& \alpha_{\mathrm{max}} \cdot f_{\mathrm{pre}}(t)
        + \alpha_{\mathrm{min}} \cdot (1 - f_{\mathrm{pre}}(t)) \\
        f_{\mathrm{pre}}(t) =& S^+(t - t_{\mathrm{sys}-\mathrm{pre}}) \cdot
         S^-(t  t_{\mathrm{dias} - \mathrm{pre}}) \\
        g_{\mathrm{pre}}(t) =& S^-(t - t_{\mathrm{dias} - \mathrm{pre}})

    with :math:`S^{\pm}` given by

    .. math::
        S^{\pm}(\Delta t) = \frac{1}{2}(1 \pm \mathrm{tanh}(\frac{\Delta t}{\gamma}))

    """

    parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        parameters = type(self).default_parameters()
        parameters.update(self.parameters)
        self.parameters = parameters
        table = Table(title="Bestel pressure model parameters")
        table.add_column("Parameter")
        table.add_column("Value")
        for k, v in parameters.items():
            table.add_row(k, str(v))
        logger.info(f"\n{log.log_table(table)}")

    @staticmethod
    def default_parameters() -> dict[str, float]:
        r"""Default parameters for the pressure model for LV only

        Returns
        -------
        dict[str, float]
            Default parameters

        Notes
        -----
        The default parameters are

        .. math::
            t_{\mathrm{sys} - \mathrm{pre}} &= 0.17 \\
            t_{\mathrm{dias} - \mathrm{pre}} &= 0.484 \\
            \gamma &= 0.005 \\
            a_{\mathrm{max}} &= 5.0 \\
            a_{\mathrm{min}} &= -30.0 \\
            \alpha_{\mathrm{pre}} &= 5.0 \\
            \alpha_{\mathrm{mid}} &= 1.0 \\
            \sigma_{\mathrm{pre}} &= 7000.0 \\
            \sigma_{\mathrm{mid}} &= 16000.0 \\
        """
        return dict(
            t_sys_pre=0.17,
            t_dias_pre=0.484,
            gamma=0.005,
            a_max=5.0,
            a_min=-30.0,
            alpha_pre=5.0,
            alpha_mid=1.0,
            sigma_pre=7000.0,
            sigma_mid=16000.0,
        )

    @staticmethod
    def default_lv_parameters() -> dict[str, float]:
        r"""Default parameters for the LV pressure model in BiV model

        Returns
        -------
        dict[str, float]
            Default parameters

        Notes
        -----
        The default parameters are

        .. math::
            t_{\mathrm{sys} - \mathrm{pre}} &= 0.17 \\
            t_{\mathrm{dias} - \mathrm{pre}} &= 0.484 \\
            \gamma &= 0.005 \\
            a_{\mathrm{max}} &= 5.0 \\
            a_{\mathrm{min}} &= -30.0 \\
            \alpha_{\mathrm{pre}} &= 5.0 \\
            \alpha_{\mathrm{mid}} &= 15.0 \\
            \sigma_{\mathrm{pre}} &= 12000.0 \\
            \sigma_{\mathrm{mid}} &= 16000.0 \\
        """
        return dict(
            t_sys_pre=0.17,
            t_dias_pre=0.484,
            gamma=0.005,
            a_max=5.0,
            a_min=-30.0,
            alpha_pre=5.0,
            alpha_mid=15.0,
            sigma_pre=12000.0,
            sigma_mid=16000.0,
        )

    @staticmethod
    def default_rv_parameters() -> dict[str, float]:
        r"""Default parameters for the RV pressure model in BiV model

        Returns
        -------
        Dict[str, float]
            Default parameters

        Notes
        -----
        The default parameters are

        .. math::
            t_{\mathrm{sys} - \mathrm{pre}} &= 0.17 \\
            t_{\mathrm{dias} - \mathrm{pre}} &= 0.484 \\
            \gamma &= 0.005 \\
            a_{\mathrm{max}} &= 5.0 \\
            a_{\mathrm{min}} &= -30.0 \\
            \alpha_{\mathrm{pre}} &= 5.0 \\
            \alpha_{\mathrm{mid}} &= 10.0 \\
            \sigma_{\mathrm{pre}} &= 3000.0 \\
            \sigma_{\mathrm{mid}} &= 4000.0 \\
        """
        return dict(
            t_sys_pre=0.17,
            t_dias_pre=0.484,
            gamma=0.005,
            a_max=5.0,
            a_min=-30.0,
            alpha_pre=1.0,
            alpha_mid=10.0,
            sigma_pre=3000.0,
            sigma_mid=4000.0,
        )

    def __call__(self, t, p):
        ps = self.parameters
        f = (
            lambda t: 0.25
            * (1 + math.tanh((t - ps["t_sys_pre"]) / ps["gamma"]))
            * (1 - math.tanh((t - ps["t_dias_pre"]) / ps["gamma"]))
        )
        a = lambda t: ps["a_max"] * f(t) + ps["a_min"] * (1 - f(t))

        f_pre = lambda t: 0.5 * (1 - math.tanh((t - ps["t_dias_pre"]) / ps["gamma"]))
        b = lambda t: a(t) + ps["alpha_pre"] * f_pre(t) + ps["alpha_mid"]

        return -abs(b(t)) * p + ps["sigma_mid"] * max(b(t), 0) + ps["sigma_pre"] * max(f_pre(t), 0)
