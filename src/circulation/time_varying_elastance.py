import numpy as np


def blanco_ventricle(
    EA: float = 1.0,
    EB: float = 0.1,
    tC: float = 0.0,
    TC: float = 0.4,
    TR: float = 0.2,
    HR: float = 1.0,
):
    r"""
    Time-varying elastance model for the left ventricle.

    Parameters
    ----------
    EA : float, optional
        Maximum elastance, by default 1.0
    EB : float, optional
        Minimum elastance, by default 0.1
    tC : float, optional
        Time of contraction, by default 0.0 seconds
    TC : float, optional
        Duration of contraction, by default 0.4 seconds
    TR : float, optional
        Duration of relaxation, by default 0.2 seconds
    HR : float, optional
        Heart rate, by default 1.0

    Returns
    -------
    elastance : function
        Function that returns the elastance at time t.

    Notes
    -----
    The elastance is defined as

    .. math::
        E(t) = E_A \cdot f(t) + E_B

    where

    .. math::

        f(t) = \begin{cases}
            0.5 \left(1 - \cos\left(\frac{\pi}{T_C} (t - t_C)\right)\right)
            & 0 \leq t - t_C < T_C \\
            0.5 \left(1 + \cos\left(\frac{\pi}{T_R} (t - t_R)\right)\right)
            & 0 \leq t - t_R < T_R
        \end{cases}

    References
    ----------
    Blanco, P. J., & Feijóo, R. A. (2010). A 3D-1D-0D computational model
    for the entire cardiovascular system. Mecánica Computacional, 29(59), 5887-5911.

    """

    time_R = tC + TC
    time_rest = time_R + TR

    # tC <= t <= tC + TC - Contraction
    case1 = lambda t: (0 <= np.mod(t - tC, HR)) * (np.mod(t - tC, HR) < TC)
    # tC + TC <= t <= tC + TC + TR  - Relaxation
    case2 = lambda t: (0 <= np.mod(t - time_R, HR)) * (np.mod(t - time_R, HR) < TR)
    # tC + TC + TR <= t <= T - Rest
    case3 = lambda t: 0 <= np.mod(t - time_rest, HR)

    f_contr = lambda t: 0.5 * (1 - np.cos(np.pi / TC * (np.mod(t - tC, HR))))
    f_relax = lambda t: 0.5 * (1 + np.cos(np.pi / TR * (np.mod(t - time_R, HR))))
    f_rest = lambda t: 0

    e = lambda t: f_contr(t) * case1(t) + f_relax(t) * case2(t) + f_rest(t) * case3(t)

    return lambda t: EA * np.clip(e(t), 0.0, 1.0) + EB
