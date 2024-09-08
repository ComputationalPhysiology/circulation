import numpy as np
import scipy
from circulation import bestel


def test_bestel_activation():
    activation = bestel.BestelActivation()
    t_span = (0.0, 1.0)
    t_eval = np.linspace(*t_span, 100)
    res = scipy.integrate.solve_ivp(
        activation,
        t_span,
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )

    assert len(res.y[0]) == 100
    assert np.isclose(res.y[0][0], 0.0)
    assert np.isclose(res.y[0].max(), 117671.77156237242)


def test_bestel_pressure():
    pressure = bestel.BestelPressure()

    t_span = (0.0, 1.0)
    t_eval = np.linspace(*t_span, 100)
    res = scipy.integrate.solve_ivp(
        pressure,
        t_span,
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )

    assert len(res.y[0]) == 100
    assert np.isclose(res.y[0][0], 0.0)
    assert np.isclose(res.y[0].max(), 16032.669867425086)
