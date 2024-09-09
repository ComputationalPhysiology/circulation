# # Bestel model

# In this example we will show the both the pressure and activation model from Bestel et al. {cite}`bestel2001biomechanical`.

import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from circulation import bestel, log

# First let us define a time array

log.setup_logging(level=logging.INFO)
t_eval = np.linspace(0, 1, 200)

# Now we will solve the activation model

activation = bestel.BestelActivation()
result_activation = solve_ivp(
        activation,
        [0, 1],
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )

# and plot the results

fig, ax = plt.subplots()
ax.plot(result_activation.t, result_activation.y[0])
ax.set_xlabel("Time [s]")
ax.set_ylabel("Active tension [Pa]")
plt.show()

# Now we will solve the pressure model

pressure = bestel.BestelPressure()
result_pressure = solve_ivp(
        pressure,
        [0, 1],
        [0.0],
        t_eval=t_eval,
        method="Radau",
    )

# and plot the results

fig, ax = plt.subplots()
ax.plot(result_pressure.t, result_pressure.y[0])
ax.set_xlabel("Time [s]")
ax.set_ylabel("Pressure [Pa]")
plt.show()

# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
