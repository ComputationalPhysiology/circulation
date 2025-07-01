# # Regazzoni 2020 with scipy

# In this example we show how to use `scipy.integrate.solve_ivp` to solve the ODEs of the Regazzoni 2020 model.
# We also compare the results with the `circulation` package's built-in solver (which uses a forward euler scheme).

from circulation.log import setup_logging
from circulation.regazzoni2020 import Regazzoni2020
import matplotlib.pyplot as plt

setup_logging()
circulation = Regazzoni2020()

from time import perf_counter
from scipy.integrate import solve_ivp
import numpy as np

time = np.linspace(0, 10, 1000)
t0 = perf_counter()
res = solve_ivp(
    circulation.rhs,
    [0, 10],
    circulation.state,
    t_eval=time,
    method="RK45",
)
t1 = perf_counter()
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))

state_names = circulation.state_names()
var_names = circulation.var_names()
vars = circulation.update_static_variables(time, res.y)

ax[0].plot(time, vars[var_names.index("p_LV"), :], label="p_LV (numpy)")
ax[0].plot(time, vars[var_names.index("p_LA"), :], label="p_LA (numpy)")
ax[0].plot(time, res.y[state_names.index("p_AR_SYS"), :], label="p_AR_SYS (numpy)")

ax[1].plot(time, res.y[state_names.index("V_LA"), :], label="V_LA (numpy)")
ax[1].plot(time, res.y[state_names.index("V_LV"), :], label="V_LV (numpy)")

t2 = perf_counter()
history = circulation.solve(num_beats=10)
t3 = perf_counter()
circulation.print_info()

ax[0].plot(history["time"], history["p_LV"], linestyle="--", label="p_LV (orig)")
ax[0].plot(history["time"], history["p_LA"], linestyle="--", label="p_LA (orig)")
ax[0].plot(history["time"], history["p_AR_SYS"], linestyle="--", label="p_AR_SYS (orig)")
ax[0].legend()
ax[1].plot(history["time"], history["V_LV"], linestyle="--", label="V_LV (orig)")
ax[1].plot(history["time"], history["V_LA"], linestyle="--", label="V_LA (orig)")
ax[1].legend()

fig.savefig("regazzoni2020_comp.png", dpi=300, bbox_inches="tight")

print("IVP solve time: ", t1 - t0)
print("Circulation solve time: ", t3 - t2)
