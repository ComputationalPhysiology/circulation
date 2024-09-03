# # Regazzoni 2020

# In this example we will use the 0D model from Regazzoni et al. [1]_ to simulate the cardiac cycle.

from circulation.regazzoni2020 import Regazzoni2020
import matplotlib.pyplot as plt

circulation = Regazzoni2020()

circulation.print_info()
history = circulation.solve(
    num_cycles=50,
)
circulation.print_info()

fig, ax = plt.subplots(2, 1)
# ax.plot(data_loop.volume, data_loop.pressure, label="3D")
ax[0].plot(history["V_LV"][-20_000:], history["p_LV"][-20_000:], label="0D")
ax[0].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False)
ax[0].set_xlabel("V [mL]")
ax[0].set_ylabel("p [mmHg]")

ax[1].plot(history["V_LV"][-1000:], history["p_LV"][-1000:], label="0D")

plt.show()
