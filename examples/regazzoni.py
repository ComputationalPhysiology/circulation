# # Regazzoni 2020

# In this example we will use the 0D model from {cite:p}`regazzoni2022cardiac` to simulate the cardiac cycle.

from circulation.regazzoni2020 import Regazzoni2020
import matplotlib.pyplot as plt

circulation = Regazzoni2020()

circulation.print_info()

history = circulation.solve(
    num_cycles=10,
)
circulation.print_info()

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
ax[0].plot(history["V_LV"], history["p_LV"])
ax[0].set_xlabel("V [mL]")
ax[0].set_ylabel("p [mmHg]")
ax[0].set_title("All beats")
ax[1].plot(history["V_LV"][-1000:], history["p_LV"][-1000:])
ax[1].set_title("Last beat")
ax[1].set_xlabel("V [mL]")


fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))
ax[0].plot(history["time"], history["p_LV"], label="p_LV")
ax[0].plot(history["time"], history["p_LA"], label="p_LA")
ax[0].plot(history["time"], history["p_AR_SYS"], label="p_AR_SYS")

ax[0].legend()
ax[1].plot(history["time"], history["V_LV"], label="V_LV")
ax[1].plot(history["time"], history["V_LA"], label="V_LA")
# ax[1].plot(history["time"], history["V_RA"], label="V_RA")
# ax[1].plot(history["time"], history["V_RV"], label="V_RV")
ax[1].legend()

plt.show()




# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
