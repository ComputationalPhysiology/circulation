# # Regazzoni 2020

# In this example we will use the 0D model from {cite:p}`regazzoni2022cardiac` to simulate the cardiac cycle.

from circulation.regazzoni2020 import Regazzoni2020
import matplotlib.pyplot as plt

circulation = Regazzoni2020()

circulation.print_info()

history = circulation.solve(
    num_cycles=50,
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
plt.show()

# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
