from circulation.zenkur import Zenkur
import matplotlib.pyplot as plt


circulation = Zenkur()
history = circulation.solve(T=3600.0, dt=1e-3)

fig, ax = plt.subplots(2, 1)
ax[0].plot(history["V_LV"], history["p_LV"], label="0D")
ax[1].plot(history["V_LV"][-1000:], history["p_LV"][-1000:], label="0D")
ax[0].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False)
ax[0].set_xlabel("V [mL]")
ax[0].set_ylabel("p [mmHg]")

plt.show()
