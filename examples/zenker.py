from circulation.zenker import Zenker
import matplotlib.pyplot as plt
import numpy as np


circulation = Zenker()
history = circulation.solve(T=1000.0, dt=1e-3, dt_eval=0.1)
start_plot = 0
time = history["time"][start_plot:]

fig, ax = plt.subplots(5, 2, sharex=True, figsize=(10, 10))
ax[0, 0].plot(time, history["Vv"][start_plot:])
ax[0, 0].set_ylabel("Vv [mL]")
ax[1, 0].plot(time, history["Va"][start_plot:])
ax[1, 0].set_ylabel("Va [mL]")
ax[2, 0].plot(time, history["V_ED"][start_plot:], label="V_ED")
ax[2, 0].set_ylabel("V_ED [mL]")
ax[3, 0].plot(time, history["V_ES"][start_plot:], label="V_ES")
ax[3, 0].set_ylabel("V_ES [mL]")

SV = np.subtract(history["V_ED"],history["V_ES"])
ax[4, 0].plot(time, SV[start_plot:], label="SV")
ax[4, 0].set_ylabel("Stroke volume [mL]")

ax[0, 1].plot(time, history["fHR"][start_plot:])
ax[0, 1].set_ylabel("fHR [Hz]")
CO = SV * history["fHR"]
ax[1, 1].plot(time, CO[start_plot:], label="CO")
ax[1, 1].set_ylabel("Cardiac output [mL/s]")


ax[2, 1].plot(time, history["Pa"][start_plot:])
ax[2, 1].set_ylabel("Pa [mmHg]")
ax[3, 1].plot(time, history["S"][start_plot:])
ax[3, 1].set_ylabel("S")
ax[4, 1].plot(time, history["Pcvp"][start_plot:])
ax[4, 1].set_ylabel("Pcvp [mmHg]")

plt.show()
