# %% [markdown]
# # The 3-Element Windkessel Circulation Model
#
# This tutorial demonstrates the 0D `ThreeElementWindkessel` model.
# It runs fully standalone using a simple Time-Varying Elastance (TVE)
# model for the Left Ventricle, requiring no external mechanics libraries.
#
# We will simulate two different preload states to observe the basic
# Frank-Starling mechanism (higher filling pressure -> larger stroke volume).

import logging
import matplotlib.pyplot as plt
import numpy as np
from circulation.windkessel import ThreeElementWindkessel
from circulation.units import ureg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    mmHg = ureg("mmHg")

    # Setup two models with different preloads, explicitly using units
    params_low = ThreeElementWindkessel.default_parameters()
    params_low["P_venous"] = 6.0 * mmHg
    model_low = ThreeElementWindkessel(parameters=params_low)

    params_high = ThreeElementWindkessel.default_parameters()
    params_high["P_venous"] = 12.0 * mmHg
    model_high = ThreeElementWindkessel(parameters=params_high)

    # Solve for 10 beats (add_units=False by default, so solve() strips units internally)
    hist_low = model_low.solve(num_beats=10, dt=1e-3)
    hist_high = model_high.solve(num_beats=10, dt=1e-3)

    # Extract the final stable beat
    # model_low.HR accesses the stripped float value (1.25)
    samples = int(1.0 / model_low.HR / 1e-3)
    slc = slice(-samples, None)
    t_plot = np.arange(samples) * 1e-3

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: PV Loops
    axs[0].plot(hist_low["V_LV"][slc], hist_low["p_LV"][slc], 'b-', lw=2.5, label="Low Preload (6 mmHg)")
    axs[0].plot(hist_high["V_LV"][slc], hist_high["p_LV"][slc], 'r-', lw=2.5, label="High Preload (12 mmHg)")
    axs[0].set_xlabel("Left Ventricular Volume (mL)", weight="bold")
    axs[0].set_ylabel("Pressure (mmHg)", weight="bold")
    axs[0].set_title("Pressure-Volume Loops", weight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Panel 2: Wiggers Diagram (High Preload)
    axs[1].plot(t_plot, hist_high["p_LV"][slc], 'r-', lw=2.5, label="LV Pressure")
    axs[1].plot(t_plot, hist_high["p_ao"][slc], 'k--', lw=2.0, label="Aortic Pressure")
    axs[1].plot(t_plot, hist_high["p_c"][slc], 'g:', lw=2.0, label="Capacitor Pressure (p_c)")
    axs[1].axhline(12.0, color='gray', linestyle=':', label="Atrial Pressure")
    axs[1].set_xlabel("Time (s)", weight="bold")
    axs[1].set_ylabel("Pressure (mmHg)", weight="bold")
    axs[1].set_title("Wiggers Diagram (High Preload)", weight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("windkessel_0D_demo.png", dpi=300)
    logging.info("Saved plot to windkessel_0D_demo.png")
    plt.show()

if __name__ == "__main__":
    main()
