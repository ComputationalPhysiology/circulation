# # Regazzoni 2020 with a bleeding event

# In this example we will use the 0D model from {cite:p}`regazzoni2022cardiac` to simulate the cardiac cycle with a bleeding event.
# We use the Zenker model to find the heart rate for normal conditions and then we simulate a bleeding event and compute the new heart rate.
# Simularly we also adjust the systemic resistance and the contractility of the heart chambers to simulate the effects of bleeding.

import numpy as np
from circulation.regazzoni2020 import Regazzoni2020
from circulation.zenker import Zenker
import matplotlib.pyplot as plt


# Run first Zenker to get the correct heart rate for normal conditions
zenker_normal = Zenker()
zenker_normal.solve(T=100.0, dt=1e-3, dt_eval=0.1, initial_state={"S": 0.2})
HR_normal = zenker_normal.results["fHR"][-1]
R_TPR_normal = zenker_normal.results["R_TPR"][-1]
C_PRSW_normal = zenker_normal.results["C_PRSW"][-1]

print(f"HR_normal = {HR_normal}, R_TPR_normal = {R_TPR_normal}, C_PRSW_normal = {C_PRSW_normal}")


# Now we will simulate a bleeding and compute a new heart rate
blood_loss_parameters = {"start_withdrawal": 1, "end_withdrawal": 2, "flow_withdrawal": -2000, "flow_infusion": 0}
zenker_bleed = Zenker(parameters=blood_loss_parameters)
zenker_bleed.solve(T=300.0, dt=1e-3, dt_eval=0.1)
HR_bleed = zenker_bleed.results["fHR"][-1]
R_TPR_bleed = zenker_bleed.results["R_TPR"][-1]
C_PRSW_bleed = zenker_bleed.results["C_PRSW"][-1]

print(f"HR_bleed = {HR_bleed}, R_TPR_bleed = {R_TPR_bleed}, C_PRSW_bleed = {C_PRSW_bleed}")

HR_factor = HR_bleed / HR_normal
R_TPR_factor = R_TPR_bleed / R_TPR_normal
C_PRSW_factor = C_PRSW_bleed / C_PRSW_normal

regazzoni_normal_parmeters = Regazzoni2020.default_parameters()
regazzoni_normal_parmeters["HR"] = HR_normal
regazzoni_normal = Regazzoni2020(parameters=regazzoni_normal_parmeters)
regazzoni_normal.print_info()


dt_eval = 0.01
regazzoni_normal.solve(num_cycles=20, dt_eval=dt_eval)
N_normal = int(regazzoni_normal.HR / dt_eval)

regazzoni_bleed_parmeters = Regazzoni2020.default_parameters()
regazzoni_bleed_parmeters["HR"] = HR_bleed
regazzoni_bleed_parmeters["circulation"]["SYS"]["R_AR"] *= R_TPR_factor
regazzoni_bleed_parmeters["circulation"]["SYS"]["R_VEN"] *= R_TPR_factor
for chamber in ["LA", "LV", "RA", "RV"]:
    regazzoni_bleed_parmeters["chambers"][chamber]["EA"] *= C_PRSW_factor
    regazzoni_bleed_parmeters["chambers"][chamber]["EB"] *= C_PRSW_factor

regazzoni_bleed_parmeters["circulation"]["external"] = blood_loss_parameters

regazzoni_bleed = Regazzoni2020(parameters=regazzoni_bleed_parmeters)
regazzoni_bleed.solve(num_cycles=20, initial_state=regazzoni_normal.state, dt_eval=dt_eval)
regazzoni_bleed.print_info()
N_bleed = int(regazzoni_bleed.HR / dt_eval)

V_LV_normal = regazzoni_normal.results["V_LV"][-N_normal:]
V_LV_ED = max(V_LV_normal)
V_LV_ES = min(V_LV_normal)
SV_normal = V_LV_ED - V_LV_ES
V_LV_bleed = regazzoni_bleed.results["V_LV"][-N_bleed:]
V_LV_ED = max(V_LV_bleed)
V_LV_ES = min(V_LV_bleed)
SV_bleed = V_LV_ED - V_LV_ES

print(f"SV_normal = {SV_normal}, SV_bleed = {SV_bleed}")



fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 5))
ax[0, 0].plot(regazzoni_normal.results["V_LV"], regazzoni_normal.results["p_LV"])
ax[0, 0].set_xlabel("V [mL]")
ax[0, 0].set_ylabel("p [mmHg]")
ax[0, 0].set_title("Normal")
ax[1, 0].plot(regazzoni_normal.results["V_LV"][-N_normal:], regazzoni_normal.results["p_LV"][-N_normal:])
ax[1, 0].set_xlabel("V [mL]")
ax[0, 1].plot(regazzoni_bleed.results["V_LV"], regazzoni_bleed.results["p_LV"])
ax[0, 1].set_xlabel("V [mL]")
ax[0, 1].set_ylabel("p [mmHg]")
ax[0, 1].set_title("Bleeding")
ax[1, 1].plot(regazzoni_bleed.results["V_LV"][-N_bleed:], regazzoni_bleed.results["p_LV"][-N_bleed:])
ax[1, 1].set_xlabel("V [mL]")
plt.show()



# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
