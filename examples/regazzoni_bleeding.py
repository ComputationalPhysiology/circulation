# # Regazzoni 2020 with a bleeding event

# In this example we will use the 0D model from {cite:p}`regazzoni2022cardiac` to simulate the cardiac cycle with a bleeding event.
# We use the Zenker model to find the heart rate for normal conditions and then we simulate a bleeding event and compute the new heart rate.
# Simularly we also adjust the systemic resistance and the contractility of the heart chambers to simulate the effects of bleeding.

from circulation.log import setup_logging
from circulation.regazzoni2020 import Regazzoni2020
from circulation.zenker import Zenker
import matplotlib.pyplot as plt

setup_logging()



# Run first Zenker to get the correct heart rate for normal conditions
zenker_normal = Zenker()
zenker_normal.solve(T=100.0, dt=1e-3, dt_eval=0.1)
history_zenker_normal = zenker_normal.history
HR_normal = history_zenker_normal["fHR"][-1]
R_TPR_normal = history_zenker_normal["R_TPR"][-1]
C_PRSW_normal = history_zenker_normal["C_PRSW"][-1]

print(f"HR_normal = {HR_normal}, R_TPR_normal = {R_TPR_normal}, C_PRSW_normal = {C_PRSW_normal}")


# # Now we will simulate a bleeding and compute a new heart rate
blood_loss_parameters = {"start_withdrawal": 1, "end_withdrawal": 2, "flow_withdrawal": -2000, "flow_infusion": 0}
zenker_bleed = Zenker(parameters=blood_loss_parameters)
zenker_bleed.solve(T=300.0, dt=1e-3, dt_eval=0.1, initial_state=zenker_normal.state)
history_zenker_bleed = zenker_bleed.history
HR_bleed = history_zenker_bleed["fHR"][-1]
R_TPR_bleed = history_zenker_bleed["R_TPR"][-1]
C_PRSW_bleed = history_zenker_bleed["C_PRSW"][-1]

print(f"HR_bleed = {HR_bleed}, R_TPR_bleed = {R_TPR_bleed}, C_PRSW_bleed = {C_PRSW_bleed}")

HR_factor = HR_bleed / HR_normal
R_TPR_factor = R_TPR_bleed / R_TPR_normal
C_PRSW_factor = C_PRSW_bleed / C_PRSW_normal

regazzoni_normal_parmeters = Regazzoni2020.default_parameters()
regazzoni_normal_parmeters["HR"] = 1.0
regazzoni_normal = Regazzoni2020(parameters=regazzoni_normal_parmeters)
regazzoni_normal.print_info()


dt_eval = 0.01
regazzoni_normal.solve(num_beats=20, dt_eval=dt_eval)
N_normal = int(regazzoni_normal.HR / dt_eval)

regazzoni_bleed_parmeters = Regazzoni2020.default_parameters()
regazzoni_bleed_parmeters["HR"] = HR_factor
regazzoni_bleed_parmeters["circulation"]["SYS"]["R_AR"] *= R_TPR_factor
regazzoni_bleed_parmeters["circulation"]["SYS"]["R_VEN"] *= R_TPR_factor
for chamber in ["LA", "LV", "RA", "RV"]:
    regazzoni_bleed_parmeters["chambers"][chamber]["EA"] *= C_PRSW_factor
    regazzoni_bleed_parmeters["chambers"][chamber]["EB"] *= C_PRSW_factor

regazzoni_bleed_parmeters["circulation"]["external"] = blood_loss_parameters

regazzoni_bleed = Regazzoni2020(parameters=regazzoni_bleed_parmeters)
regazzoni_bleed.solve(num_beats=100, initial_state=regazzoni_normal.state, dt_eval=dt_eval)
regazzoni_bleed.print_info()
N_bleed = int(regazzoni_bleed.HR / dt_eval)
history_regazzoni_normal = regazzoni_normal.history
history_regazzoni_bleed = regazzoni_bleed.history

V_LV_normal = history_regazzoni_normal["V_LV"][-N_normal:]
V_LV_ED = max(V_LV_normal)
V_LV_ES = min(V_LV_normal)
SV_normal = V_LV_ED - V_LV_ES
V_LV_bleed = history_regazzoni_bleed["V_LV"][-N_bleed:]
V_LV_ED = max(V_LV_bleed)
V_LV_ES = min(V_LV_bleed)
SV_bleed = V_LV_ED - V_LV_ES

print(f"SV_normal = {SV_normal}, SV_bleed = {SV_bleed}")


fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 5))
ax[0, 0].set_title("Normal")
ax[0, 0].plot(history_regazzoni_normal["V_LV"], history_regazzoni_normal["p_LV"], label="LV")
ax[0, 0].plot(history_regazzoni_normal["V_RV"], history_regazzoni_normal["p_RV"], label="RV")
ax[1, 0].plot(history_regazzoni_normal["V_LV"][-N_normal:], history_regazzoni_normal["p_LV"][-N_normal:], label="LV")
ax[1, 0].plot(history_regazzoni_normal["V_RV"][-N_normal:], history_regazzoni_normal["p_RV"][-N_normal:], label="RV")
ax[0, 1].set_title("Bleeding")
ax[0, 1].plot(history_regazzoni_bleed["V_LV"], history_regazzoni_bleed["p_LV"], label="LV")
ax[0, 1].plot(history_regazzoni_bleed["V_RV"], history_regazzoni_bleed["p_RV"], label="RV")
ax[1, 1].plot(history_regazzoni_bleed["V_LV"][-N_bleed:], history_regazzoni_bleed["p_LV"][-N_bleed:], label="LV")
ax[1, 1].plot(history_regazzoni_bleed["V_RV"][-N_bleed:], history_regazzoni_bleed["p_RV"][-N_bleed:], label="RV")

ax[0, 0].set_ylabel("p [mmHg]")
ax[1, 0].set_ylabel("p [mmHg]")
ax[1, 0].set_xlabel("V [mL]")
ax[1, 1].set_xlabel("V [mL]")

for axi in ax.flatten():
    axi.grid()
    axi.legend()

pressure_keys = ['p_AR_SYS', 'p_VEN_SYS', 'p_AR_PUL', 'p_VEN_PUL', 'p_LV', 'p_RV', 'p_LA', 'p_RA']
flow_keys = ["Q_MV","Q_AV", "Q_TV", "Q_PV" ,"I_ext", "Q_AR_SYS", "Q_VEN_SYS" ,  "Q_AR_PUL",  "Q_VEN_PUL"]
for case, obj in [("normal",regazzoni_normal), ("bleed", regazzoni_bleed)]:
    history = obj.history
    fig, ax = plt.subplots(3, 3, sharex=True, figsize=(10, 8))
    for axi, key in zip(ax.flatten(), flow_keys):
        axi.plot(history["time"], history[key])
        axi.set_title(key)
    fig.suptitle(f"Flow {case}")

    fig, ax = plt.subplots(4, 2, sharex=True, figsize=(10, 8))
    for axi, key in zip(ax.flatten(), pressure_keys):
        axi.plot(history["time"], history[key])
        axi.set_title(key)
    fig.suptitle(f"Pressure {case}")

    volumes = Regazzoni2020.compute_volumes(obj.parameters, obj.results_state)

    fig, ax = plt.subplots(4, 3, sharex=True, figsize=(10, 8))
    for axi, (key, v) in zip(ax.flatten(), volumes.items()):
        axi.plot(history["time"], v)
        axi.set_title(key)
    fig.suptitle(f"Volumes {case}")


plt.show()




# # References
# ```{bibliography}
# :filter: docname in docnames
# ```
