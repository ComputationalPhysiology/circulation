# # Regazzoni 2020 with dolfinx

from mpi4py import MPI
import dolfinx
import ufl
import scifem
from circulation.log import setup_logging
from circulation.regazzoni2020 import Regazzoni2020
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np

setup_logging()

comm = MPI.COMM_WORLD
domain = dolfinx.mesh.create_unit_square(comm, 5, 5)
circulation = Regazzoni2020()
y0 = circulation.state.copy()
V = scifem.create_real_functionspace(domain, value_shape=(len(y0),))
u_prev = dolfinx.fem.Function(V)
u = dolfinx.fem.Function(V)
u_prev.x.array[:] = y0
u.x.array[:] = y0
du_next = dolfinx.fem.Function(V)
du_prev = dolfinx.fem.Function(V)
du_next.x.array[:] = circulation.rhs(0.0, u_prev.x.array[:])

dt = dolfinx.fem.Constant(domain, 0.001)
t = dolfinx.fem.Constant(domain, 0.0)
theta = dolfinx.fem.Constant(domain, 1.0)
phi = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

F = ufl.inner((u - u_prev - dt*(theta * du_prev + (1 - theta) * du_next)), v) * ufl.dx

R = ufl.derivative(F, u, v)

forward_solver = scifem.NewtonSolver(
    [F], [[ufl.derivative(F, u, ufl.TrialFunction(V))]], [u]
)

time = np.arange(0, 10, dt.value)
y = np.zeros((len(y0), len(time)))
y[:, 0] = y0
t0 = perf_counter()
for i, ti in enumerate(time[1:]):
    t.value = ti
    du_next.x.array[:] = circulation.rhs(t.value, u_prev.x.array[:])
    forward_solver.solve()
    u_prev.x.array[:] = u.x.array[:]
    du_prev.x.array[:] = du_next.x.array[:]
    y[:, i + 1] = u.x.array[:]

t1 = perf_counter()

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))


state_names = circulation.state_names()
var_names = circulation.var_names()
vars = circulation.update_static_variables(time, y)

ax[0].plot(time, vars[var_names.index("p_LV"), :], label="p_LV (dolfinx)")
ax[0].plot(time, vars[var_names.index("p_LA"), :], label="p_LA (dolfinx)")
ax[0].plot(time, y[state_names.index("p_AR_SYS"), :], label="p_AR_SYS (dolfinx)")

ax[1].plot(time, y[state_names.index("V_LA"), :], label="V_LA (dolfinx)")
ax[1].plot(time, y[state_names.index("V_LV"), :], label="V_LV (dolfinx)")

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

print("Dolfinx solve time: ", t1 - t0)
print("Circulation solve time: ", t3 - t2)

fig.savefig("regazzoni2020_comp_dolfinx.png", dpi=300, bbox_inches="tight")
