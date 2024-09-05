from pathlib import Path
from mpi4py import MPI
import dolfinx.log
import dolfinx
import numpy as np
import math
import cardiac_geometries
import fenicsx_pulse
import ufl

import logging

import math
from pathlib import Path

import matplotlib.pyplot as plt
from functools import lru_cache

import circulation.log as log
from circulation.regazzoni2020 import Regazzoni2020
from circulation.units import ureg, kPa_to_mmHg
from circulation import bestel

logger = logging.getLogger(__name__)

THB = 1.0
dt = 1e-3



def print_table(time, current_volume, target_volume, pressure):
    from rich.table import Table

    table = Table(title="3D Pressure-volume data")

    table.add_column("Time", justify="right", style="cyan", no_wrap=True)
    table.add_column("Current volume", style="magenta")
    table.add_column("Target volume", justify="right", style="green")
    table.add_column("Pressure", justify="right", style="red")

    table.add_row(
        f"{time:.2f}",
        f"{current_volume:.2f}",
        f"{target_volume:.2f}",
        f"{pressure:.2f}",
    )
    logger.info(f"\n{log.log_table(table)}")


def get_cavity_volume_form(mesh, u=None, xshift=5.0):
    shift = dolfinx.fem.Constant(mesh, (xshift, 0.0, 0.0))
    X = ufl.SpatialCoordinate(mesh) - shift
    N = ufl.FacetNormal(mesh)

    if u is None:
        vol_form = (-1.0 / 3.0) * ufl.dot(X, N)
    else:
        F = ufl.Identity(3) + ufl.grad(u)

        u1 = ufl.as_vector([0.0, u[1], 0.0])
        X1 = ufl.as_vector([0.0, X[1], 0.0])
        return (-1.0 / 1.0) * ufl.dot(X1 + u1, ufl.cofac(F) * N)

    return vol_form



def model(comm):
    geodir = Path("lv_ellipsoid")

    if not geodir.exists():
        # Make sure we don't create the directory before all
        # processes have reached this point
        comm.barrier()
        cardiac_geometries.mesh.lv_ellipsoid(
            outdir=geodir,
            r_short_endo=7.0,
            r_short_epi=10.0,
            r_long_endo=17.0,
            r_long_epi=20.0,
            mu_apex_endo=-math.pi,
            mu_base_endo=-math.acos(5 / 17),
            mu_apex_epi=-math.pi,
            mu_base_epi=-math.acos(5 / 20),
            fiber_space="Quadrature_6",
            create_fibers=True,
            fiber_angle_epi=-60,
            fiber_angle_endo=60,
        )
    geo = cardiac_geometries.geometry.Geometry.from_folder(
        comm=comm,
        folder=geodir,
    )
    # logger.addFilter(lambda record: 1 if geo.mesh.comm.rank == 0 else 0)
    geo.mesh.geometry.x[:] *= 0.35
    # Now, lets convert the geometry to a `fenicsx_pulse.Geometry` object.

    geometry = fenicsx_pulse.Geometry.from_cardiac_geometries(
        geo, metadata={"quadrature_degree": 2}
    )

    # The material model used in this benchmark is the {py:class}`Guccione <fenicsx_pulse.material_models.guccione.Guccione>` model.

    material_params = fenicsx_pulse.HolzapfelOgden.orthotropic_parameters()
    material = fenicsx_pulse.HolzapfelOgden(f0=geo.f0, s0=geo.s0, **material_params)  # type: ignore
    # We use an active stress approach with 60% transverse active stress

    Ta = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    active_model = fenicsx_pulse.ActiveStress(geo.f0, activation=Ta)

    # and the model should be incompressible

    comp_model = fenicsx_pulse.Compressible()

    # and assembles the `CardiacModel`

    model = fenicsx_pulse.CardiacModel(
        material=material,
        active=active_model,
        compressibility=comp_model,
    )

    def dirichlet_bc(
        state_space: dolfinx.fem.FunctionSpace,
    ) -> list[dolfinx.fem.bcs.DirichletBC]:

        Ux = state_space.sub(0)

        V, _ = Ux.collapse()

        facets = geometry.facet_tags.find(
            geometry.markers["BASE"][0],
        )  # Specify the marker used on the boundary
        geometry.mesh.topology.create_connectivity(
            geometry.mesh.topology.dim - 1,
            geometry.mesh.topology.dim,
        )
        dofs = dolfinx.fem.locate_dofs_topological((Ux, V), 2, facets)
        u_fixed = dolfinx.fem.Function(V)
        u_fixed.x.array[:] = 0.0
        return [dolfinx.fem.dirichletbc(u_fixed, dofs, Ux)]

    traction = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(0.0))
    neumann = fenicsx_pulse.NeumannBC(traction=traction, marker=geo.markers["ENDO"][0])

    pericardium = dolfinx.fem.Constant(geometry.mesh, dolfinx.default_scalar_type(5.0))
    robin_epi = fenicsx_pulse.RobinBC(
        value=pericardium, marker=geometry.markers["EPI"][0]
    )
    robin_base = fenicsx_pulse.RobinBC(
        value=pericardium, marker=geometry.markers["BASE"][0]
    )

    # and finally combine all the boundary conditions

    bcs = fenicsx_pulse.BoundaryConditions(
        neumann=(neumann,), robin=(robin_epi, robin_base), dirichlet=(dirichlet_bc,)
    )

    # and create a Mixed problem

    problem = fenicsx_pulse.MechanicsProblem(
        model=model, geometry=geometry, bcs=bcs, parameters={"u_order": 1}
    )

    # Now we can solve the problem
    # log.set_log_level(log.LogLevel.INFO)

    problem.solve()
    V = problem.state_space
    u = dolfinx.fem.Function(V)

    vtx = dolfinx.io.VTXWriter(
        problem.geometry.mesh.comm,
        "displacement.bp",
        [u],
        engine="BP4",
    )

    normal_activation_params = bestel.default_parameters()
    normal_activation_params["t_sys"] = 0.03

    normal_activation = (
        bestel.activation_function(
            t_span=[0, THB],
            t_eval=np.arange(0, THB, dt),
            parameters=normal_activation_params,
        )
        / 1000.0
    )

    @lru_cache
    def get_activation(t: float):
        # Find index modulo 1000
        i = t * 1000 % 1000
        return normal_activation[int(i)]

    def callback(t: float, save=False):
        if save:
            u.x.array[:] = problem.state.x.array
            vtx.write(t)
        value = get_activation(t)

        logger.debug(f"Time{t} with activation: {value}")
        if Ta.value != value:
            Ta.value = value
            problem.solve()

    volume_form = get_cavity_volume_form(geometry.mesh, u=u)
    volume = dolfinx.fem.form(volume_form * geometry.ds(geometry.markers["ENDO"][0]))
    initial_volume = geo.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(volume), op=MPI.SUM
        )
    logger.info(f"Initial volume: {initial_volume}")

    @lru_cache
    def p_LV_func(V_LV, t):
        u.x.array[:] = problem.state.x.array
        current_volume = geo.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(volume), op=MPI.SUM
        )
        if traction.value <= 0.0 and current_volume > V_LV:
            return 0.0

        current_pressure = old_pressure = traction.value.copy()
        old_state = problem.state.x.array.copy()
        j = 0
        f = 1.0

        # Newton iteration to find the correct pressure
        while current_pressure >= 0.0 and j < 100 and abs(f) > 0.1:
            j += 1
            traction.value = current_pressure
            problem.solve()
            old_pressure = current_pressure.copy()

            old_state[:] = problem.state.x.array.copy()
            u.x.array[:] = problem.state.x.array
            current_volume = geo.mesh.comm.allreduce(
                dolfinx.fem.assemble_scalar(volume), op=MPI.SUM
            )
            f = current_volume - V_LV

            dp = 1.0e-3
            p1 = current_pressure + dp
            traction.value = p1
            problem.solve()
            u.x.array[:] = problem.state.x.array
            v_eps = geo.mesh.comm.allreduce(
                dolfinx.fem.assemble_scalar(volume), op=MPI.SUM
            )
            df = (v_eps - current_volume) / dp
            current_pressure -= f / df

        problem.state.x.array[:] = old_state
        traction.value = old_pressure

        # print_table(t, current_volume, V_LV, current_pressure)
        return kPa_to_mmHg(traction.value)

    return callback, p_LV_func, initial_volume


def main(comm):
    callback, p_LV_func, initial_volume = model(comm)

    mL = ureg.mL

    add_units = False

    circulation = Regazzoni2020(
        add_units=add_units, callback=callback, p_LV_func=p_LV_func, verbose=True, comm=comm,
    )

    if add_units:
        init_state = {"V_LV": initial_volume * mL}
    else:
        init_state = {"V_LV": initial_volume}

    circulation.update_state(state=init_state)
    circulation.print_info()
    history = circulation.solve(num_cycles=10, dt=dt)
    circulation.print_info()
    # -

    # Let us now visually compare the results obtanined with the $\mathcal{M}_{\text{0D}}$-$\mathcal{C}$ model and with the $\mathcal{M}_{\text{3D}}$-$\mathcal{C}$ model:

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(history["V_LV"], history["p_LV"])
    ax[0].set_xlabel("V [mL]")
    ax[0].set_ylabel("p [mmHg]")

    ax[1].plot(history["time"], history["p_LV"])
    ax[2].plot(history["time"], history["V_LV"])

    fig.savefig("pv_loop")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    log.setup_logging(comm=comm)
    main(comm)
