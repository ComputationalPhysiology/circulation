"""The `.ode` description of Regazzoni2020 must agree with the Python one.

`regazzoni2020.ode` exists so the model can be translated to other targets --
in particular to UFL, where it can be embedded directly into a finite element
variational problem. That is only worth anything if the two descriptions stay
identical, and nothing enforces that automatically: they are separate files
edited by hand.

So this compares them directly. Code is generated from the `.ode` file with
`gotranx` and its `rhs` is checked against `Regazzoni2020.rhs` over randomized
states, at times chosen to land inside and outside every activation window and
on both sides of every valve threshold -- the places where a mistranscribed
`Conditional` would hide while a single nominal state sailed through.

Exact equality is not expected: `sympy` reassociates arithmetic while
generating code, so the two evaluate the same expressions in a different order.
Agreement to ~1e-12 relative is what correctness looks like here.
"""

from __future__ import annotations

import numpy as np
import pytest

from circulation import regazzoni2020

gotranx = pytest.importorskip("gotranx", reason="gotranx is needed to read the .ode file")


@pytest.fixture(scope="module")
def generated():
    """Generate and import the Python model from the `.ode` file."""
    import gotranx.cli.gotran2py
    from gotranx.codegen.python import Format

    ode = gotranx.load_ode(regazzoni2020.ODE_FILE)
    code = gotranx.cli.gotran2py.get_code(ode, format=Format.none)
    namespace: dict = {}
    exec(code, namespace)
    return namespace


@pytest.fixture(scope="module")
def model():
    return regazzoni2020.Regazzoni2020(add_units=False)


def _to_generated_order(state, names, generated):
    """Reorder a state vector into the generated code's indexing.

    `gotranx` sorts states alphabetically, which is not the order
    `state_names()` uses, so this must go through `state_index` -- indexing by
    position would silently compare the wrong quantities.
    """
    out = np.zeros_like(state)
    for value, name in zip(state, names):
        out[generated["state_index"](name)] = value
    return out


def test_state_names_match(generated, model):
    """Both descriptions must contain the same states, whatever the ordering."""
    assert set(model.state_names()) == {
        name for name in model.state_names() if generated["state_index"](name) >= 0
    }


def test_default_parameters_round_trip(generated, model):
    """Every `.ode` parameter must be supplied by the flattening, and no others."""
    flat = regazzoni2020.flat_ode_parameters(model.parameters)
    # Raises KeyError from `parameter_index` if a name is not in the .ode file.
    for name in flat:
        assert generated["parameter_index"](name) >= 0

    n_ode_parameters = generated["init_parameter_values"]().size
    assert len(flat) == n_ode_parameters, (
        "flat_ode_parameters does not cover every parameter in the .ode file"
    )


@pytest.mark.parametrize(
    "t",
    [
        0.0,  # start of the beat
        0.05,  # atrial contraction window
        0.15,  # ventricular contraction window
        0.30,  # ventricular contraction/relaxation boundary
        0.45,  # ventricular relaxation window
        0.70,  # rest
        0.79,  # just before the beat wraps
        0.81,  # just after the beat wraps -- exercises the phase wrap
        1.55,  # a later beat entirely
        3.30,  # several beats on, where any phase drift would show
    ],
)
def test_rhs_matches_python_implementation(generated, model, t):
    """The generated rhs must agree with the hand-written one at every phase."""
    flat = regazzoni2020.flat_ode_parameters(model.parameters)
    parameters = generated["init_parameter_values"](**flat)
    names = model.state_names()

    rng = np.random.default_rng(abs(hash(("regazzoni", t))) % (2**32))
    reference_state = np.array(model.state, dtype=float)

    worst = 0.0
    for _ in range(50):
        # Perturb well away from the operating point: a sign error in a valve
        # term can vanish at a steady state and only appear off it.
        state = reference_state * rng.uniform(0.6, 1.4, size=reference_state.size)

        expected = np.asarray(model.rhs(t, state)).copy()
        actual = np.asarray(
            generated["rhs"](t, _to_generated_order(state, names, generated), parameters),
        )

        for value, name in zip(expected, names):
            got = actual[generated["state_index"](name)]
            worst = max(worst, abs(value - got) / max(abs(value), 1e-30))

    assert worst < 1e-10, f"largest relative difference {worst:.3e} at t={t}"


def test_splitting_the_lv_exposes_its_pressure(generated, model):
    """Subtracting the LV chamber must leave exactly the coupling seam.

    This is the property the 3D-0D coupling depends on: the same file serves
    both uses, and removing the chamber closure turns its pressure into a value
    supplied from outside while the chamber volume stays a state.
    """
    ode = gotranx.load_ode(regazzoni2020.ODE_FILE)
    assert ode.missing_variables == {}, "the standalone model must be self-contained"

    coupled = ode - ode.get_component("timing") - ode.get_component("LV")

    assert set(coupled.missing_variables) == {"beat_phase", "p_LV"}
    assert len(coupled.states) == len(model.state_names())
    assert "V_LV" in {state.name for state in coupled.states}
