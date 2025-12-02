import numpy as np
import circulation


def test_initial_state_constructor():
    # Test that initial_state passed to constructor is respected in solve
    custom_V_LA = 100.0
    model = circulation.regazzoni2020.Regazzoni2020(initial_state={"V_LA": custom_V_LA})

    # Solve for a very short time
    model.solve(num_beats=1, dt=1e-3)

    # Check results
    idx = model.states_names.index("V_LA")
    v_la_start = model.results_state[idx, 0]

    assert np.isclose(v_la_start, custom_V_LA), f"Expected {custom_V_LA}, got {v_la_start}"


def test_initial_state_solve():
    # Test that initial_state passed to solve overrides constructor/defaults
    model = circulation.regazzoni2020.Regazzoni2020()
    custom_V_LA = 200.0

    model.solve(num_beats=1, dt=1e-3, initial_state={"V_LA": custom_V_LA})

    idx = model.states_names.index("V_LA")
    v_la_start = model.results_state[idx, 0]

    assert np.isclose(v_la_start, custom_V_LA), f"Expected {custom_V_LA}, got {v_la_start}"


def test_parameters_constructor():
    # Test that parameters passed to constructor are respected
    custom_HR = 2.0

    model = circulation.regazzoni2020.Regazzoni2020(parameters={"HR": custom_HR})

    # Check if the parameter is set correctly
    # HR property accesses self.parameters["HR"]
    assert model.HR == custom_HR, f"Expected HR {custom_HR}, got {model.HR}"

    # Also check a nested parameter
    custom_EA = 10.0
    model = circulation.regazzoni2020.Regazzoni2020(
        parameters={"chambers": {"LA": {"EA": custom_EA}}}
    )
    assert model.parameters["chambers"]["LA"]["EA"] == custom_EA
