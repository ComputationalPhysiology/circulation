import numpy as np
from circulation import regazzoni2020


def test_regazzoni_steady_state():
    """
    Test that the default initial state is close to the limit cycle.
    We run the model for ten beats and check that the state hasn't drifted much.
    """
    model = regazzoni2020.Regazzoni2020()
    initial_state = model.state.copy()
    model.solve(num_beats=10)
    final_state = model.state

    np.testing.assert_allclose(final_state, initial_state, rtol=1e-2, verbose=True)
