import circulation


def test_Zenkur():
    model = circulation.zenkur.Zenkur()
    results = model.solve(T=1.0, dt=1e-3, dt_eval=0.1)

    for k, v in results.items():
        assert len(v) == len(results["time"]), k

    for k, v in model.default_initial_conditions().items():
        assert k in results
        print(v)
        assert results[k][0] == v.magnitude


def test_Regazzoni2020():
    model = circulation.regazzoni2020.Regazzoni2020()
    results = model.solve(T=1.0, dt=1e-3, dt_eval=0.1)

    for k, v in results.items():
        assert len(v) == len(results["time"]), k

    for k, v in model.default_initial_conditions().items():
        assert k in results
        assert results[k][0] == v.magnitude
