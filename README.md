# Cardiac circulation models

This package provides a set of models for the circulation of blood in the heart and the rest of the body.

## Install

```
python -m pip install circulation
```


## Documentation

See https://computationalphysiology.github.io/circulation

# Usage

```python
import matplotlib.pyplot as plt
import circulation

# Create a model
model = circulation.regazzoni2020.Regazzoni2020()

# Solve the model for 5 seconds with a  time step of 1 ms and an evaluation time step of 10 ms
results = model.solve(T=5.0, dt=1e-3, dt_eval=0.01)

# Print some information about the pressure, flows and volumes inside the model
model.print_info()

# Plot the pressure-volume loop
fig, ax = plt.subplots()
ax.plot(results["V_LV"], results["p_LV"])
ax.set_xlabel("V [mL]")
ax.set_ylabel("p [mmHg]")
plt.show()
```
![_](https://raw.githubusercontent.com/ComputationalPhysiology/circulation/main/docs/_static/regazzoni.png)

See more examples in the [documentation](https://computationalphysiology.github.io/circulation).

# License

MIT
