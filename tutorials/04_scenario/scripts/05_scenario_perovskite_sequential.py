# %% [markdown]
# # Sequential Scenario Analysis — Perovskite Degradation
#
# Demonstrates:
# 1. **Multiple jobs in one pipeline** — module temperature, surface humidity, IWa, and perovskite degradation all in a single `run()` call.
# 2. **Sequential job chaining (`depends_on`)** — downstream jobs receive the output of upstream jobs as named keyword arguments at run time.
# 3. **Multi-material expansion** — how to bind different degradation functions to named material layers.

# %% [markdown]
# ## 1. Imports and data

# %%
import os
import json
import pandas as pd
import pvdeg
import matplotlib.pyplot as plt

# Anchor the tutorials data directory before Scenario changes cwd
REPO_ROOT = os.path.dirname(os.path.dirname(pvdeg.__file__))
TUTORIALS_DATA = os.path.join(REPO_ROOT, "tutorials", "data")

# %%
weather_df = pd.read_csv(
    os.path.join(TUTORIALS_DATA, "psm4_golden.csv"),
    index_col=0,
    parse_dates=True,
)
with open(os.path.join(TUTORIALS_DATA, "meta_golden.json")) as f:
    meta = json.load(f)

display_year = 2020
weather_df.index = weather_df.index.map(lambda ts: ts.replace(year=display_year))
weather_df = weather_df.sort_index()

print(f"Using normalized display year: {display_year} ({len(weather_df)} rows)")
print(f"Range: {weather_df.index.min()} -> {weather_df.index.max()}")

# %% [markdown]
# ---
# ## 2. Single job — perovskite degradation rate
#
# The simplest use: one job, no sequential dependencies.
#
# `perovskite_degradation` computes the four-pathway kinetic rate:
#
# $$r = r_\text{WPO} + r_\text{DPO} + r_\text{hum} + r_\text{therm}$$
#
# All it needs from the pipeline is `temp_air` and `relative_humidity` from `weather_df`.
# The `component` kwarg selects which term (or their sum) to return.

# %% [markdown]
# ### 2a. Load parameters from the database
#
# `pvdeg.utilities.get_kinetics("D015")` returns D015 as a flat dict — the same values the function uses as defaults, but now sourced from the single authoritative location in `DegradationDatabase.json`.
# Pass it to the `parameters` argument to use the DB as the source of truth.

# %%
d015 = pvdeg.utilities.get_kinetics("D015")
# d015  # Uncomment to see material kinetics details

# %%
s1 = pvdeg.Scenario(
    name="perovskite-simple",
    weather_data=weather_df,
    meta_data=meta,
)

# Default: uses hardcoded parameter values from the paper
s1.addJob(func=pvdeg.degradation.perovskite_degradation, name="default_params")

# DB-sourced: same values, but read from DegradationDatabase.json D015
s1.addJob(
    func=(pvdeg.degradation.perovskite_degradation, {"parameters": d015}),
    name="db_params",
)

s1.run()
# display(s1)  # Uncomment to see results

# %%
rate_default = s1.results["default_params"]
rate_db = s1.results["db_params"]

# Both should be numerically identical — the DB is the same source of truth
print("Max absolute difference:", (rate_default - rate_db).abs().max())

rate_default.plot(
    title="Perovskite degradation rate — total [mol m⁻² s⁻¹]", figsize=(12, 3)
)

# %% [markdown]
# ---
# ## 3. Sequential analysis — degradation rate as a downstream job
#
# Each job feeds the next via `depends_on`.
# The pipeline injects named upstream outputs as keyword arguments at run time — no manual wiring needed.
#
# **Chain:**
# | Step | Function | Output name | Depends on |
# |------|----------|-------------|------------|
# | 1 | `temperature.module` | `"temp_mod"` | — |
# | 2 | `humidity.surface_relative` | `"rh_surface"` | `temp_module` ← `"temp_mod"` |
# | 3 | `humidity.water_vapor_pressure` | `"P_H2O"` | `temp_air` ← `"temp_mod"`, `relative_humidity` ← `"rh_surface"` |
# | 4 | `degradation.perovskite_degradation` | `"perov_rate"` | `P_H2O` ← `"P_H2O"` |
#
# Step 2 converts ambient RH to **module-surface RH** using the module temperature.
# Step 3 converts that surface RH (and module temperature) into the **water vapour partial pressure** the kinetic model requires.

# %%
s2 = pvdeg.Scenario(
    name="perovskite-sequential",
    weather_data=weather_df,
    meta_data=meta,
)

# Job 1 — module surface temperature
s2.addJob(
    func=pvdeg.temperature.module,
    name="temp_mod",
)

# Job 2 — module-surface RH; uses module temperature from Job 1
s2.addJob(
    func=(
        pvdeg.humidity.surface_relative,
        {
            "rh_ambient": weather_df["relative_humidity"],
            "temp_ambient": weather_df["temp_air"],
        },
    ),
    name="rh_surface",
    depends_on={"temp_module": "temp_mod"},
)

# Job 3 — water vapour pressure at module surface; uses temp and RH from Jobs 1 & 2
s2.addJob(
    func=pvdeg.humidity.water_vapor_pressure,
    name="P_H2O",
    depends_on={"temp_air": "temp_mod", "relative_humidity": "rh_surface"},
)

# Job 4 — perovskite degradation rate; uses P_H2O from Job 3
s2.addJob(
    func=pvdeg.degradation.perovskite_degradation,
    name="perov_rate",
    depends_on={"P_H2O": "P_H2O"},
)

s2.run()
# display(s2)  # Uncomment to see results

# %%
temp_mod = s2.results["temp_mod"]
rh_surf = s2.results["rh_surface"]
p_h2o = s2.results["P_H2O"]
perov = s2.results["perov_rate"]

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
temp_mod.plot(ax=axes[0], title="Module temperature [°C]")
rh_surf.plot(ax=axes[1], title="Module-surface relative humidity [%]")
p_h2o.plot(ax=axes[2], title="Water vapour pressure P_H2O [kPa]")
perov.plot(ax=axes[3], title="Perovskite degradation rate — total [mol m⁻² s⁻¹]")
fig.tight_layout()

# %% [markdown]
# ### 3b. Component breakdown
#
# Run all four degradation pathways as separate named jobs to compare their contributions.
# Each job is independent but they all share the same `weather_df` from the scenario.

# %%
s3 = pvdeg.Scenario(
    name="perovskite-components",
    weather_data=weather_df,
    meta_data=meta,
)

for comp in ("WPO", "DPO", "r_hum", "r_therm"):
    s3.addJob(
        func=(pvdeg.degradation.perovskite_degradation, {"component": comp}),
        name=comp,
    )

s3.run()

# Collect all four component series into a single DataFrame for comparison
components_df = pd.DataFrame(
    {comp: s3.results[comp] for comp in ("WPO", "DPO", "r_hum", "r_therm")}
)
components_df.plot(
    title="Perovskite degradation pathways [mol m⁻² s⁻¹]",
    figsize=(12, 4),
    logy=True,
)
