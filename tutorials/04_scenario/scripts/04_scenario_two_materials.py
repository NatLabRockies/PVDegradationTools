# %% [markdown]
# # Two-Material Scenario
#
# This notebook shows how to:
# 1. Load two materials from the pvdeg material databases
# 2. Bind a different degradation function to each material layer
# 3. Run the pipeline and inspect the results

# %%
import pvdeg
import os
import pandas as pd
import json

# %% [markdown]
# ## Load weather data
#
# Read a local PSM4 weather file that ships with the pvdeg tutorials.

# %%
weather_df = pd.read_csv("../data/psm4_golden.csv", index_col=0, parse_dates=True)
with open("../data/meta_golden.json", "r") as f:
    meta = json.load(f)

# %% [markdown]
# ## Create the scenario and load two materials
#
# Pass `materials` as a dict to assign a named layer to each material.
# Each entry points to a material key in one of the pvdeg JSON databases
# (`"O2permeation"`, `"H2Opermeation"`, or `"AApermeation"`).
#
# Here we load **OX003** (EVA encapsulant) and **OX004** (another encapsulant variant)
# from `O2permeation.json` as two separate layers.

# %%
scenario = pvdeg.Scenario(
    name="two-material-demo",
    weather_data=weather_df,
    meta_data=meta,
)

scenario.addModule(
    module_name="glass-polymer",
    materials={
        "encapsulant": {
            "material_file": "O2permeation",
            "material_name": "OX003",
        },
        "backsheet": {
            "material_file": "O2permeation",
            "material_name": "OX004",
        },
    },
)

# %% [markdown]
# ## Add one degradation job per material layer
#
# Use a 2-tuple `(function, layer_name)` so that `run()` knows which material
# parameters to inject for each job.
#
# - **IwaVantHoff** characterises the outdoor degradation environment for the encapsulant
# - **standoff** computes the minimum mounting standoff for the backsheet side
#
# Material parameters from the database that do not appear in a function's
# signature (e.g. `Ead`, `Do`) are silently filtered out, so you don't need
# to worry about parameter mismatches.

# %%
scenario.addJob(func=(pvdeg.degradation.IwaVantHoff, "encapsulant"))
scenario.addJob(func=(pvdeg.degradation.arrhenius, "backsheet"))

# %% [markdown]
# ## Run the pipeline

# %%
scenario.run()

# %% [markdown]
# ## Inspect results
#
# `results` is a nested dict: `results[module_name][job_id]`.
# Use `display(scenario)` to see job IDs, then index into them directly.

# %%
display(scenario)

# %%
module_results = scenario.results["glass-polymer"]

for job_id, result in module_results.items():
    print(f"job {job_id}: {result}")
