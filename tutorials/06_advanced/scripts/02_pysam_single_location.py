# %% [markdown]
# # PySAM for a Single Location
#
# Run a PySAM `pvsamv1` simulation for a single location, inspect the model output
# dictionary, and visualize the spatial rear-side **ground irradiance** profile
# beneath a bifacial array.
#
# This notebook uses a cached NSRDB weather file for New York City that ships with
# the repository, so it runs fully offline with no API key or network access
# required.

# %%
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvdeg

# %%
# Load cached NSRDB weather for New York City that ships with the repo.
# This keeps the notebook reproducible and fully offline (no API key or network).
repo_root = os.path.dirname(os.path.dirname(pvdeg.__file__))
weather_path = os.path.join(repo_root, "tutorials", "data", "psm4_nyc.csv")
meta_path = os.path.join(repo_root, "tutorials", "data", "meta_nyc.json")

weather = pd.read_csv(weather_path, index_col=0, parse_dates=True)
with open(meta_path, "r") as f:
    meta = json.load(f)

meta

# %%
out_dict = pvdeg.pysam.pysam(
    weather_df=weather,
    meta=meta,
    pv_model="pvsamv1",
    pv_model_default="FlatPlatePVCommercial",
)
for key in sorted(out_dict.keys()):
    print(key)

# %%
# Most outputs are scalars or 1-D time series. A few are 2-D matrices, returned
# as a tuple whose entries are themselves tuples (rows). Find those.
for key, item in out_dict.items():
    if isinstance(item, tuple) and item and isinstance(item[0], tuple):
        print(key)

# %% [markdown]
# ## Spatial ground irradiance
#
# A few outputs describe irradiance that varies *across* the ground between module
# rows and are returned as 2-D matrices (a tuple of row tuples):
#
# - `subarray1_ground_rear_spatial` &mdash; rear-side irradiance reaching the ground.
# - `subarray1_poa_rear_spatial` &mdash; rear-side plane-of-array irradiance.
#
# We visualize `subarray1_ground_rear_spatial`, which matters for agrivoltaics: it
# tells us how much light reaches the ground beneath a bifacial array. Its layout is:
#
# - **Row `0`** &mdash; the ground positions (in metres) where irradiance is
#   evaluated. The leading value is a `0` placeholder.
# - **Rows `1:`** &mdash; one row per hourly timestep. The leading value is the
#   timestep index; the remaining values are the irradiance at each position.

# %%
spatial = out_dict["subarray1_ground_rear_spatial"]

# Row 0 holds the ground positions (drop the leading placeholder).
# In every data row, column 0 is the timestep index, so drop it too.
distances = np.array(spatial[0])[1:]
ground_irradiance = np.array(spatial[1:])[:, 1:]  # shape: (hours, positions)

# Pick the sunniest day (greatest total rear ground irradiance) to visualize.
daily_total = ground_irradiance.reshape(-1, 24, distances.size).sum(axis=(1, 2))
best_day = int(daily_total.argmax())

day_slice = slice(best_day * 24, best_day * 24 + 24)
day_index = weather.index[day_slice]
day_irradiance = ground_irradiance[day_slice]

# %%
# Ground irradiance varies across the pitch because the module rows cast a moving
# shadow. Plot the spatial profile at a few hours to see the shading band shift.
fig, ax = plt.subplots(figsize=(8, 5))

for hour in (8, 12, 16):
    ax.plot(
        distances,
        day_irradiance[hour],
        marker="o",
        label=day_index[hour].strftime("%H:%M"),
    )

ax.set_xlabel("Ground position between module rows [m]")
ax.set_ylabel("Rear ground irradiance [W/m$^2$]")
ax.set_title(
    "Rear-side ground irradiance beneath a bifacial array\n"
    f"New York City \u2014 {day_index[0].strftime('%B')} {day_index[0].day}"
)
ax.legend(title="Hour of day")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
