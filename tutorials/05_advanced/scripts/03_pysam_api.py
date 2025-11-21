# %%
import pvdeg
from pvdeg import TEST_DATA_DIR
import pandas as pd
import os
import pickle
import xarray as xr

# %% [markdown]
# ## Pysam
#
# rundown on pysam...
#
# https://nrel-pysam.readthedocs.io/en/main/inputs-from-sam.html

# %% [markdown]
# ### PVGIS
#
# Only works with PVGIS

# %%
weather_pvgis, meta_pvgis = pvdeg.weather.get(
    database="PVGIS", id=(25.783388, -81.189029)
)

# %%
results = pvdeg.pysam.pysam(
    weather_df=weather_pvgis,
    meta=meta_pvgis,
    pv_model="pysamv1",
    pv_model_default="FlatPlatePVCommercial",
)

# %%
results["annual_energy"]

# %% [markdown]
# ### Local Geospatial
#
# Using PySAM with geospatial data requires proper formatting of the weather DataFrame to match PySAM's expectations.

# %%
GEO_META = pd.read_csv(os.path.join(TEST_DATA_DIR, "summit-meta.csv"), index_col=0)
GEO_WEATHER = xr.open_dataset(os.path.join(TEST_DATA_DIR, "summit-weather.nc"))


# %% [markdown]
# ### Local Geospatial - PySAM Integration
#
# The geospatial weather data is in half-hourly format (17520 timesteps) but PySAM expects hourly data (8760 timesteps). The wrapper function below handles this conversion automatically.


# %%
# this is just a wrapper to grab the result we want
def pysam_annual_energy(
    weather_df, meta, pv_model="pysamv1", pv_model_default="FlatPlatePVCommercial"
):
    # Drop the gid column if present (added by geospatial conversion)
    weather_df = weather_df.drop(columns=["gid"])

    # Resample half-hourly data to hourly (PySAM expects hourly)
    weather_df = weather_df.resample("h").mean()

    results = pvdeg.pysam.pysam(
        weather_df=weather_df,
        meta=meta,
        pv_model=pv_model,
        pv_model_default=pv_model_default,
    )

    return results["annual_energy"]


# %%
# Select a small subset (2 gids) for demonstration to avoid timeouts
# PySAM calculations are computationally expensive
subset_gids = GEO_META.index[:2]
GEO_META_SUB = GEO_META.loc[subset_gids]
GEO_WEATHER_SUB = GEO_WEATHER.sel(gid=subset_gids)

template = pvdeg.geospatial.output_template(
    ds_gids=GEO_WEATHER_SUB,
    shapes={
        "Annual Energy": ("gid",),
    },
)

geo_res = pvdeg.geospatial.analysis(
    weather_ds=GEO_WEATHER_SUB,
    meta_df=GEO_META_SUB,
    func=pysam_annual_energy,
    template=template,
)

# %%
geo_res

# %% [markdown]
# ## NSRDB API

# %%
weather_db = "PSM4"
weather_id = (25.783388, -80.189029)
weather_arg = {"api_key": "DEMO_KEY", "email": "user@mail.com", "map_variables": True}

weather_df, meta = pvdeg.weather.get(weather_db, weather_id, **weather_arg)

# %% [markdown]
# ### Geospatial Scenario

# %%
location_grabber = pvdeg.GeospatialScenario()

location_grabber.addLocation(country="United States", downsample_factor=80)

# %%
location_grabber.plot_coords()

# %%
geo_weather, geo_meta = location_grabber.geospatial_data

# Select a small subset (2 gids) for demonstration to avoid timeouts
# PySAM calculations are computationally expensive
subset_gids = geo_meta.index[:2]
geo_meta_sub = geo_meta.loc[subset_gids]
geo_weather_sub = geo_weather.sel(gid=subset_gids)

template = pvdeg.geospatial.output_template(
    ds_gids=geo_weather_sub,
    shapes={
        "pysam_annual_energy": ("gid",),
    },
)

geo_res = pvdeg.geospatial.analysis(
    weather_ds=geo_weather_sub,
    meta_df=geo_meta_sub,
    func=pysam_annual_energy,  # using wrapper from before
    template=template,
)

# %%
pvdeg.geospatial.plot_sparse_analysis(geo_res, data_var="pysam_annual_energy")

# %%
# Check weather data time dimension compatibility
if "time" in GEO_WEATHER.dims:
    times = pd.to_datetime(GEO_WEATHER["time"].values)
    years = times.year
    unique_years = set(years)
    if len(unique_years) != 1:
        print(
            f"Warning: Weather data contains multiple years: {unique_years}. Pysam expects a single year."
        )
    hours_per_year = times.size / len(unique_years)
    if hours_per_year not in [8760, 8784]:
        print(
            f"Warning: Unexpected number of timesteps per year: {hours_per_year}. Expected 8760 or 8784."
        )
else:
    print("Warning: No 'time' dimension found in weather data. Pysam may fail.")
