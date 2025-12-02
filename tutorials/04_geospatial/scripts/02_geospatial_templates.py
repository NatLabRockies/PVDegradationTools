# %%
import pvdeg
from pvdeg import TEST_DATA_DIR
import pandas as pd
import os
import xarray as xr


# %% [markdown]
# # Geospatial Templates
#
# When running a geospatial analysis using `pvdeg.geospatial.analysis` on arbitary `pvdeg` functions you will need to specify a template for the shape of the output data. This is because the input data comes with dimensions of gid and time while the output will have data in a different shape usually corresonding to coordinates.
# - gid, identification number corresponding to an NSRDB datapoint's location
# - time, timeseries corresponding to the hourly time indicies of NSRDB datapoint's yearly meteorological data.
#
# Follow the steps below to see how we generate templates before running the analysis.
#
# The only functions where this is not required are currently `pvdeg.standards.standoff`, `pvdeg.humidity.moduke` and, `letid.calc_letid_outdoors` as they are predefined within the package.

# %% [markdown]
# # Loading Geospatial Data
#
# This step skips over making the `pvdeg.weather.get` call with `geospatial == True`. See the [Duramat Demo](./DuraMAT%20Live%20Demo.ipynb) for information on how to do this traditionally.
#
# We can also use a `GeospatialScenario` object. See the [Geospatial Scenario Tutorial](./Scenario%20-%20Geospatial.ipynb) for more information on how to use this approach.
#
# *The cell below loads a pickled xarray object, this is not the best way to do this. xarray datasets should be stored as `.nc` - netcdf files*

# %%
geo_meta = pd.read_csv(os.path.join(TEST_DATA_DIR, "summit-meta.csv"), index_col=0)

# Use xarray to open NetCDF file instead of pickle
geo_weather = xr.open_dataset(os.path.join(TEST_DATA_DIR, "summit-weather.nc"))

geo_weather


# %% [markdown]
# # Creating Templates Manually
#
# `pvdeg.geospatial.ouput_template` we can produce a template for our result data.
#
# We need to do this because different functions return different types of values, some return multiple values as tuples, some return only single numerics, others return timeseries results. We need to specify the shape of our data to create an output xarray dataset.

# %% [markdown]
# # Examples
#
# ## 98ᵗʰ module percential temperature at Standoff Height
#
# Say we want to estimate the 98ᵗʰ percential temperature for the module at the given tilt, azimuth, and x_eff. `PVDeg` has a function to do this, `pvdeg.standards.T98_estimate` BUT it doesn't have a preset geospatial template. We will need to make one.
#
# - look at the function return values.
# From the docstring we can see that `T98_estimate` only has one return value. IMPORTANT, this value is a single float, NOT a timeseries. This means our output shape will only be dependent on the input identifier and NOT time.
#
# Therefore we will map the output variable `T98` to the location identifier `gid` using a dictionary with `str: tuple` mappings.
#
#  *IMPORTANT: you must use the syntax below where the variable maps to a tuple of the coordinates. in this case there needs to be a trailing comma in the tuple or python will iterate over the characters in the tuple instead of the elements. See further examples to alleviate confusion.*

# %%
# define output shape
shapes = {
    "T98": (
        "gid",
    )  # one return value at each datapoint, only dependent on datapoint, not time
}

# create xarray template for output to be populated when analysis is run
template = pvdeg.geospatial.output_template(
    ds_gids=geo_weather,
    shapes=shapes,
)

# %%
geo_estimate_temp = pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.standards.T98_estimate,
    template=template,
)

# %% [markdown]
# # Glass Glass Estimated Module Temperature
#
# Now we want to calculate geospatial timeseries temperature values for a module using `pvdeg.temperature.module`. This is not super practical because all `pvdeg` functions that need to use tempeature for their calculations preform the temperature calculation internally, this is just for show.
#
# This calculation differs from the above because the temperature functions return the module temperature in a timeseries format. So we care about 2 dimensions, location identifier and TIME.

# %%
# define output shape
shapes = {
    "module_temperature": (
        "gid",
        "time",
    )  # one return value at each datapoint, only dependent on datapoint, not time
}

# create xarray template for output to be populated when analysis is run
temperature_template = pvdeg.geospatial.output_template(
    ds_gids=geo_weather,
    shapes=shapes,
)

# %%
geo_temperature_res = pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.temperature.module,
    template=temperature_template,  # use the template we created
    conf="open_rack_glass_glass",  # provide kwargs for function here
)

# %%
# plot the temperature at ONE of the geospatial analysis result locations but we have calculated all of these.
import matplotlib.pyplot as plt

module_temps = (
    geo_temperature_res["module"].sel(latitude=39.89, longitude="-106.42").values
)

plt.plot(module_temps)

# %% [markdown]
# # Self Explaining Code
#
# If we are looking at adding templates for other functions, we can also look at the 3 presaved templates for existing `pvdeg` functions. Visit [pvdeg.geospatial.template_parameters](../../pvdeg/geospatial.py) and inspect this function to see how these different target functions utilize templates and shapes.

# %% [markdown]
# # Creating Templates Programatically
#
# We can use `pvdeg.geospatial.autotemplate` to generate a template for a given function. This can return a bad result which will fail or work improperly when running `pvdeg.geospatial.analysis` with the generated template. Results should be scrutinized to make sure they are the right format.

# %% [markdown]
# # Examples Below
# Steps
# - Create template using autotemplating function. Pulls in information about function to determine shape of output. Not usable on functions with ambigious return types.
# - Call geospatial analysis function using template

# %% [markdown]
# # Geospatial Cell Temperature Calculation
# As shown below, we have two options, we can choose to provide a template that is generated by a function which supports autotemplating. Or we can provide the function to `geospatial.analysis` and let it generate a template internally.
#
# ## Providing a Template with `Geospatial.auto_template`

# %%
# create a template using auto_template for the desired function
cell_temp_template = pvdeg.geospatial.auto_template(
    func=pvdeg.temperature.cell, ds_gids=geo_weather
)

# run the geospatial analysis with the template
pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.temperature.cell,
    template=cell_temp_template,
)

# %% [markdown]
# # Analysis Without Providing a Template
#
# If a function is supported by `geospatial.auto_template` we do not need to create a template outside of the function as shown in the cell above. We can simply pass the function to `geospatial.analysis` and it will create a template for us.

# %%
pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.temperature.cell,
)

# %% [markdown]
# # Geospatial Module Temperature Calculation

# %%
module_temp_template = pvdeg.geospatial.auto_template(
    func=pvdeg.temperature.module, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.temperature.module,
    template=module_temp_template,
)

# %% [markdown]
# # Geospatial Solar Position Calculation

# %%
solar_position_template = pvdeg.geospatial.auto_template(
    func=pvdeg.spectral.solar_position, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.spectral.solar_position,
    template=solar_position_template,
)

# %% [markdown]
# # Geospatial POA Irradiance Calculation

# %%
poa_irradiance_template = pvdeg.geospatial.auto_template(
    func=pvdeg.spectral.poa_irradiance, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.spectral.poa_irradiance,
    template=poa_irradiance_template,
)

# %% [markdown]
# # Geospatial 98th Percentile Operating Temperature Calculation

# %%
standoff_template = pvdeg.geospatial.auto_template(
    func=pvdeg.standards.T98_estimate, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.standards.T98_estimate,
    template=standoff_template,
)

# %% [markdown]
# # Geospatial Module Humidity Calculation

# %%
humidity_template = pvdeg.geospatial.auto_template(
    func=pvdeg.humidity.module, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.humidity.module,
    template=humidity_template,
    backsheet_thickness=0.3,
    back_encap_thickness=0.5,
)

# %% [markdown]
# # Geospatial IwaVantHoff Environment Characterization Calculation

# %%
iwa_vant_hoff_template = pvdeg.geospatial.auto_template(
    func=pvdeg.degradation.IwaVantHoff, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.degradation.IwaVantHoff,
    template=iwa_vant_hoff_template,
)

# %% [markdown]
# # Geospatial Edge Seal Width Calculation

# %%
edge_seal_template = pvdeg.geospatial.auto_template(
    func=pvdeg.design.edge_seal_width, ds_gids=geo_weather
)

pvdeg.geospatial.analysis(
    weather_ds=geo_weather,
    meta_df=geo_meta,
    func=pvdeg.design.edge_seal_width,
    template=edge_seal_template,
)

# %%
edge_seal_template
