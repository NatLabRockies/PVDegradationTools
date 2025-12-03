# %% [markdown]
# # PVGIS Distributed
#

# %%
import pvdeg
from global_land_mask import globe
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import LocalCluster, Client

# %% [markdown]
# # Creating Coordinates List
#
# Lets Generate a Grid of Latitude and Longitude Coordinates over the UK.

# %%
# decrease the arange step size for more fine resolution
# increase the arange step size for increased granularity
lon_UK = np.arange(-10.5, 1.76, 1)
lat_UK = np.arange(49.95, 60, 2)
lon_grid_UK, lat_grid_UK = np.meshgrid(lon_UK, lat_UK)
land_UK = globe.is_land(lat_grid_UK, lon_grid_UK)

lon_land_UK = lon_grid_UK[land_UK]
lat_land_UK = lat_grid_UK[land_UK]

lon_Scan = np.arange(-10.5, 31.6, 0.3)
lat_Scan = np.arange(60, 71.2, 0.3)
lon_grid_Scan, lat_grid_Scan = np.meshgrid(lon_Scan, lat_Scan)
land_Scan = globe.is_land(lat_grid_Scan, lon_grid_Scan)

lon_land_Scan = lon_grid_Scan[land_Scan]
lat_land_Scan = lat_grid_Scan[land_Scan]

coords = list(
    zip(lat_land_UK, lon_land_UK)
)  # easiest way to make a list of the right shape

# %% [markdown]
# # Dask
#
# We are using dask to parallelize the weather API calls. We need to start a dask client as shown below. This can also be done with `pvdeg.geospatial.start_dask`.
#
# Click on the link to open a localhost dashboard for the dask client. This will allow us to see what happens when we run functions using dask.

# %%
workers = 4

cluster = LocalCluster(
    n_workers=workers,
    processes=True,
)

client = Client(cluster)

print(client.dashboard_link)

# %% [markdown]
# # Requesting Weather
#
# Now that we have our coordinates, and dask client initialized we can make our parallel api calls using `pvdeg.weather.weather_distributed`.
#
# We want data from PVGIS so we will use that as the database. For more information about the databases look at the `pvdeg.weather.get` docstring. We will pass in the coordinates list which is a list of tuple pairs for the latitude and longiutudes to request weather at.
#
# `weather_ds` is the collected weather dataset from the API calls to PVGIS.
# `meta_df` is the collected meta dataframe from the API calls to PVGIS.
# `failed_gids` will be a list of the failed indexes from the input coordinates list that we could not get weather from using PVGIS. These may have failed randomly so it is worth trying again.

# %%
weather_ds, meta_df, failed_gids = pvdeg.weather.weather_distributed(
    database="PVGIS", coords=coords
)

# %% [markdown]
# # Viewing Result
#
# The result is stored in an xarray dataset with a dask array backend. This allows us to parallelize the computation/api requests but makes it a little harder to view the data. We can inspect the dataset using the following but we will not be able to inspect any values.
#
# ```
# weather_ds
# ```
#
# To load the values from the dask arrays we need to use `.compute()` as follows.
#

# %%
weather_ds.compute()

# %% [markdown]
# # Saving Geospatial Data Locally
#
# The goal of `pvdeg.store` is to create a living local database of meteoroligical data that grows overtime as your geospatial data needs grow. To do this `PVDeg` will save to a folder called `PVDeg-Meteorological` your user home directory. For me this is located at `C:\Users\tford\PVDeg-Meteorological`. This directory will contain a `zarr` store, this is a popular format for storing multi-dimensional array data, not dissimilar to `h5` files. It was chosen over `h5` because `zarr` stores arrays in chunked compressed files that make access very easy without opening an entire file like `h5`. This is an oversimplification of the design process but we felt `zarr` was a better fit.
#
# ## Store
#
# We can use `pvdeg.store.store` to save geospatial data to our living dataset in the common form provided by `pvdeg`. The data is stored in various groups and subfolders but they will be arranged based on the *source* and *periodicity*.
#
# For example:
#     - Hourly PVGIS data will be saved to a group called "PVGIS-1hr"
#     - 30 minute PVGIS to a group called "PVGIS-30min"
#     - 15 minute PVGIS will be saved to a group called "PVGIS-15min"

# %%
pvdeg.store.store(weather_ds, meta_df)

# %% [markdown]
# # Load
#
# `PVDeg` makes use of `dask` to handle larger than memory datasets. Trandionally, this was only useful in our HPC environment but as your local database grows overtime, it will eventually surpass the limits of your computer's volatile memory. Additionally, `dask` allows us to parallelize geospatial calculations via `pvdeg.geospatial.analysis`. This ability can be utilized on local machines or HPC clusters alike.
#
# `PVDeg` implements the ability to access your local living database via `pvdeg.store.get`. This method takes a string called `group`. Groups are created automatically in your store when you save data using `pvdeg.store.store`. As described in the `pvdeg.store.store` docstring and the *Store* section above, NSRDB will follow a similar scheme but it not implemented yet.
#     - Hourly PVGIS data will be saved to a group called "PVGIS-1hr"
#     - 30 minute PVGIS to a group called "PVGIS-30min"
#     - 15 minute PVGIS will be saved to a group called "PVGIS-15min"

# %% [markdown]
# # Load PVGIS-1hr Data
#
# The example below shows us loading the hourly tmy data from PVGIS that we gathered and saved to our zarr store in the above cells. This gets us the form of a weather xarray.Dataset (`geo_weather` in this example) and a metadata dataframe (`geo_meta` in this example).
#
# These can be treated like any other geospatial data shown in the `pvdeg` tutorials and tools or documentation.

# %%
geo_weather, geo_meta = pvdeg.store.get(group="PVGIS-1hr")

# %% [markdown]
# # Inspecting the Results
#
# explain *.compute() and dask here*

# %%
plt.plot(geo_weather.sel(gid=0).dni)

# %%
geo_meta

# %% [markdown]
# # Geospatial Calculations from Locally Stored Data
#
# As shown above we can load from our `zarr` store and treat it like any other geospatial data in `pvdeg`.
#
# For demonstration we can run the analysis below to estimate effective standoff height and operating temperatures for the provided data. It may look like the `geo_res` contains empty results but that is because we did not have input data for all of the points in the input grid (think of this as a 2D plane formed between the latitude and longitude axes). Clicking on the stack of three circles in the bottom cell will expand the datavariable (like an attribute of the multidimensional array structure) and show the results.
#
# Additionally, we can interpolate and plot the results.

# %%
func = pvdeg.standards.standoff

template = pvdeg.geospatial.auto_template(func=func, ds_gids=geo_weather)

geo_res = pvdeg.geospatial.analysis(
    weather_ds=geo_weather, meta_df=geo_meta, func=func, template=template
)

# %%
geo_res

# %% [markdown]
# This plot lacks information on the area and does not include some political boundary lines. For more information on plotting look at the `Scenario - Non-uniform Mountain Downselect.ipynb` tutorial in the tutorials and tools folder.

# %%
pvdeg.geospatial.plot_sparse_analysis_land(
    geo_res, data_var="T98_0", method="nearest", resolution=10j
)

# %% [markdown]
# # Growing Our Living Store
#
# What if we want to download more points from Europe? We can keep our old download in the store and shelve it to look at northern Europe.
#
# We will start by generating a range of points that cover Europe.

# %%
lon_EU = np.arange(-25.0, 51.0, 1)  # Adjusted for EU longitudes
lat_EU = np.arange(34.0, 73.0, 2)  # Adjusted for EU latitudes

# Create meshgrid for EU
lon_grid_EU, lat_grid_EU = np.meshgrid(lon_EU, lat_EU)

# Check land coverage in the EU
land_EU = globe.is_land(lat_grid_EU, lon_grid_EU)

# Extract land coordinates in the EU
lon_land_EU = lon_grid_EU[land_EU]
lat_land_EU = lat_grid_EU[land_EU]

# Define the Scan grid ranges
lon_Scan = np.arange(-10.5, 31.6, 0.3)
lat_Scan = np.arange(60, 71.2, 0.3)
lon_grid_Scan, lat_grid_Scan = np.meshgrid(lon_Scan, lat_Scan)
land_Scan = globe.is_land(lat_grid_Scan, lon_grid_Scan)

lon_land_Scan = lon_grid_Scan[land_Scan]
lat_land_Scan = lat_grid_Scan[land_Scan]

# %%
plt.scatter(lon_land_EU, lat_land_EU, c="r", s=1)

# %%
w, m, failed_gids = pvdeg.weather.weather_distributed(
    database="PVGIS", coords=[(lat_land_EU[0], lon_land_EU[0])]
)

# %%
pvdeg.store.store(weather_ds=w, meta_df=m)

# %%
loaded_weather, loaded_meta = pvdeg.store.get(group="PVGIS-1hr")

# %%
loaded_meta

# %%
loaded_weather

# %%
loaded_weather.sel(gid=22).compute()

# %%
import matplotlib.pyplot as plt

plt.plot(loaded_weather.sel(gid=0).dhi)

# %%
wet, met = pvdeg.store.get("PVGIS-1hr")

# %%
wet

# %%
loaded_meta

# %%
func = pvdeg.standards.standoff

template = pvdeg.geospatial.auto_template(func=func, ds_gids=loaded_weather)

loaded_geo_res = pvdeg.geospatial.analysis(
    weather_ds=loaded_weather, meta_df=loaded_meta, func=func, template=template
)

# %%
pvdeg.geospatial.plot_sparse_analysis_land(loaded_geo_res, data_var="T98_inf")

# %%
