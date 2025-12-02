# %%
import pvdeg
import numpy as np

# %% [markdown]
# # Adding Points
#
# We are going to add all of the points in the American West to the scenario and downsample by a factor of 1. This will include only half of the points in the latitude axis and half in the longitude axis.

# %%
dynamic_points = pvdeg.GeospatialScenario(name="dynamic-selection")

dynamic_points.addLocation(
    state=["CO", "UT"],  # , 'NM', 'NV', 'ID', 'WY', 'AZ', 'CA', 'OR', 'WA'],
    downsample_factor=1,
)

# %% [markdown]
# # Preview The Scenario's Points
#
# Use `plot_cords` to get a quick snapshot of all coordinates included in the scenario's metadata.

# %%
dynamic_points.plot_coords(
    coord_1=[48.574790, -130.253906],  # uncomment to see Larger scale view
    coord_2=[25.482951, -68.027344],
    size=0.005,
)

# %%
dynamic_points.meta_data

# %% [markdown]
# # Downselecting
#
# Using weighted random choices based on elevation we will create a sparse grid from the full metadata for fast calculations. This requires sklearn to be installed but this is not in the `pvdeg` dependency list to you will have to install it seperately.
#
# ## Normalization
#
# At each metadata point in our dataset we will calculate a weight based on its changes in elevation compared to its neighbors. The higher the weight, the greater the change in elevation from a point's immediate neighbors. The downselection methods and functions use these weights to randomly select a subset of the datapoints, prefferentially selecting those with higher weights.
#
# We have some control over which points get selected because all points' weights must be normalized (mapped from 0 to 1) before downselecting. We can apply a function such as $e^x$ or $\log x$ to the weights during normalization. This could help change the distribution of weights that are chosen. This could remove points from the mountains and add them to areas with fewer changes in elevation, or vice versa.
#
# *Note: `pvdeg`'s downselection functions use `numpy.random`, the random seed is not fixed so the result will change between runs.*

# %% [markdown]
# # Providing a KdTree
#
# As shown below the lines to create a kdtree are commented out.
#

# %%
# Set random seed for reproducible results
np.random.seed(42)

# west_tree = pvdeg.geospatial.meta_KDtree(meta_df=dynamic_points.meta_data)

dynamic_points.downselect_elevation_stochastic(
    # kdtree=west_tree,
    downselect_prop=0.5,
    normalization="linear",
)

# %%
dynamic_points.plot_coords()

# %% [markdown]
# # Extracting from Scenario
#
# Scenarios provide an easy way to select and downsample geospatial data but we can easily pull out the data to use other `pvdeg` functions on it. In the cell below, we extract the weather data and meta data from the scenario and take only the matching entries from the weather. Then we load the xarray dataset into memory. Previously, it was stored lazily out of memory but we want to do operations on it. (Chunking causes issues when calculating so this eliminates any chunks)

# %%
weather = dynamic_points.weather_data

sub_weather = weather.sel(
    gid=dynamic_points.meta_data.index
)  # downselect weather using restricted metadata set

sub_weather = sub_weather.compute()  # load into memory

# %% [markdown]
# # Geospatial Calculation
#
# Run a standoff calculation on the extracted scenario weather data and scenario meta data.

# %%
# geospatial analysis now

geo = {
    "func": pvdeg.standards.standoff,
    "weather_ds": sub_weather,
    "meta_df": dynamic_points.meta_data,
}

analysis_result = pvdeg.geospatial.analysis(**geo)

# %% [markdown]
# # Viewing Results
#
# Inspecting the xarray dataset below shows us that we have many Not a Number (NaN) entries. These occur because we did not provide weather data at every point in the grid of possile latitude-longitude pairs. Expanding the `x` datavariable shows that there are some valid results but these are uncommon.

# %%
analysis_result

# %% [markdown]
# # Plotting Sparse Data I
#
# If we try to plot existing data with the current plotting methods exposed by `pvdeg` we will encounter issues. This will produce weak plotting results.

# %%
# This cell demonstrates the issue with plotting sparse data directly
# It will raise a TypeError because there's no numeric data to plot
try:
    pvdeg.geospatial.plot_USA(analysis_result["x"])
except TypeError as e:
    print(f"Expected error when plotting sparse data: {e}")
    print("This is why we need to use plot_sparse_analysis() instead (see next cell)")

# %%
pvdeg.geospatial.plot_sparse_analysis(analysis_result, data_var="x", method="linear")
