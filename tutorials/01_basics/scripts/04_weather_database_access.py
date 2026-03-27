# %% [markdown]
# # Weather Database Access
#
# **Requirements:**
# - Internet access
# - NSRDB API key. API keys are free. You can request and obtain an API key in about 5 minutes. To get your own key, visit https://developer.nrel.gov/signup/
# - Step **1.** is for Kestrel HPC users. You will need an account with NREL's Kestrel computer for this method.
#
# **Objectives:**
#
# Using direct access to large scale weather databases, we're going to estimate the minimum standoff distance for a roof mounted PV system. We'll do this in 3 ways using both the NSRDB and PVGIS database.
# 1. Single Location, NSRDB via NREL's high performance computer, Kestrel.
# 2. Single Location via NSRDB public API key.
# 3. Single Location via the PVGIS public database
#
# **Background:**
#
# This journal will demonstrate all existing built-in methods for directly accessing public weather databases. Some methods are restriced to certain user groups. For general users, see methods **2** and **3**. For users with an active Kestrel HPC account, you may use method **1** as well as **2** and **3**.
#
# For all users and all steps: This journal will run significantly longer than other tutorials and have significant internet traffic as you fetch large datasets.

# %% [markdown]
# This example demonstrates the calculation of a minimum standoff distance necessary for roof-mounted PV modules to ensure that the $T_{98}$ operational temperature remains under 70°C, in which case the more rigorous thermal stability testing requirements of IEC TS 63126 would not needed to be considered. We use data from [Fuentes, 1987] to model the approximate exponential decay in temperature, $T(X)$, with increasing standoff distance, $X$, as,
#
# $$ X = -X_0 \ln\left(1-\frac{T_0-T}{\Delta T}\right)$$
#
# where $T_0$ is the temperature for $X=0$ (insulated back) and $\Delta T$ is the temperature difference between an insulated back ($X=0$) and open rack mounting configuration ($X=\infty)$.
#
# The following figure showcases this calulation for the entire United States. We used pvlib and data from the National Solar Radiation Database (NSRDB) to calculate the module temperatures for different mounting configuration and applied our model to obtain the standoff distance for roof-mounted PV systems.

# %% [markdown]
# # Single location example

# %%
import pvdeg
import pandas as pd

# %% [markdown]
# # 1. NSRDB - HSDS on Kestrel
#
# This method requires a direct connection to NREL's high performance computer "Kestrel". If you are not running this journal from Kestrel, skip this section and proceed to section **2.**
#
# In this step:
#
# First we select a database. Here, we will use the NSRDB. Since we are modeling a single location, we can pass the `weather_id` as tuple (lat, long). A location gid can be used as well. 'gid' is a unique identifier to a geographic location within the NSRDB. We'll look at how to find gids later on.
#
# Next, we want to select a satellite, named dataset (year of data), and what weather attributes we want to fetch. For further options, see the documentation for `pvdeg.weather.get`

# %% [markdown]
# `pvdeg.weather.get` returns the same variables as `weather.read` which we have used in each journal before this. We get a weather DataFrame and a meta-data dicitonary. Each contains a minimum of consistent fields, but may have additional fields based on the database accessed or the attributes requested.
#
# Lets verify the weather data we fetched by running a familiar calculation; standoff distance.

# %%
# Load pre-saved weather data for this tutorial
# This avoids API rate limits during testing and builds
import json

weather_df = pd.read_csv("../data/psm4_golden.csv", index_col=0, parse_dates=True)
with open("../data/meta_golden.json", "r") as f:
    meta = json.load(f)

# Uncomment below to fetch fresh data with your own API key:
# API_KEY = "your_api_key_here"
# # The example API key here is for demonstation and is rate-limited per IP.
# # To get your own API key, visit https://developer.nrel.gov/signup/
# weather_db = "PSM4"
# weather_id = (39.741931, -105.169891)
# weather_arg = {
#     "api_key": "DEMO_KEY",
#     "email": "user@mail.com",
#     "map_variables": True,
# }
# weather_df, meta = pvdeg.weather.get(weather_db, weather_id, **weather_arg)

# Perform calculation and output interpretation or results
res = pvdeg.standards.standoff(
    weather_df=weather_df,
    meta=meta,
    tilt=None,
    azimuth=180,
    sky_model="isotropic",
    temp_model="sapm",
    conf_0="insulated_back_glass_polymer",
    conf_inf="open_rack_glass_polymer",
    T98=70,
    x_0=6.5,
    wind_factor=0.33,
)
print(pvdeg.standards.interpret_standoff(res))

# Clean metadata for consistent output
meta_clean = {k: v for k, v in meta.items() if k not in ["irradiance_time_offset"]}
print(meta_clean)

# %%
# weather_db = "PVGIS"
# # weather_id = (39.741931, -105.169891)
# weather_id = (24.7136, 46.6753)  # Riyadh, Saudi Arabia
# # weather_arg = {'map_variables': True}

# # TMY
# weather_df, meta = pvdeg.weather.get(weather_db, weather_id)

# # Perform calculation
# res = pvdeg.standards.standoff(
#     weather_df=weather_df,
#     meta=meta,
#     tilt=None,
#     azimuth=180,
#     sky_model="isotropic",
#     temp_model="sapm",
#     conf_0="insulated_back_glass_polymer",
#     conf_inf="open_rack_glass_polymer",
#     T98=70,
#     x_0=6.5,
#     wind_factor=0.33,
# )
# print(pvdeg.standards.interpret_standoff(res))

# # Clean metadata for consistent output (remove variable fields)
# meta_clean = {k: v for k, v in meta.items() if k not in ["irradiance_time_offset"]}
# print(meta_clean)
