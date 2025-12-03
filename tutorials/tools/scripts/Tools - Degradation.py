# %% [markdown]
# # Degradation and Acceleration Factors
# This tool will provide a simple method for estimating degradation and for calculating acceleration factors. It interfaces with the degradation database to simplify acquisition of the degradation parameters.
#
# **Requirements**:
# - compatible weather file (e.g., PSM3, TMY3, EPW...)
# - Accelerated testing chamber parameters
#     - chamber irradiance [W/m^2]
#     - chamber temperature [C]
#     - chamber humidity [%]
#     - & etc.
# - Activation energies for test material [kJ/mol]
# - Other degradation parameters
#
# **Objectives**:
# 1. Read in the weather data
# 2. Gather basic degradation modeling data for a material of interest
# 3. Calculate absolute degradation rate
# 4. Run Monte Carlo simulation at a single site
# 5. Generate chamber or field data for environmental comparison
# 6. Calculate degradation acceleration factor of field location to chamber (or another location)
# 7. Produce a map of acceleration factors for a geographic region
#     Select a geographic region of interest
#     downsample to select specific site coordinates
#     Download or access the meteorological data for the chosen site coordinates.
#     Run the calculation

# %%
# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# # !pip install pvdeg

# %%
import os
import pvdeg
import pandas as pd
import numpy as np
from pvdeg import DATA_DIR
import json
from IPython.display import display, Math

import pvlib

print(pvlib.__version__)
from pvlib import iotools

# %%
# This information helps with debugging and getting support :)
import sys, platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)
print(DATA_DIR)

# %% [markdown]
# ## 1. Read In the Weather Data
#
# The function has these minimum requirements when using a weather data file:
# - Weather data containing (at least) DNI, DHI, GHI, Temperature, RH, and Wind-Speed data at module level.
# - Site meta-data containing (at least) latitude, longitude, and time zone
#
# Alternatively one may can get meterological data from the NSRDB or PVGIS with just the longitude and latitude. This function for the NSRDB (via NSRDB 'PSM3') works primarily for most of North America and South America. PVGIS works for most of the rest of the world (via SARAH 'PVGIS'). See the tutorial "Weather Database Access.ipynb" tutorial on PVdeg or Jensen et al. https://doi.org/10.1016/j.solener.2023.112092 for satellite coverage information.

# %%
# Get data from a supplied data file (Do not use the next box of code if using your own file)
weather_file = os.path.join(DATA_DIR, "psm3_demo.csv")
weather_df, meta = pvdeg.weather.read(weather_file, "csv", find_meta=True)
print(weather_file)
print(meta)

# %%
# This routine will get a meteorological dataset from anywhere in the world where it is available
# weather_id = (24.7136, 46.6753) #Riyadh, Saudi Arabia
# weather_id = (35.6754, 139.65) #Tokyo, Japan
# weather_id = (-43.52646, 172.62165) #Christchurch, New Zealand
# weather_id = (64.84031, -147.73836) #Fairbanks, Alaska
# weather_id = (65.14037, -21.91633) #Reykjavik, Iceland
weather_id = (33.4152, -111.8315)  # Mesa, Arizona
# weather_id = (0,0) # Somewhere else you are interested in.
weather_df, meta = pvdeg.weather.get_anywhere(id=weather_id)
print(meta)
# display(weather_df)

# %% [markdown]
# #### POA Irradiance
# Next we need to calculate the stress parameters including temperature and humidity. We start with POA irradiance.
# Irradiance_kwarg governs the array orientation for doing the POA calculations.
# It is defaulted to a north-south single axis tracking. A fixed tilt set of parameters is included but is blocked out.
# Look in spectral.py and/or PVLib here for 1-axis kwargs,
# https://pvlib-python.readthedocs.io/en/v0.7.2/generated/pvlib.tracking.singleaxis.html#pvlib.tracking.singleaxis
# and for fixed tilt,
# https://pvlib-python.readthedocs.io/en/v0.7.2/generated/pvlib.irradiance.gti_dirint.html?highlight=poa .
# Here, the POA global calculation is appended to the meteorolgical data dataframe.

# %%
# irradiance_kwarg ={
# "tilt": None,
# "azimuth": None,
# "module_mount": 'fixed'}
irradiance_kwarg = {"axis_tilt": 0, "axis_azimuth": 180, "module_mount": "single_axis"}
poa_df = pvdeg.spectral.poa_irradiance(
    weather_df=weather_df, meta=meta, **irradiance_kwarg
)

weather_df["poa_global"] = poa_df["poa_global"]

# %% [markdown]
#
# #### Get Spectrally Resolved Irradiance Data
# This first set of commands will calculate spectrally resolved irradiance data. This may or may not be needed for a given degradation model and can be skipped here.

# %%
# this whole block needs to be replaced with call to calculate spectrally resolved irradiance.

from pvdeg import TEST_DATA_DIR

INPUT_SPECTRA = os.path.join(TEST_DATA_DIR, r"spectra_pytest.csv")
data = pd.read_csv(INPUT_SPECTRA)
# display(data)
print(INPUT_SPECTRA)

# Test function
# cusotm_albedo['Summer']
# custom_albedo['Winter']
# custom_albedo['Snow']
# defaults - Grass, Dry Grass, Snow
# Flexible to add complexity later
# merge in development branch changes
# KGPCY Python package
custom_albedo_summer = "A006"
custom_albedo_winter = {  # required: startDate, wavelength (if len(albedo) > 1), albedo, isSnow defaults to False
    "data_entry_person": "Michael Kempe",
    "date_entered": "7/28/2025",
    "DOI": "10.3390/ijerph15071507",
    "source_title": "Ultraviolet Radiation Albedo and Reflectance in Review: The Influence to Ultraviolet Exposure in Occupational Settings",
    "authors": "Joanna Turner, Alfio V. Parisi",
    "reference": "Turner J, Parisi AV. Ultraviolet Radiation Albedo and Reflectance in Review: The Influence to Ultraviolet Exposure in Occupational Settings. Int J Environ Res Public Health. 2018 Jul 17;15(7):1507.",
    "keywords": "snow, ground",
    "months": "1,2,3,10,11,12",
    "startDate": "January 1",  # Day of Year? 0-365
    "HourOfYear": "1",  # Hour of Year? 1-8760
    "isSnow": "False",
    "comments": "Data is emperically extrapolated from 280 nm to 297 nm. Data extracted from Turner et al. Figure 1 as a reference to Doda & Green Snow-Ground. Doda D., Green A. Surface Reflectance Measurements in the UV from an Airborne Platform. Part 1. Appl. Opt. 1980;19:2140-2145. doi: 10.1364/AO.19.002140. Doda D., Green A. Surface Reflectance Measurements in the Ultraviolet from an Airborne Platform. Part 2. Appl. Opt. 1981;20:636-642. doi: 10.1364/AO.20.000636.",
    "wavelength": "280, 297.32034, 300.02435, 301.8514, 305.79782, 310.10962, 313.21558, 317.4543, 322.86237, 329.9513, 331.19366, 339.89038, 343.06943, 350.34103, 360.02435, 369.96347, 380.3776, 386.77222, 390.0609, 400.14615",
    "albedo": "20, 29.515152, 28.30303, 29.454546, 28.90909, 34.696968, 36.757576, 39.363636, 39.21212, 38.60606, 41.272728, 40.909092, 42.242424, 42.21212, 40.575756, 43.21212, 43.090908, 43.454544, 43.60606, 39.757576",
}
# Startdate, albedo, wavelength -> then next one + boolean logic for snow ()
custom_albedo_snow = {}
# custom_albedo_snow
spectra_folder = "spectra"  # If you have already pulled the spectra from SMARTS, pass the folder path to avoid going through the donwload process again.
wavelengths = np.arange(
    280, 400, 25
)  # Example wavelengths from 280 nm to 400 nm in steps of 25 nm
# data = pvdeg.spectral.spectrally_resolved_irradiance(weather_df=weather_df, meta=meta, wavelengths=wavelengths, frontResultsOnly=None,
#                                                     spectra_folder=spectra_folder, custom_albedo_summer=custom_albedo_summer, custom_albedo_winter=custom_albedo_winter, **irradiance_kwarg)
# return front, back, or both (True, False, None)
# bool frontResultsonly = True for front only
# separate columns for front and back irradiance: spectra_front: etc. , spectra_back: etc. (see spectra_pytest.csv)
# Check albedo boolean snow, winter non-snow, summer non-snow


# %% [markdown]
# #### Get Cell Temperature and Module Surface Temperature
# The following will calculate the cell and module surface temperature using the King model as a default. Other models can be used as described at,
# https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html. The difference is less than one °C for ground mounted systems
# but can be as high as 3 °C for a high temperature building integrated system.
#
# Here the temperatures are added to the dataframe and the 'temp_module' temperature is selected as the default 'temperature' for the degradation calculations. If it is a cell degradation that is being investigated, 'temp_cell' should be used for the temperature.

# %%
temp_cell = pvdeg.temperature.cell(weather_df=weather_df, meta=meta, poa=poa_df)
temp_module = pvdeg.temperature.module(weather_df=weather_df, meta=meta, poa=poa_df)

weather_df["temp_cell"] = temp_cell
weather_df["temp_module"] = temp_module

weather_df["temperature"] = weather_df["temp_module"]
# weather_df['temperature'] = weather_df['temp_cell']

# %% [markdown]
# #### Humidity
# Depending on the component for which the calculation is being run on, the desired humidity may be the atmospheric humidity, the module surface humidity, the humidity in front of a cell with a permeable backsheet, the humidity in the backsheet, the humidity in the back encapsulant or another custom humidity location such as a diffusion limited location. The folowing are options for doing all of these calculations. Here all the different humidities are put in the weather_df dataframe, but to select one to be specifically used it should be named 'RH' for most degradation functions (check the documentation of a specific degradation calculation if in doubt). Here the surface humidity is selected as a default.
#
# Append the calculated values into the weather DataFrame.
# Note: putting the values into the weather_df DataFrame is not strictly necessary, but may be convenient for later use in the degradation calculations.

# %%
RH_module = pvdeg.humidity.module(
    weather_df=weather_df,
    poa=poa_df,
    temp_module=temp_module,
    backsheet="W017",
    backsheet_thickness=0.30,
    encapsulant="W001",
    back_encap_thickness=0.50,
)

weather_df = pd.concat([weather_df, RH_module], axis=1)

# %% [markdown]
# Each of the necessary arrays of data can be individually sent to a function for calculation in the function call, or they can be combined into a single dataframe. The degradation functions are set up to first check for a specific data set in the function call but if not found it looks for specific data or a suitable substitute in the weather dataframe.
#
# You can select one of the RH values to be used as the relative humidity in the degradation model calculations by assigning it to to column "RH" in the dataframe.
# Alternatively, the "RH" data can be sent to the degradation function explicitly in the function call.

# %%

weather_df["RH"] = RH_module["RH_surface_outside"]
# weather_df['RH'] = RH_module['RH_front_encap']
# weather_df['RH'] = RH_module['Ce_back_encap']
# weather_df['RH'] = RH_module['RH_back_encap']
# weather_df['RH'] = RH_module['RH_backsheet']

# %% [markdown]
# ## 2. Gather Basic Degradation Modeling Data for a Material of Interest
#
# First we need to gather in the parameters for the degradation process of interest. This includes things such as the activiation energy and parameters defining the sensitivity to moisture, UV light, voltage, and other stressors.
# For this tutorial we will need solar position, POA, PV cell and module temperature. Let's gernate those individually with their respective functions.
# The blocked out text will produce a list of key fields from the database for each entry.

# %%
# kwarg_variables = pvdeg.utilities._read_material(name=None, fname="DegradationDatabase", item=("Material", "Equation", "KeyWords", "EquationType"))
# print(json.dumps(kwarg_variables, skipkeys = True, indent = 0 ).replace("{" + "\n", "{").replace('\"' + "\n", "\"").replace(': {' , ':' + "\n" + "{").replace('},' + "\n", '},' +'\n' +'\n'))
pvdeg.utilities.display_json(pvdeg_file="DegradationDatabase", fp=DATA_DIR)

# %% [markdown]
# This next set of codes will take the data from the extracted portion of the Json library and create a list of variables from it. If more variables need to be modified or added, this is where it should be done.

# %%
deg_data = pvdeg.utilities.read_material(
    fp=DATA_DIR, key="D036", pvdeg_file="DegradationDatabase"
)
display(deg_data)

# %% [markdown]
# Here we pull out the relevant equation code identifier needed for running the calculations.

# %%
func = "pvdeg.degradation." + deg_data["EquationType"]
print(func)
display(Math("\\Large " + deg_data["Equation"]))

# %% [markdown]
# ## 3. Calculate Absolute Degradation Rate
#
# To do this calculation, we must have degradation parameter data for a process that is complete with all the necessary variables.

# %%
func_call = getattr(pvdeg.degradation, deg_data["EquationType"])
degradation = func_call(weather_df=weather_df, parameters=deg_data)
print(
    "Average degradation rate for a year", degradation / 8760, deg_data["R_0"]["units"]
)

# %%
