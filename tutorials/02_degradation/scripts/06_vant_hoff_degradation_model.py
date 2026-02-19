# %% [markdown]
# # Van't Hoff Degradation Model
# ## Calculate site specific degradation according to the Van't Hoff equation
#
# Michael Kempe
#
# 2023.08.31
#
# **Requirements**:
# - compatible weather file (PSM3, TMY3, EPW) or lattitude and longitude of desired site
# - Accelerated testing chamber parameters
#     - chamber irradiance [W/m^2]
#     - chamber temperature [°C]
# - 10°C acceleration factor
#
# **Steps**:
# 1. Read/find the weather data
# 2. Generate basic modeling data
# 3. Calculate VantHoff degradation acceleration factor
# 4. Expand calculations to a region

# %%
# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# pip install pvdeg

# %%
import os

import pvdeg
from pvdeg import DATA_DIR

# %% [markdown]
# # 1. Read In the Weather File
#
# This is usually the first step. Use a PSM3, TMY3, or EPW file. For this demo, use the provided PSM3 weather file.

# %%
PSM_FILE = os.path.join(DATA_DIR, "psm3_demo.csv")
WEATHER, META = pvdeg.weather.read(PSM_FILE, "psm")
print(
    "Latitude =",
    META["latitude"],
    "Longitude =",
    META["longitude"],
    META["Country"],
    META["City"],
)

# %% [markdown]
# # 2. Generate Basic Modeling Data
#
# For this tutorial we will need solar position, POA, PV cell and module temperature. Let's gernate those individually with their respective functions.

# %%
sol_pos = pvdeg.spectral.solar_position(weather_df=WEATHER, meta=META)

poa_df = pvdeg.spectral.poa_irradiance(
    weather_df=WEATHER, meta=META, sol_position=sol_pos
)

temp_cell = pvdeg.temperature.cell(weather_df=WEATHER, meta=META, poa=poa_df)

temp_module = pvdeg.temperature.module(weather_df=WEATHER, meta=META, poa=poa_df)

# %% [markdown]
# # 3. VantHoff Degradation
#
# Van't Hoff Irradiance Degradation Equation:
# $$ R_o = R_D ·  G^p  · T_f^{\frac{T}{10} }$$
#
# For the yearly average degredation outdoors to be the same as the controlled environmnet, the lamp settings will need to be set to *G$_{WA}$* and the temperature set to *T$_{oeq}$*.
#
# As with most `pvdeg` functions, the following functions will always require two arguments (weather_df and meta)

# %%
# chamber irradiance (W/m²)
I_chamber = 1600
# chamber temperature (°C)
temp_chamber = 85
# Schwartzchild Coefficient
p = 0.64
# Acceleration factor for every 10°C
Tf = 1.41

# calculate the Van't Hoff Acceleration factor
vantHoff_deg = pvdeg.degradation.vantHoff_deg(
    weather_df=WEATHER,
    meta=META,
    I_chamber=I_chamber,
    temp_chamber=temp_chamber,
    poa=poa_df,
    temp=temp_cell,
    p=p,
    Tf=Tf,
)

# calculate the Van't Hoff weighted irradiance
irr_weighted_avg_v = pvdeg.degradation.IwaVantHoff(
    weather_df=WEATHER, meta=META, poa=poa_df, temp=temp_cell, p=p, Tf=Tf
)

print(
    "AF =",
    round(vantHoff_deg, 1),
    "(°C) , and G_WA =",
    round(irr_weighted_avg_v),
    "(W/m²)",
)

# %% [markdown]
# # 4. Arrhenius
# Calculate the Acceleration Factor between the rate of degredation of a modeled environmnet versus a modeled controlled environmnet
#
# Example: "If the *AF*=25 then 1 year of Controlled Environment exposure is equal to 25 years in the field"
#
# Equation:
# $$ AF = N · \frac{ G_{chamber}^x · RH_{chamber}^n · e^{\frac{- E_a}{k T_{chamber}}} }{ \Sigma (G_{POA}^x · RH_{outdoor}^n · e^{\frac{-E_a}{k T_outdoor}}) }$$
#
# Function to calculate *G$_{WA}$*, the Environment Characterization (W/m²). If the controlled environmnet lamp settings are set at *G$_{WA}$*, and the temperature set to *T$_{eq}$*, then the degradation will be the same as the yearly average outdoors.
#
# Equation:
# $$ G_{WA} = [ \frac{ \Sigma (G_{outdoor}^x · RH_{outdoor}^n e^{\frac{-E_a}{k T_{outdood}}}) }{ N · RH_{WA}^n · e^{- \frac{E_a}{k T_eq}} } ]^{\frac{1}{x}} $$

# %%
# relative humidity within chamber (%)
rh_chamber = 15
# arrhenius activation energy (kj/mol)
Ea = 40

rh_surface = pvdeg.humidity.surface_relative(
    rh_ambient=WEATHER["relative_humidity"],
    temp_ambient=WEATHER["temp_air"],
    temp_module=temp_module,
)

arrhenius_deg = pvdeg.degradation.arrhenius_deg(
    weather_df=WEATHER,
    meta=META,
    rh_outdoor=rh_surface,
    I_chamber=I_chamber,
    rh_chamber=rh_chamber,
    temp_chamber=temp_chamber,
    poa=poa_df,
    temp=temp_cell,
    Ea=Ea,
)

irr_weighted_avg_a = pvdeg.degradation.IwaArrhenius(
    weather_df=WEATHER,
    meta=META,
    poa=poa_df,
    rh_outdoor=WEATHER["relative_humidity"],
    temp=temp_cell,
    Ea=Ea,
)

# %% [markdown]
# # 5. Quick Method (Degradation)
#
# For quick calculations, you can omit POA and both module and cell temperature. The function will calculate these figures as needed using the available weather data with the default options for PV module configuration.

# %%
# chamber settings
I_chamber = 1000
temp_chamber = 60
rh_chamber = 15

# activation energy
Ea = 40

vantHoff_deg = pvdeg.degradation.vantHoff_deg(
    weather_df=WEATHER, meta=META, I_chamber=I_chamber, temp_chamber=temp_chamber
)

irr_weighted_avg_v = pvdeg.degradation.IwaVantHoff(weather_df=WEATHER, meta=META)

# %%
rh_surface = pvdeg.humidity.surface_relative(
    rh_ambient=WEATHER["relative_humidity"],
    temp_ambient=WEATHER["temp_air"],
    temp_module=temp_module,
)

arrhenius_deg = pvdeg.degradation.arrhenius_deg(
    weather_df=WEATHER,
    meta=META,
    rh_outdoor=rh_surface,
    I_chamber=I_chamber,
    rh_chamber=rh_chamber,
    temp_chamber=temp_chamber,
    Ea=Ea,
)

irr_weighted_avg_a = pvdeg.degradation.IwaArrhenius(
    weather_df=WEATHER, meta=META, rh_outdoor=WEATHER["relative_humidity"], Ea=Ea
)

# %% [markdown]
# # 6. Solder Fatigue
#
# Estimate the thermomechanical fatigue of flat plate photovoltaic module solder joints over the time range given using estimated cell temperature. Like other `pvdeg` funcitons, the minimal parameters are (weather_df, meta). Running the function with only these two inputs will use default PV module configurations ( open_rack_glass_polymer ) and the 'sapm' temperature model over the entire length of the weather data.

# %%
fatigue = pvdeg.fatigue.solder_fatigue(weather_df=WEATHER, meta=META)

# %% [markdown]
# If you wish to reduce the span of time or use a non-default temperature model, you may specify the parameters manually. Let's try an explicit example.
# We want the solder fatigue estimated over the month of June for a roof mounted glass-front polymer-back module.
#
# 1. Lets create a datetime-index for the month of June.
# 2. Next, generate the cell temperature. Make sure to explicity restrict the weather data to our dt-index for June. Next, declare the PV module configuration.
# 3. Calculate the fatigue. Explicity specify the time_range (our dt-index for June from step 1) and the cell temperature as we caculated in step 2

# %%
# select the month of June
time_range = WEATHER.index[WEATHER.index.month == 6]

# calculate cell temperature over our selected date-time range.
# specify the module configuration
temp_cell = pvdeg.temperature.cell(
    weather_df=WEATHER.loc[time_range],
    meta=META,
    temp_model="sapm",
    conf="insulated_back_glass_polymer",
)


fatigue = pvdeg.fatigue.solder_fatigue(
    weather_df=WEATHER, meta=META, time_range=time_range, temp_cell=temp_cell
)
