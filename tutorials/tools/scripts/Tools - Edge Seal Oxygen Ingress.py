# %% [markdown]
# # Edge Seal Oxygen Ingress Calculator tool
#
# ## Calculation of oxygen ingress profile through an edge seal and into the encapsulant.
#
# **Requirements:**
# - Local weather data file or site longitude and latittude.
# - Properties and dimensions of the edge seal.
#
# **Objectives:**
# 1. Import weather data.
# 2. Set up the calculations.
# 3. Calculate oxygen ingress into an edge seal.
# 3. Incorporate an oxygen consumption model.
# 4. Plot the data.
#
# **Background:**
#
# This performs a 1-D finite difference model for oxygen ingress through an edge seal and into an encapsulant. This is effectively an infinitely long module with a prescribed width.The output is then displayed graphically.

# %%
# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# # !pip install pvdeg

# %%
import os
import pvdeg
import pandas as pd
from pvdeg import DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
import json

# %%
# This information helps with debugging and getting support :)
import sys, platform

print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("pvdeg version ", pvdeg.__version__)

# %% [markdown]
# ## 1. Import Weather Data
#
# The function has these minimum requirements when using a weather data file:
# - Weather data containing (at least) DNI, DHI, GHI, Temperature, RH, and Wind-Speed data at module level.
# - Site meta-data containing (at least) latitude, longitude, and time zone
#
# Alternatively one may can get meterological data from the NSRDB or PVGIS with just the longitude and latitude. This function for the NSRDB (via NSRDB 'PSM3') works primarily for most of North America and South America. PVGIS works for most of the rest of the world (via SARAH 'PVGIS'). See the tutorial "Weather Database Access.ipynb" tutorial on PVdeg or Jensen et al. https://doi.org/10.1016/j.solener.2023.112092 for satellite coverage information.

# %%
# Get data from a supplied data file (Do not use the next box of code if using your own file)
weather_file = os.path.join(DATA_DIR, "psm3_demo.csv")
weather_df, meta = pvdeg.weather.read(weather_file, "psm")
print(sorted(meta.keys()))

# %%
# This routine will get a meteorological dataset from anywhere in the world where it is available
# weather_id = (24.7136, 46.6753) #Riyadh, Saudi Arabia
# weather_id = (35.6754, 139.65) #Tokyo, Japan
# weather_id = (-43.52646, 172.62165) #Christchurch, New Zealand
# weather_id = (64.84031, -147.73836) #Fairbanks, Alaska
# weather_id = (65.14037, -21.91633) #Reykjavik, Iceland
weather_id = (33.4152, -111.8315)  # Mesa, Arizona
weather_df, meta = pvdeg.weather.get_anywhere(id=weather_id, database="PVGIS")
print(meta)

# %%
# This computes a module temperature. Here the default is an open rack system, but other options include:
#       'open_rack_glass_glass',
#       'close_mount_glass_glass',
#       'insulated_back_glass_polymer'

temperature = pvdeg.temperature.temperature(
    weather_df=weather_df,
    meta=meta,
    cell_or_mod="module",
    temp_model="sapm",
    conf="open_rack_glass_polymer",
)

temperature = pd.DataFrame(temperature, columns=["module_temperature"])
temperature["time"] = list(range(8760))

# %% [markdown]
# ## 2. Set up the Calculations
#
# There is a library of some materials and the relevant oxygen ingress parameters that can be used.

# %%
es = "OX005"  # This is the number for the edge seal in the json file
enc = "OX003"  # This is the number for the encapsulant in the json file
esw = 1.5  # This is the edge seal width in [cm]
encw = 10  # This is the encapsulant width in [cm]
sn = 20  # This is the number of edge seal nodes to use
en = 50  # This is the number of encapsulant nodes to use
pressure = 0.2109 * (1 - 0.0065 * meta.get("altitude") / 288.15) ** 5.25588
print(
    pvdeg.utilities.read_material(
        pvdeg_file="O2permeation", key="OX003", values_only=True
    )
)
print(
    pvdeg.utilities.read_material(
        pvdeg_file="H2Opermeation", key="W003", values_only=True
    )
)

# %% [markdown]
# ## 3. Run the Calculations
#
# This runs the calculations for diffusion using a simple 1-D finite difference calculation. The first set of calculations is just for diffusion, then the next two (when written) will include some consumption of oxygen. In typical PV applications, it is common for oxygen ingress distance to be limited by its consumption rate in the encapsulant.

# %%
oxygen_profile = pvdeg.diffusion.esdiffusion(
    temperature=temperature,
    edge_seal=es,
    encapsulant=enc,
    edge_seal_width=esw,
    encapsulant_width=encw,
    seal_nodes=sn,
    encapsulant_nodes=en,
    press=pressure,
    repeat=2,
)

# %%
# This sets up an a variable with the output folder information.
output_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.getcwd())), "TEMP", "results"
)
try:
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")
except OSError as error:
    print(error)

# %%
n_lines = 10
times = oxygen_profile.index.tolist()
for index in range(n_lines):
    plt.plot(
        oxygen_profile.iloc[
            int(np.trunc((index + 1) * (len(oxygen_profile) - 1) / n_lines))
        ],
        label=np.round(
            times[int(np.trunc((index + 1) * ((len(oxygen_profile) - 1) / n_lines)))]
            / 365.25
            / 24,
            2,
        ),
    )
plt.legend(title="Time [year]")
plt.ylabel("Oxygen Concentration [g/cm³]")
plt.xlabel("Distance From Edge [cm]")
plt.ticklabel_format(axis="y", style="plain")

plt.savefig(
    os.path.join(output_folder, "Edge_Seal_O2_ingress.png"), bbox_inches="tight"
)  # Creates an image file of the standoff plot
plt.show()

# %% [markdown]
# ## 5. Save data outputs.
#
# This cell contains a number of pre-scripted commands for exporting and saving data. The code to save plots is located after the plot creation. First check that the output folder exists.

# %%
fpath = os.path.join(DATA_DIR, "O2permeation.json")
with open(fpath) as f:
    data = json.load(f)
f.close()

material_list = ""
for key in data:
    if "name" in data[key].keys():
        material_list = material_list + key + "=" + data[key]["name"] + "\n"
material_list = material_list[0 : len(material_list) - 1]
print(material_list)

# %%
print("Your results will be stored in %s" % output_folder)
print("The folder must already exist or the file will not be created")

# Writes the meterological data to an *.csv file.
pvdeg.weather.write(
    data_df=weather_df,
    metadata=meta,
    savefile=os.path.join(output_folder, "WeatherFile.csv"),
)

# Writes a file with the edge seal oxygen profile calculations.
pd.DataFrame(oxygen_profile).to_csv(
    os.path.join(output_folder, "ES_Oxygen_profile.csv")
)

# Writes a file with temperature data used in the model calculations.
pd.DataFrame(temperature).to_csv(
    os.path.join(output_folder, "ES_Temperature_profile.csv")
)
