# %% [markdown]
# # Acetic Acid in EVA
# ## Acetic acid generation and power loss from electrode corrosion
#
# Md Habibur Rahman
#
# 2026.11.11
#
# Acetic acid forms when EVA takes up water, and it corrodes the cell contacts.
# `pvdeg.acetic` holds the model of Irikawa et al. (2025),
# [doi:10.1016/j.solmat.2024.113375](https://doi.org/10.1016/j.solmat.2024.113375).
# Each step below is checked against the numbers printed in the paper.
#
# **Requirements:** `pvdeg`, `numpy`, `pandas`

# %%
# if running on google colab, uncomment the next line and execute this cell to install the dependencies and prevent "ModuleNotFoundError" in later cells:
# # !pip install pvdeg

# %%
import os
import json
import numpy as np
import pandas as pd
import pvdeg
from pvdeg import acetic, DATA_DIR

# %% [markdown]
# ## Moisture in the EVA
#
# Eq. 3: $C_{w,\mathrm{sat}} = P_{\mathrm{H_2O}}\,S_0\,\exp(E_s/k_BT)$ with
# $S_0 = 0.1$ µg g⁻¹ atm⁻¹ and $E_s = 0.33$ eV.
# The paper gives 0.45 eV for the vapor pressure, 0.11 eV for $C_{w,\mathrm{sat}}$
# at 100 % RH, and 3000 µg/g at 85 °C, 100 % RH.

# %%
rows = []
for t_low in [25, 30, 40, 50]:
    T = np.arange(t_low, 86, 1.0)
    x = 1 / (acetic.K_B * (T + 273.15))
    ea_p = -np.polyfit(x, np.log(acetic.vapor_pressure(T)), 1)[0]
    ea_c = -np.polyfit(x, np.log(acetic.moisture_saturation(T, T, 100)), 1)[0]
    rows.append([f"{t_low}-85 °C", round(ea_p, 3), round(ea_c, 3)])
pd.DataFrame(rows, columns=["fit range", "vapor pressure Ea (eV)", "Cw,sat Ea (eV)"])

# %%
round(float(acetic.moisture_saturation(85, 85, 100)))

# %% [markdown]
# Both activation energies match the paper to the printed digit for fits that
# start at 25-30 °C and 40-50 °C. At 85 °C Eq. 3 gives 2514 µg/g; the paper's
# 3000 µg/g comes from a fit to measured data, and $S_0 = 0.119$ would give it.
#
# ## Rate of acid generation under light
#
# Table 2 gives the rate constant of five samples. An Arrhenius fit gives the
# paper's 0.08 eV and 32 at 85 °C.

# %%
tm = np.array([54, 54, 54, 54, 86])
a = np.array([31.0, 29.5, 16.7, 26.9, 32.6])
slope, icpt = np.polyfit(1 / (acetic.K_B * (tm + 273.15)), np.log(a), 1)
a85 = np.exp(slope / (acetic.K_B * 358.15) + icpt)
print(f"Ea {-slope:.3f} eV, a(85 °C) {a85:.1f}")

# %% [markdown]
# In the dark the paper uses 17 at 85 °C and 0.65 eV (Table 3).

# %%
T = np.array([25, 45, 65, 85])
pd.DataFrame(
    {
        "T (°C)": T,
        "light": acetic.acid_rate_constant(T, 32, 0.08).round(1),
        "dark": acetic.acid_rate_constant(T, 17, 0.65).round(2),
    }
)

# %% [markdown]
# ## Power loss and acid content
#
# Fig. 5: $P/P_0 = 1 - 0.2\,C_a/C_{a,\mathrm{Life}}$, with
# $C_{a,\mathrm{Life}}$ = 10000 µg/g for the most and 600 µg/g for the least
# acid-resistant modules.

# %%
ca_life = np.array([600, 10000])
pd.DataFrame(
    {
        "Ca_life (ug/g)": ca_life,
        "P/P0 at Ca_life": acetic.power_retention(ca_life, ca_life),
        "loss per ug/g (%)": 100 * (1 - acetic.power_retention(1, ca_life)),
    }
)

# %% [markdown]
# ## Time to reach the threshold
#
# From Fig. 10, the time for $C_a$ to reach 600 µg/g follows 0.78 eV and is
# 13000 h at 75 °C. The paper gives over 620000 h below 30 °C.

# %%
t30 = 13000 * np.exp(0.78 / acetic.K_B * (1 / 303.15 - 1 / 348.15))
round(t30)

# %% [markdown]
# ## Field power
#
# Table 4: $P/P_0 = 1 - (85\,D\,t/C_{a,\mathrm{Life}}^{1/2})^2$, where $D$ is
# the yearly light dose. For the Cfa climate, Fig. 16(b) reaches 0 at 13 years
# with $C_{a,\mathrm{Life}}$ = 600 µg/g, which fixes $D$.

# %%
dose = 600**0.5 / 85 / 13.0
years = np.arange(0, 14, 2)
pd.DataFrame({"years": years, "P/P0": acetic.power_field(years, dose, 600).round(3)})

# %% [markdown]
# Table 3b of the paper gives seasonal inputs for four climates. The dose from
# these inputs differs from the one implied by Fig. 16 by one constant factor,
# which the paper does not print. Fitting it on Cfa predicts the other three
# within the reading error of Fig. 16, from which the target doses are read.

# %%
# (air T, module T, RH, sun hours per sunny day, suns, days); sunny and cloudy days alternate
TABLE3B = {
    "A": [(30, 54, 75, 6, 0.8, 366)],
    "Cfa": [(30, 54, 75, 6, 0.8, 214), (10, 28, 70, 5, 0.6, 152)],
    "Cfb": [(20, 44, 70, 6, 0.8, 120), (10, 28, 80, 6, 0.6, 246)],
    "Csa": [(30, 54, 50, 6, 0.8, 154), (20, 38, 50, 6, 0.6, 212)],
}
FIG16 = {
    "A": 600**0.5 / 85 / 9.28,
    "Cfa": dose,
    "Cfb": (1 - 0.81) ** 0.5 * 10000**0.5 / 85 / 30,
    "Csa": (1 - 0.91) ** 0.5 * 10000**0.5 / 85 / 30,
}
table = {
    c: sum(d / 2 * h * acetic.light_dose(s, rh, tmod) for _, tmod, rh, h, s, d in r)
    for c, r in TABLE3B.items()
}
factor = FIG16["Cfa"] / table["Cfa"]
res = pd.DataFrame(
    {
        "Fig. 16 dose": [FIG16[c] for c in table],
        "Table 3b dose x factor": [table[c] * factor for c in table],
    },
    index=list(table),
)
res["difference (%)"] = (100 * (res.iloc[:, 1] / res.iloc[:, 0] - 1)).round(1)
res.round(4)

# %% [markdown]
# ## With PVDeg weather
#
# Miami, from the weather file that ships with PVDeg. The dose is in the units of
# Table 3b, before the factor fitted above.

# %%
weather = pd.read_csv(
    os.path.join(DATA_DIR, "psm4_miami.csv"), index_col=0, parse_dates=True
)
with open(os.path.join(DATA_DIR, "meta_miami.json")) as f:
    meta = json.load(f)

poa = pvdeg.spectral.poa_irradiance(
    weather, meta, surface_tilt=meta["latitude"], surface_azimuth=180
)
t_mod = pvdeg.temperature.module(weather, meta, poa=poa, conf="open_rack_glass_polymer")
cw = acetic.moisture_saturation(
    t_mod, weather["temp_air"], weather["relative_humidity"]
)
d_miami = acetic.light_dose(
    poa["poa_global"] / 1000, weather["relative_humidity"], t_mod
)
print(f"mean Cw,sat {np.mean(cw):.0f} ug/g, light dose {d_miami:.4f} sun h/yr")
