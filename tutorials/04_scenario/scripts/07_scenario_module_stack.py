# %% [markdown]
# # Multi-Layer Module Stack — Scenario Pipeline Demo
#
# Demonstrates the **`Scenario`** class feature set using a hypothetical perovskite module stack:
#
# > Glass / EVA front encapsulant / CsPbI₃ perovskite absorber / EVA back encapsulant / PET backsheet
#
# **Features demonstrated:**
# 1. `addModule()` — module configuration (temperature model, racking) and named material layers
# 2. Sequential job chaining with `depends_on` — upstream results forwarded as kwargs at run time
# 3. Module-based results — accessed via `s.results[module_name][job_name]`
# 4. Multi-location execution — same pipeline run for three climate zones
#
# **Pipeline (moisture transport chain):**
#
# | Step | Function | Output | Depends on |
# |------|----------|--------|------------|
# | 1 | `spectral.poa_irradiance` | `poa` | — |
# | 2 | `temperature.module` | `temp_mod` | `poa` |
# | 3 | `rh_surface_job` | `rh_surface` | `temp_mod` |
# | 4 | `front_encap_job` (EVA W001) | `rh_front_encap` | `temp_mod` |
# | 5 | `back_encap_job` (EVA W001 + PET W017) | `rh_back_encap` | `temp_mod`, `rh_surface` |
#
# Steps 3–5 form a sequential moisture ingress chain from ambient air through the PET backsheet and into the EVA layers.
#
# **Post-processing (section 6):** `temp_mod` from the pipeline is used directly to compute acetic acid (HAc) generation and accumulation in the EVA encapsulant — a known corrosion precursor in aged PV modules.
#

# %% [markdown]
# ## 1. Imports and data

# %%
import os
import json
import tempfile
import numpy as np
import pandas as pd
import pvdeg
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(pvdeg.__file__))
TUTORIALS_DATA = os.path.join(REPO_ROOT, "tutorials", "data")

# Redirect all Scenario job folders to the system temp directory.
# Keeps the repo working directory free of pvd_job_* clutter.
pvdeg.config.SCENARIO_OUTPUT_PATH = tempfile.gettempdir()


# %%
LOCATIONS = {
    "Golden, CO": ("psm4_golden.csv", "meta_golden.json"),
    "Miami, FL": ("psm4_miami.csv", "meta_miami.json"),
    "New York, NY": ("psm4_nyc.csv", "meta_nyc.json"),
}

all_weather, all_meta = {}, {}
for loc, (wf, mf) in LOCATIONS.items():
    df = pd.read_csv(
        os.path.join(TUTORIALS_DATA, wf),
        index_col=0,
        parse_dates=True,
    )
    df.index = df.index.map(lambda ts: ts.replace(year=2020))
    df = df.sort_index()
    with open(os.path.join(TUTORIALS_DATA, mf)) as _f:
        mt = json.load(_f)
    all_weather[loc], all_meta[loc] = df, mt
    print(f"  {loc}: {len(df)} rows  ({mt['latitude']:.2f}°N, {mt['longitude']:.2f}°E)")
# Single-location aliases used by the demonstration cells below
weather_df = all_weather["Golden, CO"]
meta = all_meta["Golden, CO"]


# %% [markdown]
# ## 2. Pipeline wrapper functions
#
# The Scenario pipeline injects `weather_df` and `meta` into every function automatically.
# Some `pvdeg.humidity` functions require individual weather columns (`rh_ambient`,
# `temp_ambient`) rather than the full DataFrame. Three thin wrappers bridge this gap,
# making those functions compatible with the pipeline while keeping the call site clean.
#


# %%
def rh_surface_job(weather_df, meta, temp_module):
    """Surface RH: scales ambient RH to module temperature.

    temp_module is injected via depends_on from the temp_mod job.
    """
    return pvdeg.humidity.surface_relative(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
    )


def front_encap_job(weather_df, meta, temp_module):
    """Front encapsulant (glass-side) RH.

    Uses EVA W001 (Kempe 2006) defaults from pvdeg H2Opermeation database.
    Returns the diffusivity-weighted average moisture in the front EVA layer.
    """
    return pvdeg.humidity.front_encapsulant(
        rh_ambient=weather_df["relative_humidity"],
        temp_ambient=weather_df["temp_air"],
        temp_module=temp_module,
        # encapsulant="W001" is the default
    )


def back_encap_job(weather_df, meta, temp_module, rh_surface):
    """Back encapsulant (backsheet-side) RH.

    Moisture diffuses through the PET backsheet (W017), then into EVA (W001).
    Uses the quasi-steady-state permeation model (Kempe 2006).
    rh_surface is injected via depends_on from the rh_surface job.
    backsheet_thickness=0.3 mm and back_encap_thickness=0.46 mm are typical
    values for PET and EVA respectively (not stored in the W017/W001 DB entries).
    """
    return pvdeg.humidity.back_encapsulant_water_concentration(
        temp_module=temp_module,
        rh_surface=rh_surface,
        backsheet="W017",  # PET (Kempe, unpublished)
        encapsulant="W001",  # EVA (Kempe 2006)
        backsheet_thickness=0.3,  # mm — typical PET backsheet
        back_encap_thickness=0.46,  # mm — typical EVA encapsulant
        output="rh",
    )


# %% [markdown]
# ## 3. Scenario: addModule and addJob
#
# `addModule` registers the module-level configuration (temperature model, racking)
# and named material layers. Here we load two materials from the `H2Opermeation` database:
#
# - **`encapsulant`**: EVA W001 (Kempe 2006) — front and back encapsulant
# - **`backsheet`**: PET W017 (Kempe, unpublished) — backsheet moisture barrier

# %%
s = pvdeg.Scenario(
    name="perov-module-stack",
    weather_data=weather_df,
    meta_data=meta,
)

# Register the module with temperature model settings and material layers
s.addModule(
    module_name="glass-eva-perov-eva-pet",
    racking="open_rack_glass_polymer",
    temperature_model="sapm",
    materials={
        "encapsulant": {
            "material_file": "H2Opermeation",
            "material_name": "W001",  # EVA (Kempe 2006)
        },
        "backsheet": {
            "material_file": "H2Opermeation",
            "material_name": "W017",  # PET (Kempe, unpublished)
        },
    },
)

# Job 1 — POA irradiance
s.addJob(func=pvdeg.spectral.poa_irradiance, name="poa")

# Job 2 — module temperature
s.addJob(
    func=pvdeg.temperature.module,
    name="temp_mod",
    depends_on={"poa": "poa"},
)

# Job 3 — surface RH (moisture ingress branch)
s.addJob(
    func=rh_surface_job,
    name="rh_surface",
    depends_on={"temp_module": "temp_mod"},
)

# Job 4 — front encapsulant RH (EVA W001, glass side)
s.addJob(
    func=front_encap_job,
    name="rh_front_encap",
    depends_on={"temp_module": "temp_mod"},
)

# Job 5 — back encapsulant RH (PET W017 → EVA W001, backsheet side)
s.addJob(
    func=back_encap_job,
    name="rh_back_encap",
    depends_on={"temp_module": "temp_mod", "rh_surface": "rh_surface"},
)

print("Module layers:", list(s.modules[0]["material_params"].keys()))


# %%
# Show the material parameters loaded from H2Opermeation for each layer
if "s" in globals():
    mat = s.modules[0]["material_params"]
elif "all_scenarios" in globals() and len(all_scenarios) > 0:
    sample_location = next(iter(all_scenarios.keys()))
    mat = all_scenarios[sample_location].modules[0]["material_params"]
else:
    raise RuntimeError("Run the Scenario setup cell first.")

for layer, params in mat.items():
    name = params.get("name", "?")
    alias = params.get("alias", "?")
    print(f"\n{layer.upper()} — {alias} ({name})")
    skip = {"name", "alias", "contributor", "source", "Fickian"}
    for k, v in params.items():
        if k not in skip:
            val = v["value"] if isinstance(v, dict) else v
            units = v.get("units", "") if isinstance(v, dict) else ""
            print(f"  {k:6s} = {val}  [{units}]")


# %% [markdown]
# ## 4. Run the pipeline

# %%
all_scenarios = {}
all_results = {}
module_name = "glass-eva-perov-eva-pet"

for loc in LOCATIONS:
    s_loc = pvdeg.Scenario(
        name=f"perov-module-stack-{loc.replace(', ', '-').replace(' ', '_')}",
        weather_data=all_weather[loc],
        meta_data=all_meta[loc],
    )

    # Register the module with temperature model settings and material layers
    s_loc.addModule(
        module_name=module_name,
        racking="open_rack_glass_polymer",
        temperature_model="sapm",
        materials={
            "encapsulant": {
                "material_file": "H2Opermeation",
                "material_name": "W001",  # EVA (Kempe 2006)
            },
            "backsheet": {
                "material_file": "H2Opermeation",
                "material_name": "W017",  # PET (Kempe, unpublished)
            },
        },
    )

    # Job 1 — POA irradiance
    s_loc.addJob(func=pvdeg.spectral.poa_irradiance, name="poa")

    # Job 2 — module temperature
    s_loc.addJob(
        func=pvdeg.temperature.module,
        name="temp_mod",
        depends_on={"poa": "poa"},
    )

    # Job 3 — surface RH (moisture ingress branch)
    s_loc.addJob(
        func=rh_surface_job,
        name="rh_surface",
        depends_on={"temp_module": "temp_mod"},
    )

    # Job 4 — front encapsulant RH (EVA W001, glass side)
    s_loc.addJob(
        func=front_encap_job,
        name="rh_front_encap",
        depends_on={"temp_module": "temp_mod"},
    )

    # Job 5 — back encapsulant RH (PET W017 -> EVA W001, backsheet side)
    s_loc.addJob(
        func=back_encap_job,
        name="rh_back_encap",
        depends_on={"temp_module": "temp_mod", "rh_surface": "rh_surface"},
    )

    s_loc.run()

    res_loc = s_loc.results[module_name]
    all_scenarios[loc] = s_loc
    all_results[loc] = res_loc

print("Pipeline complete for:", list(all_results.keys()))


# %% [markdown]
# ## 5. Moisture transport results
#
# Pipeline results for all three locations overlaid on a single figure:
#
# | Panel | Quantity | Physical meaning |
# |-------|----------|------------------|
# | 1 | Module temperature [°C] | Thermal driver for all degradation mechanisms |
# | 2 | Surface RH & front encapsulant RH [%] | Moisture at the glass/EVA interface |
# | 3 | Back encapsulant RH [%] | Moisture arriving through the PET backsheet |
#
# The front encapsulant RH is lower than the surface RH because EVA moisture
# solubility is temperature dependent. The back encapsulant RH is lower still
# because the PET backsheet acts as a moisture barrier.
#

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for loc, res_loc in all_results.items():
    res_loc["temp_mod"].plot(ax=axes[0], linewidth=1.0, alpha=0.8, label=loc)
    res_loc["rh_surface"].plot(
        ax=axes[1], linewidth=0.8, alpha=0.45, linestyle="--", label=f"{loc} surface RH"
    )
    res_loc["rh_front_encap"].plot(
        ax=axes[1], linewidth=1.2, alpha=0.9, label=f"{loc} front encap RH"
    )
    res_loc["rh_back_encap"].plot(ax=axes[2], linewidth=1.0, alpha=0.8, label=loc)

axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Module temperature")
axes[1].set_ylabel("RH [%]")
axes[1].set_title("Surface and front encapsulant RH")
axes[2].set_ylabel("RH [%]")
axes[2].set_title("Back encapsulant RH")
axes[2].set_xlabel("Time")

axes[0].legend(ncol=3, fontsize=8)
axes[1].legend(ncol=2, fontsize=7)
axes[2].legend(ncol=3, fontsize=8)

for ax in axes:
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.suptitle(
    "Multi-layer stack — moisture transport by location (2020)",
    y=1.01,
    fontsize=12,
)


# %% [markdown]
# ## 6. Acetic acid generation in EVA
#
# Acetic acid (HAc) is produced by hydrolysis of the vinyl acetate groups in EVA, a
# reaction that requires both heat and moisture. Its accumulation in the encapsulant
# is a known precursor to corrosion of cell metallization and glass/EVA interface
# degradation.
#
# Using `temp_mod` from the Scenario pipeline, two post-processing functions estimate
# the HAc chemistry directly in the EVA layer:
#
# - `acetic_acid_generation`: instantaneous HAc generation rate [ng/min/g] — nanograms of acetic acid produced per minute per gram of EVA
# - `acetic_acid_cumulative`: cumulative HAc concentration [mg/g] integrated over the full year
#
# Both functions apply **Arrhenius kinetics driven by temperature only**. The
# baseline `Ro` and activation energy `Ea` are derived from Kempe (2007), who
# measured HAc generation at multiple temperatures under fixed 85% RH (damp-heat
# conditions). Because all measurements were conducted at a single humidity level,
# the humidity dependence of the hydrolysis rate is not characterised in the
# literature reviewed; `Ro` therefore implicitly assumes the EVA is
# moisture-saturated near 85% RH. An explicit humidity scaling is not applied, as
# no multi-humidity experimental dataset exists to validate its functional form.
#
# > **Note:** The encapsulant RH values computed in Steps 3–5 (`rh_surface`,
# > `rh_front_encap`, `rh_back_encap`) characterise the moisture environment through
# > the stack and are visualised in Section 5, but are not consumed by this section.
# > A humidity-coupled HAc model — once validated against multi-humidity experimental
# > data — could use those pipeline outputs directly as an additional input here via
# > `depends_on`.
#
# Parameters are loaded from the `AApermeation` database (entry AA002), based on
# Kempe (2007) and validated against Gnocchi et al. (2018), who measured ~0.5–0.6 mg/g
# after 3000 h of damp-heat exposure at 85 °C / 85 % RH.
#

# %%
# Summary metrics for all locations
print(
    f"{'Location':<18}  {'Rate start':>16}  {'Cumul. 1000 h':>16}  {'Cumul. 3000 h':>16}"
)
print(f"{'':18}  {'[ng/min/g]':>16}  {'[mg/g]':>16}  {'[mg/g]':>16}")
print("-" * 72)
for loc, res_loc in all_results.items():
    temp_loc = res_loc["temp_mod"]
    hac_rate_loc = pvdeg.degradation.acetic_acid_generation(
        temp_module=temp_loc, encapsulant="AA002"
    )
    hac_cum_loc = pvdeg.degradation.acetic_acid_cumulative(
        temp_module=temp_loc, encapsulant="AA002"
    )
    print(
        f"{loc:<18}  {float(hac_rate_loc.iloc[0]):>16.6f}"
        f"  {float(hac_cum_loc.iloc[999]):>16.6f}"
        f"  {float(hac_cum_loc.iloc[2999]):>16.6f}"
    )


# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for loc, res_loc in all_results.items():
    temp_loc = res_loc["temp_mod"]

    hac_rate_loc = pvdeg.degradation.acetic_acid_generation(
        temp_module=temp_loc, encapsulant="AA002"
    )
    hac_cum_loc = pvdeg.degradation.acetic_acid_cumulative(
        temp_module=temp_loc, encapsulant="AA002"
    )

    hac_rate_loc.plot(ax=axes[0], linewidth=0.9, alpha=0.8, label=loc)
    hac_cum_loc.plot(ax=axes[1], linewidth=1.2, alpha=0.85, label=loc)

axes[0].set_ylabel(
    "Acetic acid generation rate (ng/min/g)"
)  # nanograms of acetic acid produced per minute per gram of EVA
axes[0].set_title("Acetic acid generation rate in EVA (AA002) by location")
axes[1].set_ylabel("Cumulative acetic acid total (mg/g)")
axes[1].set_title("Cumulative acetic acid generation in EVA (AA002) by location")
axes[1].set_xlabel("Time")

axes[0].legend(ncol=3, fontsize=8)
axes[1].legend(ncol=3, fontsize=8)

for ax in axes:
    ax.grid(alpha=0.25)

fig.tight_layout()

# %% [markdown]
# ---
# ## Summary and next steps
#
# This notebook demonstrates the **`Scenario` pipeline architecture** for encapsulant
# degradation chemistry in a multi-layer perovskite module stack. The key features
# demonstrated are `addModule()` (racking, temperature modeling, and named material
# layers), sequential job chaining via `depends_on`, permeation database lookup, and
# multi-location execution across three climate zones.
#
# The pipeline produces a connected physical story:
# - **Steps 1–2** establish the module's thermal environment (`poa` → `temp_mod`)
# - **Steps 3–5** propagate moisture through the stack: ambient air → module surface → front EVA → PET backsheet → back EVA
# - **Section 6** uses `temp_mod` directly to estimate acetic acid build-up in EVA (Arrhenius kinetics, AA002 database entry)
#
# The moisture and HAc outputs represent **encapsulant stressors** at different layers.
# Combining them into a module-level degradation metric (e.g. optical coupling loss
# from delamination, or corrosion rate from HAc at metallization) requires additional
# models not yet implemented in pvdeg. When available, they can be chained directly
# via `depends_on`.
#
# ### See also
#
# For perovskite absorber CE degradation, lifetime projections, and US-wide choropleth maps,
# see **`06_scenario_perovskite_ey.ipynb`**.
#
