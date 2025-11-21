# %%
import pvdeg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
weather, meta = pvdeg.weather.get(
    database="PVGIS", id=(40.633365593159226, -73.9945801019899)
)

# %%
out_dict = pvdeg.pysam.pysam(
    weather_df=weather,
    meta=meta,
    pv_model="pysamv1",
    pv_model_default="FlatPlatePVCommercial",
)
for key in sorted(out_dict.keys()):
    print(key)

# %%
for key, item in out_dict.items():
    if isinstance(item, tuple):
        print(key)

# %% [markdown]
# subarray1_poa_ground_front_cs
# subarray1_ground_rear_spatial

# %%
x = out_dict["subarray1_ground_rear_spatial"][
    0
]  # these are the distances where the calculations are done

data = out_dict["subarray1_ground_rear_spatial"][1:]
row = data[15]

# %%
row_arr = np.array(row)
plot = sns.heatmap([row_arr], square=True, cmap="magma")

plt.show()
