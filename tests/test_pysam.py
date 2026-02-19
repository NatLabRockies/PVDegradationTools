import pvdeg

import builtins
import os
import pvdeg.utilities
import xarray as xr
import pandas as pd
from pathlib import Path
import pytest
import logging


TRACKING = {"01", "02", "03", "04", "05"}
FIXED = {"06", "07", "08", "09"}
FIXED_VERTICAL = {"10"}

GEO_WEATHER = xr.load_dataset(os.path.join(pvdeg.TEST_DATA_DIR, "summit-weather.nc"))
GEO_META = pd.read_csv(
    os.path.join(pvdeg.TEST_DATA_DIR, "summit-meta.csv"), index_col=0
)

# fill in dummy wind direction and albedo values
GEO_WEATHER = GEO_WEATHER.assign(wind_direction=GEO_WEATHER["temp_air"] * 0 + 0)
GEO_WEATHER = GEO_WEATHER.assign(albedo=GEO_WEATHER["temp_air"] * 0 + 0.2)

# single location weather and metadata
WEATHER_SINGLE_LOC = GEO_WEATHER.isel(gid=0).to_dataframe()
META_SINGLE_LOC = GEO_META.iloc[0].to_dict()


def test_pysam_missing_nrel_pysam_deps(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PySAM" or name.startswith("PySAM."):
            raise ModuleNotFoundError("No module named 'PySAM'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    res = pvdeg.pysam.pysam(
        weather_df=pd.DataFrame(), meta={}, pv_model="pvwatts8", config_files={}
    )

    caplog.set_level(logging.INFO)
    assert "pysam not found" in caplog.text.lower()
    assert res is None


def test_pysam_results_list_conf0():
    """
    Test with provided custom config rather than defautl pvsamv1 configs
    """
    res = pvdeg.pysam.pysam(
        weather_df=GEO_WEATHER.isel(gid=0).to_dataframe()[::2],
        meta=META_SINGLE_LOC,
        config_files={
            "pv": os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))
        },
        pv_model="pvsamv1",
        results=["annual_energy"],
        practical_pitch_tilt_considerations=True,
    )

    expected = 53184.367920227654

    assert len(res) == 1
    assert "annual_energy" in res
    assert res["annual_energy"] == pytest.approx(expected, rel=1e-6)


def test_pysam_results_list_default_model():
    """Use default model instead of custom configuration."""
    res = pvdeg.pysam.pysam(
        weather_df=WEATHER_SINGLE_LOC[::2],
        meta=META_SINGLE_LOC,
        pv_model="pvsamv1",
        pv_model_default="FlatPlatePVCommercial",
        results=["annual_energy"],
    )

    expected = 359351.2976863122

    assert len(res) == 1
    assert "annual_energy" in res
    assert res["annual_energy"] == pytest.approx(expected, rel=1e-6)


# split into multiple tests?
def test_pysam_inspire_practical_tilt():
    mid_lat = GEO_META.iloc[0].to_dict()

    res = pvdeg.pysam.pysam(
        weather_df=WEATHER_SINGLE_LOC[::2],
        meta=mid_lat,
        config_files={
            "pv": os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))
        },
        pv_model="pvsamv1",
        practical_pitch_tilt_considerations=True,
    )

    # use latitude tilt under 40 deg N latitude
    assert mid_lat["latitude"] == max(res["subarray1_surf_tilt"])

    high_lat = GEO_META.iloc[0].to_dict()
    high_lat["latitude"] = 45  # cant use latitude tilt above 40 deg N

    res = pvdeg.pysam.pysam(
        weather_df=WEATHER_SINGLE_LOC[::2],
        meta=high_lat,
        config_files={
            "pv": os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))
        },
        pv_model="pvsamv1",
        practical_pitch_tilt_considerations=True,
    )

    # latitude point is above 40 deg N
    # so we floor to 40 deg tilt due to practical racking considerations
    assert 40 == max(res["subarray1_surf_tilt"])

    res = pvdeg.pysam.pysam(
        weather_df=WEATHER_SINGLE_LOC[::2],
        meta=high_lat,
        config_files={
            "pv": os.path.join(pvdeg.TEST_DATA_DIR, Path("SAM/08/08_pvsamv1.json"))
        },
        pv_model="pvsamv1",
        practical_pitch_tilt_considerations=False,
    )

    # flag is set to false, ignore practical considerations
    assert 45 == max(res["subarray1_surf_tilt"])


def test_inspire_configs_pitches():
    """
    Test pitches on three different configurations

    01 - SAT
    06 - Fixed tilt, variable pitch
    10 - Vertical
    """
    configs = ["01", "08", "10"]
    config_paths = [
        os.path.join(pvdeg.TEST_DATA_DIR, Path(f"SAM/{conf}/{conf}_pvsamv1.json"))
        for conf in configs
    ]

    mid_lat = GEO_META.iloc[0].to_dict()
    high_lat = GEO_META.iloc[0].to_dict()
    high_lat["latitude"] = 45

    for conf in config_paths:
        config_files = {"pv": conf}

        res_mid_lat = pvdeg.pysam.inspire_ground_irradiance(
            weather_df=GEO_WEATHER.isel(gid=0).to_dataframe()[::2],
            meta=mid_lat,
            config_files=config_files,
        )

        res_high_lat = pvdeg.pysam.inspire_ground_irradiance(
            weather_df=GEO_WEATHER.isel(gid=0).to_dataframe()[::2],
            meta=high_lat,
            config_files=config_files,
        )

        cw = 2  # collector width [m]

        if conf in TRACKING:
            # tracking, we leave the pitch unchanged from the original sam config
            assert res_mid_lat.pitch.item() == (
                pvdeg.utilities._load_gcr_from_config(config_files=config_files) / cw
            )  # use original pitch
            assert res_high_lat.pitch.item() == (
                pvdeg.utilities._load_gcr_from_config(config_files=config_files) / cw
            )  # use original pitch

            # tracking does not have fixed tilt
            assert res_mid_lat.tilt.item() == -999
            assert res_mid_lat.tilt.item() == -999

        elif conf in FIXED:
            #
            tilt, pitch, gcr = pvdeg.utilities.inspire_practical_pitch(
                latitude=GEO_META["latitude"], cw=cw
            )

            assert res_mid_lat.pitch.item() == pitch
            assert res_high_lat.pitch.item() == 12  # max of 12 meters

            assert res_mid_lat.tilt == res_mid_lat["latitude"]
            assert res_high_lat.tilt == 40  # no latitude tilt above 40

        elif conf in FIXED_VERTICAL:
            assert res_mid_lat.pitch.item() == (
                pvdeg.utilities._load_gcr_from_config(config_files=config_files) / cw
            )  # use original pitch
            assert res_high_lat.pitch.item() == (
                pvdeg.utilities._load_gcr_from_config(config_files=config_files) / cw
            )  # use original pitch

            assert res_mid_lat.tilt == 90  # fixed vertical tilt
            assert res_high_lat.tilt == 90


def test_inspire_ground_irradiance_bad_weather_meta_input():
    with pytest.raises(
        ValueError, match=r"weather_df must be pandas DataFrame, meta must be dict\."
    ):
        pvdeg.pysam.inspire_ground_irradiance(weather_df=[], meta=0, config_files={})


def test_inspire_ground_irradiance_bad_configuration_input():
    with pytest.raises(
        ValueError,
        match=r"Valid config not found, config name must contain setup name from 01-10",
    ):
        pvdeg.pysam.inspire_ground_irradiance(
            weather_df=WEATHER_SINGLE_LOC,
            meta=META_SINGLE_LOC,
            config_files={"pv": "/bad/path.json"},
        )
