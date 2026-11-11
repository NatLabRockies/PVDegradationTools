import numpy as np
import pytest
from pvdeg import acetic


def test_vapor_pressure_boiling():
    assert acetic.vapor_pressure(100) == pytest.approx(1.0, rel=0.02)


def test_moisture_saturation_85c():
    # Eq. 3 of Irikawa et al. 2025 at 85 C, 100 % RH
    assert acetic.moisture_saturation(85, 85, 100) == pytest.approx(2514.4, abs=0.1)


def test_acid_rate_constant_table2():
    # Table 2 of Irikawa et al. 2025: Arrhenius fit gives 0.08 eV and 32 at 85 C
    tm = np.array([54, 54, 54, 54, 86]) + 273.15
    a = np.array([31.0, 29.5, 16.7, 26.9, 32.6])
    slope, icpt = np.polyfit(1 / (acetic.K_B * tm), np.log(a), 1)
    assert -slope == pytest.approx(0.08, abs=0.0005)
    assert acetic.acid_rate_constant(85, 32, 0.08) == pytest.approx(32)
    assert acetic.acid_rate_constant(54, 32, 0.08) == pytest.approx(25.03, abs=0.01)


def test_power_retention_thresholds():
    assert acetic.power_retention(10000, 10000) == pytest.approx(0.8)
    assert acetic.power_retention(600, 600) == pytest.approx(0.8)
    assert 100 * (acetic.power_retention(1, 10000) - 1) == pytest.approx(-0.002)


def test_light_dose_single_hour():
    assert acetic.light_dose(0.8, 75, 54) == pytest.approx(4.3213e-4, rel=1e-4)


def test_power_field_forms():
    dose = 600**0.5 / 85 / 13.0
    assert acetic.power_field(13.0, dose, 600) == pytest.approx(0.0, abs=1e-12)
    dose = 0.07 / 10 * 600 / 191
    assert acetic.power_field(10, dose, 600, form="linear") == pytest.approx(0.93)
    with pytest.raises(ValueError):
        acetic.power_field(1, dose, 600, form="cubic")
