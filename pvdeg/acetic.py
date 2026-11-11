"""Acetic acid generation in EVA and the power loss it causes.

Model of Irikawa et al. [1]_.

References
----------
.. [1] J. Irikawa, H. Hashimoto, H. Kanno, M. Taguchi, "Correlation between
   damp-heat test and field operation for electrode corrosion in photovoltaic
   modules", Solar Energy Materials and Solar Cells 284, 113375 (2025).
   https://doi.org/10.1016/j.solmat.2024.113375
"""

import numpy as np

K_B = 8.62e-5  # Boltzmann constant [eV/K], value used in [1]


def vapor_pressure(temp):
    """Saturation water vapor pressure from the Tetens equation.

    Parameters
    ----------
    temp : numeric
        Temperature [°C].

    Returns
    -------
    numeric
        Saturation vapor pressure [atm].
    """
    t = np.asarray(temp, dtype=float)
    return 0.61078 * np.exp(17.27 * t / (t + 237.3)) / 101.325


def moisture_saturation(temp_module, temp_air, rh, s0=0.1, es=0.33):
    """Saturation moisture content of EVA, Eq. 3 of [1].

    .. math:: C_{w,sat} = P_{H_2O} S_0 \\exp(E_s / k_B T)

    Parameters
    ----------
    temp_module : numeric
        Module temperature [°C].
    temp_air : numeric
        Air temperature [°C], sets the water vapor pressure.
    rh : numeric
        Relative humidity of the air [%].
    s0 : float, optional
        Solubility prefactor [µg/g/atm].
    es : float, optional
        Solubility activation energy [eV].

    Returns
    -------
    numeric
        Saturation moisture content [µg/g].
    """
    p_h2o = np.asarray(rh) / 100 * vapor_pressure(temp_air)
    return p_h2o * s0 * np.exp(es / (K_B * (np.asarray(temp_module) + 273.15)))


def acid_rate_constant(temp_module, a_ref, ea, temp_ref=85.0):
    """Arrhenius rate constant of acetic acid generation.

    [1] gives ``a_ref`` = 32, ``ea`` = 0.08 eV under 1 sun and ``a_ref`` = 17,
    ``ea`` = 0.65 eV in the dark, both at 85 °C.

    Parameters
    ----------
    temp_module : numeric
        Module temperature [°C].
    a_ref : float
        Rate constant at ``temp_ref``.
    ea : float
        Activation energy [eV].
    temp_ref : float, optional
        Reference temperature [°C].

    Returns
    -------
    numeric
        Rate constant, same units as ``a_ref``.
    """
    t = np.asarray(temp_module) + 273.15
    return a_ref * np.exp(-ea / K_B * (1 / t - 1 / (temp_ref + 273.15)))


def power_retention(ca, ca_life):
    """Relative maximum power from the acetic acid content, Fig. 5 of [1].

    .. math:: P/P_0 = 1 - 0.2\\,C_a / C_{a,Life}

    Parameters
    ----------
    ca : numeric
        Acetic acid content of the front EVA [µg/g].
    ca_life : float
        Acetic acid content at 80 % power [µg/g]. [1] gives 600 for the
        least and 10000 for the most acid-resistant modules.

    Returns
    -------
    numeric
        Pmax / Pmax0.
    """
    return 1 - 0.2 * np.asarray(ca) / ca_life


def light_dose(irradiance, rh, temp_module, n=1.5, ea=0.2):
    """Light dose of the field model, Table 4 of [1].

    .. math:: \\sum I_L \\, (RH/100)^n \\exp(-E_a / k_B T_m)

    Parameters
    ----------
    irradiance : array-like
        Plane-of-array irradiance [suns], one value per hour.
    rh : array-like
        Relative humidity [%].
    temp_module : array-like
        Module temperature [°C].
    n : float, optional
        Humidity exponent.
    ea : float, optional
        Activation energy [eV].

    Returns
    -------
    float
        Dose summed over the time steps [sun h].
    """
    t = np.asarray(temp_module) + 273.15
    i_l = np.asarray(irradiance)
    return np.sum(i_l * (np.asarray(rh) / 100) ** n * np.exp(-ea / (K_B * t)))


def power_field(years, dose, ca_life, form="t2"):
    """Relative maximum power in the field, Table 4 of [1].

    ``t2``: :math:`P/P_0 = 1 - (85\\,D\\,t / C_{a,Life}^{1/2})^2`.
    ``linear``: :math:`P/P_0 = 1 - 191\\,D\\,t / C_{a,Life}`.

    Parameters
    ----------
    years : numeric
        Time in the field [years].
    dose : float
        Yearly light dose from :py:func:`light_dose` [sun h].
    ca_life : float
        Acetic acid content at 80 % power [µg/g].
    form : str, optional
        ``"t2"`` or ``"linear"``.

    Returns
    -------
    numeric
        Pmax / Pmax0.
    """
    t = np.asarray(years, dtype=float)
    if form == "t2":
        return 1 - (85 * dose * t / ca_life**0.5) ** 2
    if form == "linear":
        return 1 - 191 * dose * t / ca_life
    raise ValueError("form must be 't2' or 'linear'")
