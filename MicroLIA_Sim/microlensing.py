#!/usr/bin/env python3
# source envs/lightcurvelynx/bin/activate

from __future__ import annotations

import os 
# Thread limits (need to set before numpy/scipy imports) which we used to parallelize the code before, not needed at the moment
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("MKL_NUM_THREADS", "1")
#os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from collections import OrderedDict
from dataclasses import dataclass
import operator as op
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

import pyLIMA.priors.parameters_boundaries
from pyLIMA import event
from pyLIMA.simulations import simulator
from pyLIMA.models import PSPL_model, USBL_model, FSPLarge_model
from pyLIMA.magnification.impact_parameter import impact_parameter
from pyLIMA.toolbox import brightness_transformation as bt

from importlib import resources # For loading the t_m table Miguel provided

# In the brightness_transformation module from pyLIMA these are defaulted to the Roman telescope!
# Overwrite to ensure proper conversion always
ZERO_POINT = 31.4
EXPOSURE_TIME = 1.0
bt.ZERO_POINT = ZERO_POINT
bt.EXPOSURE_TIME = EXPOSURE_TIME

MJD_OFFSET = 2400000.5


def mag2nJy(mag: float | np.ndarray) -> float | np.ndarray:
    """
    Convert AB magnitude(s) to flux density in nanoJanskys (nJy).

    This uses the module-level ``ZERO_POINT`` (set to 31.4 by default as the AB definition is zp=8.9 for 1 Jy) and the
    standard AB magnitude relation.

    Parameters
    ----------
    mag : float or numpy.ndarray
        AB magnitude value(s).

    Returns
    -------
    float or numpy.ndarray
        Flux density value(s) in nJy.
    """
    mag = np.asarray(mag)

    return 10 ** ((ZERO_POINT - mag) / 2.5)


def flux2mag(flux_njy: float | np.ndarray) -> float | np.ndarray:
    """
    Convert flux density in nanoJanskys (nJy) to AB magnitude(s).

    This uses the module-level ``ZERO_POINT`` (set to 31.4 by default as the AB definition is zp=8.9 for 1 Jy) and the
    inverse AB relation. Note that to avoid log of zero or negative flux issues, flux values are clipped to a
    minimum of ``1e-10`` nJy prior to conversion.

    Parameters
    ----------
    flux_njy : float or numpy.ndarray
        Flux density value(s) in nJy.

    Returns
    -------
    float or numpy.ndarray
        AB magnitude value(s).
    """
    flux = np.maximum(np.asarray(flux_njy), 1e-10)

    return ZERO_POINT - 2.5 * np.log10(flux)


def abmag_to_njy(mag: float | np.ndarray) -> float | np.ndarray:
    """
    Convert AB magnitude(s) to flux density in nanoJanskys (nJy) using pyLIMA.

    This calls ``pyLIMA.toolbox.brightness_transformation.magnitude_to_flux``.
    Note that in this pipeline, the pyLIMA brightness transformation module is configured
    at import time by overwriting its module-level ``ZERO_POINT`` and ``EXPOSURE_TIME`` so 
    the conversion is always in the same calibrated physical units. 

    Parameters
    ----------
    mag : float or numpy.ndarray
        AB magnitude value(s).

    Returns
    -------
    float or numpy.ndarray
        Flux density value(s) in nJy.
    """
    return bt.magnitude_to_flux(mag)


def njy_to_abmag(flux_njy: float | np.ndarray) -> float | np.ndarray:
    """
    Convert flux density in nanoJanskys (nJy) to AB magnitude(s) using pyLIMA.

    This calls ``pyLIMA.toolbox.brightness_transformation.flux_to_magnitude``.
    In this pipeline, the pyLIMA brightness transformation module is configured
    at import time by overwriting its module-level ``ZERO_POINT`` and
    ``EXPOSURE_TIME`` so the conversion is always in the same calibrated physical units. 

    Parameters
    ----------
    flux_njy : float or numpy.ndarray
        Flux density value(s) in nJy.

    Returns
    -------
    float or numpy.ndarray
        AB magnitude value(s).
    """
    return bt.flux_to_magnitude(flux_njy)


def t_m_boundaries(event=None, model=None):
    """
    Return parameter bounds for the extended-lens timescale parameter ``t_m``.

    This function is assigned to ``pyLIMA.priors.parameters_boundaries.t_m_boundaries``
    for pyLIMA compatability, and is used only when sampling/validating the extended dark object models.

    Parameters
    ----------
    event : object, optional
        Placeholder argument for pyLIMA API compatibility. Not used.
    model : object, optional
        Placeholder argument for pyLIMA API compatibility. Not used.

    Returns
    -------
    list of float
        Lower and upper bounds for ``t_m`` in days: ``[0.1, 10.0]``.
    """
    return [0.1, 10.0]


#
# Setting the priors for pyLIMA compatability
pyLIMA.priors.parameters_boundaries.t_m_boundaries = t_m_boundaries
#

# Model requirements, helper function to help designate these as explicitly as possible
@dataclass(frozen=True)
class ModelSpec:
    """
    Container specifying parameter requirements for a microlensing model.

    This dataclass describes which entries are required/optional in the user-facing
    ``model_params`` dictionary for each supported model type. It also supports
    alias mapping (to accommodate alternate naming conventions, just in case) and defaults for
    optional parameters.

    Attributes
    ----------
    required : tuple of str
        Names of required keys that must be present in ``model_params`` for this model.
    optional : tuple of str
        Names of optional keys allowed in ``model_params`` for this model.
    defaults : dict of str to Any
        Default values injected into ``model_params`` when missing. Only applies to
        keys listed in ``optional`` (or otherwise expected by the model wrapper).
    aliases : dict of str to str
        Mapping from alias key -> canonical key. If an alias is provided and the
        canonical key is missing, the alias value is copied to the canonical key.
        Intended to tolerate common synonyms and/or upstream naming changes.
    notes : str
        Human-readable description of the model and any important usage notes.
    example : dict of str to Any
        Example ``model_params`` dictionary illustrating typical usage.
    """
    required: tuple[str, ...]
    optional: tuple[str, ...]
    defaults: dict[str, Any]
    aliases: dict[str, str]
    notes: str
    example: dict[str, Any]


# Spelling these out now in very explicit terms so users always know what they need!
MODEL_SPECS: dict[str, ModelSpec] = {
    "PSPL": ModelSpec(
        required=(),
        optional=(),
        defaults={},
        aliases={},
        notes="Standard model, Point-source point-lens. Requires only (t0,u0,tE) + photometry (+ optional parallax).",
        example={},
    ),
    "FSPL": ModelSpec(
        required=("rho",),
        optional=(),
        defaults={},
        aliases={},
        notes="Finite-source point-lens. Only additional parameter that is required is rho.",
        example={"rho": 1e-3},
    ),
    "USBL": ModelSpec(
        required=("s", "q", "rho"),
        optional=("alpha", "origin"),
        defaults={"alpha": 0.0, "origin": "center_of_mass"},
        aliases={
            "sep": "s",
            "separation": "s",
            "mass_ratio": "q",
        },
        notes=(
            "The Uniform-source binary-lens model, in addition to rho it also requires s, q, alpha (optional) and origin (optional). "
        ),
        example={"s": 1.2, "q": 1e-3, "rho": 1e-3, "alpha": 0.5, "origin": "center_of_mass"},
    ),
    "NFW": ModelSpec(
        required=("t_m",),
        optional=(),
        defaults={},
        aliases={},
        notes="Extended dark object. Requires t_m and the loaded mass-function table.",
        example={"t_m": 1.0},
    ),
    "BS": ModelSpec(
        required=("t_m",),
        optional=(),
        defaults={},
        aliases={},
        notes="Extended dark object. Requires t_m and the loaded mass-function table.",
        example={"t_m": 1.0},
    ),
}


def describe_model_requirements(model_type: str) -> str:
    """
    Return a human-readable description of parameter requirements for a model.

    This helper summarizes the required and optional ``model_params`` keys for the
    requested model type, and includes any defaults, aliases, notes, and an example
    (when available). It is primarily used to generate informative error messages.

    Parameters
    ----------
    model_type : str
        Model identifier (must match a key in ``MODEL_SPECS``).

    Returns
    -------
    str
        Multi-line description of the model's requirements. If ``model_type`` is not
        supported, returns a message listing the allowed model types.
    """
    if model_type not in MODEL_SPECS:
        allowed = ", ".join(sorted(MODEL_SPECS))
        return f"Unknown model_type={model_type!r}. Models currently allowed are: {allowed}"

    spec = MODEL_SPECS[model_type]

    lines = [
        f"Model {model_type} requirements:",
        f"required model_params: {list(spec.required)}",
        f"optional model_params: {list(spec.optional)}",
    ]

    if spec.defaults:
        lines.append(f"defaults: {spec.defaults}")
    if spec.aliases:
        lines.append(f"aliases: {spec.aliases}")
    if spec.notes:
        lines.append(f"notes: {spec.notes}")
    if spec.example is not None:
        lines.append(f"example model_params: {spec.example}")
    return "\n".join(lines)


def _normalize_model_type(model_type: str) -> str:
    """
    Normalize a user-supplied model type string to its identifier.

    Normalization is case-insensitive and trims whitespace. A small set of common
    variants are mapped to supported canonical names (e.g., ``"FSPLarge"`` -> ``"FSPL"``).

    Parameters
    ----------
    model_type : str
        User-provided model type string.

    Returns
    -------
    str
        Canonical model type string used for lookups in ``MODEL_SPECS``.
    """
    mt = str(model_type).strip().upper()
    if mt == "FSPLARGE": # allow common variations just in case
        return "FSPL"
    return mt


def normalize_model_params(model_type: str, model_params: dict | None) -> dict[str, Any]:
    """
    Normalize a ``model_params`` dictionary for a given model type.

    This function applies alias mappings (e.g., ``sep`` -> ``s``) when canonical keys are absent and
    removes alias keys when both alias and canonical are present (to avoid duplicates). It also injects 
    default values for missing optional parameters defined in ``MODEL_SPECS``.

    Parameters
    ----------
    model_type : str
        Model identifier (case-insensitive). Normalized via ``_normalize_model_type``.
    model_params : dict or None
        User-supplied model parameters. If None, an empty dictionary is assumed.

    Returns
    -------
    dict of str to Any
        A new dictionary containing normalized model parameters.
    """
    mt = _normalize_model_type(model_type)
    spec = MODEL_SPECS.get(mt)
    if spec is None:
        raise ValueError(describe_model_requirements(mt))

    mp: dict[str, Any] = {} if model_params is None else dict(model_params)

    # Apply alias mapping
    for alias_key, canonical in spec.aliases.items():
        if (canonical not in mp) and (alias_key in mp):
            mp[canonical] = mp[alias_key]

    # Remove alias keys if both exist or if mapped (avoid confusing duplicates)
    for alias_key in spec.aliases.keys():
        if alias_key in mp and spec.aliases[alias_key] in mp:
            mp.pop(alias_key, None)

    # Apply defaults
    for k, v in spec.defaults.items():
        mp.setdefault(k, v)

    return mp


def validate_model_params(model_type: str, model_params: dict[str, Any] | None) -> dict[str, Any]:
    """
    Validate and normalize model-specific parameters.

    This enforces (1) required keys for the requested model, and (2) rejection of
    unknown keys to catch typos early. It also applies alias mappings and default
    values via :func:`normalize_model_params`.

    Parameters
    ----------
    model_type : str
        Model identifier (case-insensitive). Normalized via ``_normalize_model_type``.
    model_params : dict of str to Any or None
        User-supplied model parameters. If None, an empty dictionary is assumed.

    Returns
    -------
    dict of str to Any
        Normalized ``model_params`` dictionary with aliases applied and defaults
        injected (when defined in ``MODEL_SPECS``).
    """
    mt = _normalize_model_type(model_type)
    if mt not in MODEL_SPECS:
        raise ValueError(describe_model_requirements(mt))

    spec = MODEL_SPECS[mt]
    mp = normalize_model_params(mt, model_params)

    missing = [k for k in spec.required if (k not in mp) or (mp[k] is None)]
    if missing:
        raise ValueError(
            f"{mt}: missing required model_params {missing}\n\n{describe_model_requirements(mt)}"
        )

    allowed = set(spec.required) | set(spec.optional)
    unknown = sorted([k for k in mp.keys() if k not in allowed])
    if unknown:
        raise ValueError(
            f"{mt}: unknown model_params keys {unknown}\n"
            f"Allowed keys: {sorted(allowed)}\n\n"
            f"{describe_model_requirements(mt)}"
        )

    return mp


def validate_parallax(parallax_params: dict | None) -> dict | None:
    """
    Validate and normalize parallax parameters for pyLIMA.

    Parallax is represented by the North/East components of the microlensing
    parallax vector: ``piEN`` and ``piEE``.

    Parameters
    ----------
    parallax_params : dict or None
        Parallax parameter dictionary. If provided, it must contain both keys
        ``"piEN"`` and ``"piEE"``. If None, parallax is treated as disabled.

    Returns
    -------
    dict or None
        If ``parallax_params`` is None, returns None. Otherwise returns a new dict
        ``{"piEN": float(...), "piEE": float(...)}``.
    """
    if parallax_params is None:
        return None

    if ("piEN" not in parallax_params) or ("piEE" not in parallax_params):
        raise ValueError(
            "parallax_params must contain both 'piEN' and 'piEE', e.g. {'piEN': 0.1, 'piEE': -0.05}."
        )

    return {"piEN": float(parallax_params["piEN"]), "piEE": float(parallax_params["piEE"])}


def validate_photometry(
    *,
    bands: tuple[str, ...],
    source_mags: dict[str, float] | None,
    blend_mags: dict[str, float] | None,
    blend_g: dict[str, float] | float | None,
) -> None:
    """
    Validate photometric inputs for simulated multi-band light curves.

    This checks that ``source_mags`` is a non-empty dict and provides an AB magnitude for every requested band in ``bands``.
    It also checks that blending is specified in exactly one way, either ``blend_mags`` (explicit blend magnitudes per band) 
    or ``blend_g`` (blend-to-source flux ratio), but not both. Also, if ``blend_mags`` is provided, it includes all requested bands,
    and if ``blend_g`` is a dict, it includes all requested bands (a single scalar is allowed and applies to all bands).

    Parameters
    ----------
    bands : tuple of str
        Photometric band names that will be simulated (e.g., ``("g", "r", "i")``).
    source_mags : dict of str to float or None
        Source AB magnitudes per band. Must include every band in ``bands``.
    blend_mags : dict of str to float or None
        Blend-only AB magnitudes per band. If provided, must include every band in ``bands``. Mutually exclusive with ``blend_g``.
    blend_g : dict of str to float, float, or None
        Blend-to-source flux ratio ``g = f_blend / f_source``. May be a scalar applied to all bands or a per-band dictionary. Mutually exclusive with ``blend_mags``.

    Returns
    -------
    None
        This function returns None if validation passes.
    """
    if source_mags is None or not isinstance(source_mags, dict) or len(source_mags) == 0:
        raise ValueError("source_mags must be a non-empty dict of AB mags for the requested bands.")

    for b in bands:
        if b not in source_mags:
            raise ValueError(
                f"source_mags missing band {b!r}. "
                f"Provide AB mags for all requested bands={bands}."
            )

    if blend_mags is not None and blend_g is not None:
        raise ValueError(
            "Provide either blend_mags OR blend_g (not both). "
            "blend_mags = physical blending; blend_g = 'Anibal' method."
        )

    if blend_mags is not None:
        for b in bands:
            if b not in blend_mags:
                raise ValueError(
                    f"blend_mags missing band {b!r}. "
                    f"Provide blend mags for all requested bands={bands} (or use blend_g instead)."
                )

    if isinstance(blend_g, dict):
        for b in bands:
            if b not in blend_g:
                raise ValueError(
                    f"blend_g dict missing band {b!r}. "
                    f"Provide blend_g for all requested bands={bands} (or pass a scalar blend_g)."
                )


# NFW / BS (need to load the mass-function before the modeling can be performed, which is stored in the data directory)
m_nfw_inner = None
dm_nfw_inner = None
m_boson_inner = None
dm_boson_inner = None

NFW_LOADED = False
BS_LOADED = False

# Package-default filenames (these live inside the package's data directory!)
_NFW_TABLE = "mt_nfw_list.csv"
_BS_TABLE  = "mt_boson_list.csv"

def load_and_interpolate_mass_function(
    csv_path_or_name: str | None = None,
    drop_row: int | None = None,
    base_dir: str | None = None,
):
    """
    Load a 1D extended-lens mass-function table and build spline interpolants.

    The input CSV is expected to contain two columns with no header -- a column 0: ``t`` (dimensionless coordinate; 
    may be stored as strings that can be evaluated to floats), and a column 1: ``mt`` (dimensionless enclosed-mass function evaluated at ``t``)

    For numerical stability and symmetry, the table is mirrored to negative ``t``
    by reversing the original arrays and negating ``t``. Optionally, one row in the
    mirrored half may be dropped (useful when the original table includes an endpoint
    that would be duplicated by the mirroring). If ``drop_row`` is None, a point
    ``(t=0, mt=0)`` is appended to the mirrored half before concatenation.

    The function returns a cubic B-spline interpolant ``m_inner(t)`` and its derivative
    ``dm_inner(t)`` as callable objects suitable for the extended-lens magnification calculation.

    Parameters
    ----------
    csv_path_or_name : str
        Path to the CSV file, or a filename to be joined with ``base_dir``. This file is 
        saved in the package's data directory and will be loaded when this is None! This should
        really never be set by the user...
    drop_row : int or None, optional
        If provided, drops this row index from the mirrored (negative-``t``) half
        before concatenation. If None, appends ``(0, 0)`` to the mirrored half
        instead. Default is None.
    base_dir : str or None, optional
        If provided and ``csv_path_or_name`` is a relative path, the file path is
        constructed as ``os.path.join(base_dir, csv_path_or_name)``. Default is None.
        This should really never be set by the user...

    Returns
    -------
    tuple
        ``(m_inner, dm_inner)``, where both entries are callable spline objects.
        If loading or interpolation fails, returns ``(None, None)``.
    """
    if csv_path_or_name is None:
        return None, None

    try:
        # Case 1: developer override via explicit filesystem path
        if base_dir is not None:
            csv_path = csv_path_or_name
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(base_dir, csv_path)
            mt_data = pd.read_csv(csv_path, header=None, names=["t", "mt"])

        # Case 2: default: load from package data/ using importlib.resources
        else:
            # If someone passes an absolute path, allow it
            if os.path.isabs(csv_path_or_name):
                mt_data = pd.read_csv(csv_path_or_name, header=None, names=["t", "mt"])
            else:
                pkg = __package__
                res = resources.files(pkg).joinpath("data", csv_path_or_name)
                with res.open("r", encoding="utf-8") as fp:
                    mt_data = pd.read_csv(fp, header=None, names=["t", "mt"])

        # Keep your original parsing behavior
        mt_data["t"] = mt_data["t"].apply(eval)

        mt_data_2 = mt_data.iloc[::-1].copy()
        mt_data_2["t"] = -1.0 * mt_data_2["t"]

        if drop_row is not None:
            mt_data_2 = mt_data_2.drop(drop_row, axis=0)
        else:
            mt_data_2.loc[len(mt_data_2)] = [0.0, 0.0]

        mt_both = pd.concat([mt_data_2, mt_data], axis=0, ignore_index=True)
        m_inner = make_interp_spline(mt_both["t"].values, mt_both["mt"].values)
        return m_inner, m_inner.derivative()

    except Exception:
        return None, None


def load_custom_extended_lens_tables(
    nfw_csv_path: str | None = None,
    boson_csv_path: str | None = None,
    *,
    nfw_drop_row: int = 0,
    base_dir: str | None = None,
) -> None:
    """
    Load and cache mass-function spline tables for extended-lens models (NFW / BS).

    NOTE: The csv_path_or_name and base_dir SHOULD JUST BE NONE ALWAYS! End user should not be updating this table 
    manually (unless it's our team with an updated file!)

    Parameters
    ----------
    nfw_csv_path : str or None
        Path (or filename) to the NFW mass-function CSV. If None, the NFW tables
        are not loaded/modified.
    boson_csv_path : str or None
        Path (or filename) to the boson-star (BS) mass-function CSV. If None, the
        BS tables are not loaded/modified.
    nfw_drop_row : int, optional
        Row index to drop from the mirrored half of the NFW table before
        concatenation. Default is 0 (matching the original pipeline behavior).
    base_dir : str or None, optional
        Base directory prepended to relative paths for both CSV inputs. Default is None.

    Returns
    -------
    None
        This function returns None, it only updates module-level spline objects and flags.
    """
    global m_nfw_inner, dm_nfw_inner, m_boson_inner, dm_boson_inner, NFW_LOADED, BS_LOADED

    # Hardcode defaults to packaged filenames
    if nfw_csv_path is None:
        nfw_csv_path = _NFW_TABLE
    if boson_csv_path is None:
        boson_csv_path = _BS_TABLE

    m_nfw_inner, dm_nfw_inner = load_and_interpolate_mass_function(
        nfw_csv_path, drop_row=nfw_drop_row, base_dir=base_dir
    )
    NFW_LOADED = (m_nfw_inner is not None)

    m_boson_inner, dm_boson_inner = load_and_interpolate_mass_function(
        boson_csv_path, drop_row=None, base_dir=base_dir
    )
    BS_LOADED = (m_boson_inner is not None)


def calculate_magnification(tau, beta, t_m, m, dm):
    """
    Compute photometric magnification for an extended lens via root finding.

    This routine evaluates the magnification for an extended (non-point) lens model
    by solving a 1D lens equation for each impact parameter ``u(t)``. For each time
    sample, the method first computes the instantaneous impact parameter ``u_t`` from 
    the source trajectory coordinates (``tau``, ``beta``). It then finds all real roots 
    of the extended-lens lens equation (image positions) on a fixed 1D scan interval by 
    detecting sign changes and refining each root with ``scipy.optimize.root_scalar``. It then
    computes the signed magnification contribution for each image using the Jacobian factors 
    that depend on the enclosed-mass function ``m(t, t_m)`` and its derivative ``dm(t, t_m)``.
    The total magnification that is returned is then the sum of absolute image magnifications.

    Parameters
    ----------
    tau : array-like
        Dimensionless trajectory coordinate along the direction of motion (as returned
        by pyLIMA's trajectory builder). 
    beta : array-like
        Dimensionless trajectory coordinate perpendicular to the direction of motion.
    t_m : float
        Extended-lens characteristic scale.
    m : callable
        Enclosed-mass function for the model. Must accept ``(t, t_m)`` and return
        ``m(t, t_m)`` evaluated elementwise for array-like ``t``.
    dm : callable
        Derivative of the enclosed-mass function with respect to ``t``. Must accept
        ``(t, t_m)`` and return ``dm(t, t_m)`` evaluated elementwise for array-like ``t``.

    Returns
    -------
    numpy.ndarray
        Array of magnifications with the same length as the computed impact-parameter
        array. If no roots are found for a given time sample, the magnification is
        set to 1.0 (i.e., no lensing).
    """
    def find_roots(u_t, t_m):
        """
        Find image-position roots of the extended-lens 1D lens equation for a given ``u_t``.

        This helper scans a fixed grid in the image-plane coordinate ``t`` to locate
        sign changes in the lens equation residual

        ``f(t) = -u_t + t - m(t, t_m) / t``,

        then refines each bracketed root using ``scipy.optimize.root_scalar``. The point
        ``t = 0`` is excluded to avoid division by zero.

        Parameters
        ----------
        u_t : float
            Scalar impact parameter at a single epoch (dimensionless).
        t_m : float
            Extended-lens characteristic scale (passed through to ``m``).

        Returns
        -------
        numpy.ndarray
            1D array of real roots (image positions) found within the scan interval.
            If no sign changes are detected or all bracketed solves fail, returns an
            empty array.
        """
        _t = np.arange(-1e2, 1e2, 1e-1)
        _t = _t[_t != 0]
        _m = m(_t, t_m)
        _f = -u_t + _t - _m / _t
        mask_sign_change = _f[:-1] * _f[1:] < 0
        idxs = np.nonzero(mask_sign_change)[0]

        roots = []
        for idx in idxs:
            try:
                _sol = root_scalar(lambda t: -u_t + t - m(t, t_m)/t, bracket=[_t[idx], _t[idx+1]])
                roots.append(_sol.root)
            except: pass

        return np.array(roots)

    def mu_function(t, t_m):
        """
        Compute the (absolute) magnification contribution for a single image position.

        Given an image-plane coordinate ``t`` (a root of the lens equation), this
        evaluates the magnification factor based on Jacobian terms involving the
        enclosed-mass function ``m(t, t_m)`` and its derivative ``dm(t, t_m)``.

        Parameters
        ----------
        t : float
            Image position (root of the lens equation).
        t_m : float
            Extended-lens characteristic scale (passed through to ``m`` and ``dm``).

        Returns
        -------
        float
            Magnification contribution for this image. If the denominator evaluates
            to zero, returns 1.0 as a numerical safeguard.
        """
        denom = np.abs(1 - m(t, t_m)/(t**2)) * np.abs(1 + m(t, t_m)/(t**2) - dm(t, t_m)/t)

        return 1 / denom if denom != 0 else 1.0

    u_t = impact_parameter(tau, beta)

    magnification = []
    for ut in u_t:
        roots = find_roots(ut, t_m)
        mus = [mu_function(r, t_m) for r in roots]
        magnification.append(np.sum(np.abs(mus)) if mus else 1.0)

    return np.array(magnification)


def m_nfw(t, t_m):
    """
    Evaluate the dimensionless enclosed-mass function for the NFW extended lens.

    Parameters
    ----------
    t : float or numpy.ndarray
        Image-plane coordinate(s) at which to evaluate the mass function.
    t_m : float
        Extended-lens scale parameter defining the truncation boundary.

    Returns
    -------
    float or numpy.ndarray
        Enclosed-mass function value(s) ``m(t, t_m)``. The return type matches the
        shape of ``t``.
    """
    return np.where(np.abs(t) < t_m, m_nfw_inner(t / t_m), 1.0)


def dm_nfw(t, t_m):
    """
    Evaluate the derivative of the NFW enclosed-mass function with respect to ``t``.

    Parameters
    ----------
    t : float or numpy.ndarray
        Image-plane coordinate(s) at which to evaluate the derivative.
    t_m : float
        Extended-lens scale parameter defining the truncation boundary.

    Returns
    -------
    float or numpy.ndarray
        Derivative value(s) ``dm(t, t_m)``. The return type matches the shape of ``t``.
    """
    return np.where(np.abs(t) < t_m, dm_nfw_inner(t / t_m) / t_m, 0.0)


def m_boson(t, t_m):
    """
    Evaluate the dimensionless enclosed-mass function for the boson-star (BS) extended lens.

    Parameters
    ----------
    t : float or numpy.ndarray
        Image-plane coordinate(s) at which to evaluate the mass function.
    t_m : float
        Extended-lens scale parameter defining the truncation boundary.

    Returns
    -------
    float or numpy.ndarray
        Enclosed-mass function value(s) ``m(t, t_m)``. The return type matches the
        shape of ``t``.
    """
    return np.where(np.abs(t) < t_m, m_boson_inner(t / t_m), 1.0)


def dm_boson(t, t_m):
    """
    Evaluate the derivative of the boson-star (BS) enclosed-mass function with respect to ``t``.

    Parameters
    ----------
    t : float or numpy.ndarray
        Image-plane coordinate(s) at which to evaluate the derivative.
    t_m : float
        Extended-lens scale parameter defining the truncation boundary.

    Returns
    -------
    float or numpy.ndarray
        Derivative value(s) ``dm(t, t_m)``. The return type matches the shape of ``t``.
    """
    return np.where(np.abs(t) < t_m, dm_boson_inner(t / t_m) / t_m, 0.0)


class ExtendedLensModel(PSPL_model.PSPLmodel):
    """
    Base class for extended-lens photometric microlensing models in pyLIMA.

    This class subclasses pyLIMA's ``PSPLmodel`` but replaces the point-lens
    magnification with an extended-lens magnification computed from a model-specific
    enclosed-mass function ``m(t, t_m)`` and its derivative ``dm(t, t_m)``.

    Subclasses must provide a ``model_mass_functions`` property returning the pair
    ``(m, dm)`` callables used by :func:`calculate_magnification`. In this pipeline,
    concrete subclasses include ``NFWmodel`` and ``BSmodel``.
    """

    def paczynski_model_parameters(self):
        """
        Define the ordered set of core photometric parameters for the extended lens.

        This extends the standard Paczyński parameterization by including the
        additional extended-lens scale parameter ``t_m``. The returned mapping is
        used to build pyLIMA's internal parameter dictionary ordering.

        Returns
        -------
        dict of str to int
            Mapping from parameter name to index in the ordered pyLIMA parameter
            vector. The base ordering is:
            ``{"t0": 0, "u0": 1, "tE": 2, "t_m": 3}``.
        """
        return {"t0": 0, "u0": 1, "tE": 2, "t_m": 3}

    def define_pyLIMA_standard_parameters(self):
        """
        Construct pyLIMA's standard parameter dictionary for this extended-lens model.

        This method builds the full parameter set expected by pyLIMA by starting from
        the extended Paczyński parameters (``t0``, ``u0``, ``tE``, ``t_m``) and then
        adding any relevant astrometric, second-order, and per-telescope flux
        parameters using pyLIMA helper methods.

        The resulting dictionary is stored in ``self.pyLIMA_standards_dictionnary``
        as an ``OrderedDict`` sorted by the assigned parameter indices. The parameter
        boundaries are set to ``(0, +inf)`` for all parameters as default.

        Returns
        -------
        None
            This method updates instance attributes used internally by pyLIMA.
        """
        model_dict = self.paczynski_model_parameters()
        model_dict = self.astrometric_model_parameters(model_dict)
        self.second_order_model_parameters(model_dict)
        self.telescopes_fluxes_model_parameters(model_dict)
        self.pyLIMA_standards_dictionnary = OrderedDict(sorted(model_dict.items(), key=lambda x: x[1]))
        self.standard_parameters_boundaries = [(0, np.inf)] * len(self.pyLIMA_standards_dictionnary)

    def model_magnification(self, telescope, pyLIMA_parameters, return_impact_parameter=False):
        """
        Compute the photometric magnification for an extended lens at telescope epochs.

        The source trajectory is computed using pyLIMA's trajectory machinery and
        then passed to :func:`calculate_magnification`, along with the extended-lens
        scale parameter ``t_m`` and the model-specific mass-function callables.

        Parameters
        ----------
        telescope : object
            pyLIMA telescope object containing timestamps and (after simulation) a ``lightcurve`` attribute.
        pyLIMA_parameters : dict-like
            pyLIMA parameter container/dictionary with at least the key ``"t_m"``, along with the usual microlensing parameters needed to compute the trajectory.
        return_impact_parameter : bool, optional
            Included for API compatibility with pyLIMA. This implementation ignores the flag and always returns the magnification array.

        Returns
        -------
        numpy.ndarray or None
            Array of magnifications evaluated at the telescope timestamps. If the
            telescope has no lightcurve attached (``telescope.lightcurve is None``),
            returns None.
        """
        if telescope.lightcurve is None:
            return None

        traj = self.sources_trajectory(telescope, pyLIMA_parameters, data_type="photometry")

        return calculate_magnification(
            traj[0], traj[1], pyLIMA_parameters["t_m"],
            self.model_mass_functions[0], self.model_mass_functions[1],
        )


class NFWmodel(ExtendedLensModel):
    """
    Extended-lens microlensing model using an NFW-like enclosed-mass profile.
    """

    @property
    def model_mass_functions(self):
        """
        Return the enclosed-mass function and its derivative for the NFW model.

        Returns
        -------
        tuple of callable
            ``(m_nfw, dm_nfw)``, callables with signature ``(t, t_m)``.
        """
        return (m_nfw, dm_nfw)

    @property
    def _model_type(self):
        """
        Return the model type label used internally by this pipeline.

        Returns
        -------
        str
            Model type string: ``"NFW"``.
        """
        return "NFW"


class BSmodel(ExtendedLensModel):
    """
    Extended-lens microlensing model using a boson-star enclosed-mass profile.
    """

    @property
    def model_mass_functions(self):
        """
        Return the enclosed-mass function and its derivative for the BS model.

        Returns
        -------
        tuple of callable
            ``(m_boson, dm_boson)``, callables with signature ``(t, t_m)``.
        """
        return (m_boson, dm_boson)

    @property
    def _model_type(self):
        """
        Return the model type label used internally by this pipeline.

        Returns
        -------
        str
            Model type string: ``"BS"``.
        """
        return "BS"


# Time grid generator, following Rache's suggestion of adaptive sampling (only used in case users don't specify cadence)
def build_time_grid_jd(
    *,
    t0_mjd: float,
    tE_days: float,
    window_size_days: float = 4000.0,
    peak_width_factor: float = 5.0,
    step_dense_days: float = 0.1,
    step_sparse_days: float = 10.0,
) -> np.ndarray:
    """
    Construct an adaptive Julian Date (JD) time grid centered on a microlensing event.

    The grid is sampled densely near the event peak and sparsely in the wings, with a dense "core" region (``t0 plus/minus (peak_width_factor * tE)``, 
    sampled every ``step_dense_days``) and sparse wings (outside the core region out to ``t0 plus/minus window_size_days``, sampled every ``step_sparse_days``).

    This is primarily used when the user does not provide an explicit cadence and wants a "perfect" (noise-free) model.

    Parameters
    ----------
    t0_mjd : float
        Event time of maximum magnification in Modified Julian Date (MJD).
    tE_days : float
        Einstein timescale ``tE`` in days.
    window_size_days : float, optional
        Half-width of the full time window around ``t0`` (in days). The grid spans
        ``[t0 - window_size_days, t0 + window_size_days]`` in JD. Default is 4000.0.
    peak_width_factor : float, optional
        Sets the half-width of the densely sampled core as
        ``peak_width_factor * tE_days``. Default is 5.0.
    step_dense_days : float, optional
        Step size (days) in the densely sampled core region. Default is 0.1.
    step_sparse_days : float, optional
        Step size (days) in the sparsely sampled wings. Default is 10.0.

    Returns
    -------
    numpy.ndarray
        1D array of timestamps in Julian Date (JD), sorted in ascending order.
    """
    t0_jd = float(t0_mjd) + MJD_OFFSET
    peak_width = float(peak_width_factor) * float(tE_days)

    left = np.arange(t0_jd - window_size_days, t0_jd - peak_width, step_sparse_days)
    core = np.arange(t0_jd - peak_width, t0_jd + peak_width, step_dense_days)
    right = np.arange(t0_jd + peak_width, t0_jd + window_size_days, step_sparse_days)

    return np.concatenate([left, core, right])


# Blending helper function
def _compute_fsource_fblend_for_band(
    *,
    band: str,
    source_mag: float,
    blend_mags: dict[str, float] | None,
    blend_g: dict[str, float] | float | None,
) -> tuple[float, float]:
    """
    Compute per-band source and blend fluxes (nJy) from magnitude-based inputs.

    This helper converts the photometric inputs for a single band into the
    ``(fsource, fblend)`` values expected by pyLIMA. It supports two mutually
    exclusive blending conventions:

    1. ``blend_mags`` (physical blending):
       - ``source_mag`` is the source-only AB magnitude in this band.
       - ``blend_mags[band]`` is the blend-only AB magnitude in this band.

    2. ``blend_g`` (blend ratio method):
       - ``source_mag`` is the total baseline AB magnitude in this band (source + blend).
       - ``blend_g`` defines the flux ratio ``g = f_blend / f_source`` (either a scalar applied to all bands or a per-band dict).

    If neither blending input is provided, the blend flux is set to 0 and the source flux is computed directly from ``source_mag``.

    Parameters
    ----------
    band : str
        Band label (e.g., ``"g"``, ``"r"``, ``"i"``). Used to index ``blend_mags`` or dict-style ``blend_g`` when provided.
    source_mag : float
        AB magnitude in the specified band. Interpretation depends on blending mode:
        source-only if ``blend_mags`` is provided, otherwise total baseline if
        ``blend_g`` is provided, otherwise source-only with no blend.
    blend_mags : dict of str to float or None
        Blend-only AB magnitudes per band. If provided, must contain ``band``.
    blend_g : dict of str to float, float, or None
        Blend-to-source flux ratio ``g = f_blend / f_source``. May be a scalar or
        per-band dictionary. If a dict is provided, it must contain ``band``.

    Returns
    -------
    tuple of float
        ``(fsource_njy, fblend_njy)`` in nJy.
    """
    if blend_mags is not None:
        bmag = float(blend_mags[band])
        f_s = float(abmag_to_njy(float(source_mag)))
        f_b = float(abmag_to_njy(bmag))
        return f_s, f_b

    if blend_g is not None:
        g_val = float(blend_g[band]) if isinstance(blend_g, dict) else float(blend_g)
        g_val = max(g_val, 1e-12)
        f_total = float(abmag_to_njy(float(source_mag)))  # source_mag is TOTAL baseline mag
        f_s = f_total / (1.0 + g_val)
        f_b = f_s * g_val
        return float(f_s), float(f_b)

    f_s = float(abmag_to_njy(float(source_mag)))
    f_b = 0.0

    return float(f_s), float(f_b)

# The explicit simulator (i.e., all inputs are required!)
def simulate_perfect_event(
    *,
    model_type: str,
    ra: float,
    dec: float,
    t0_mjd: float,
    u0: float,
    tE: float,
    bands: tuple[str, ...] = ("u", "g", "r", "i", "z", "y"),
    time_grid_jd: np.ndarray | None = None,
    model_params: dict | None = None,
    parallax_params: dict | None = None,
    source_mags: dict[str, float] | None = None,
    blend_mags: dict[str, float] | None = None,
    blend_g: dict[str, float] | float | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Simulate a noise-free ("perfect") multi-band microlensing light curve with pyLIMA.

    This function constructs a pyLIMA ``Event`` with one simulated telescope per band,
    evaluates the selected microlensing model on a user-specified (or automatically
    generated) time grid, and returns per-band light curves in both magnitude and
    flux (nJy). No observational noise is added; the intent is for downstream code
    to resample to a survey cadence and inject noise separately.

    Parameters
    ----------
    model_type : str
        Microlensing model identifier. Supported values are the keys of
        ``MODEL_SPECS`` (e.g., ``"PSPL"``, ``"FSPL"``, ``"USBL"``, ``"NFW"``, ``"BS"``).
        The value is normalized via ``_normalize_model_type``.
    ra : float
        Right ascension of the event in degrees.
    dec : float
        Declination of the event in degrees.
    t0_mjd : float
        Time of maximum magnification in Modified Julian Date (MJD).
    u0 : float
        Impact parameter at closest approach (dimensionless).
    tE : float
        Einstein timescale in days.
    bands : tuple of str, optional
        Photometric bands to simulate. One pyLIMA telescope is created per band.
        Default is ``("u", "g", "r", "i", "z", "y")``.
    time_grid_jd : numpy.ndarray or None, optional
        Array of timestamps in Julian Date (JD). If None, an adaptive grid is
        generated with :func:`build_time_grid_jd`. Default is None.
    model_params : dict or None, optional
        Model-specific parameters. Requirements depend on ``model_type`` and are
        enforced by :func:`validate_model_params`. Examples:
        - FSPL: ``{"rho": 1e-3}``
        - USBL: ``{"s": 1.15, "q": 1e-3, "rho": 1e-3, "alpha": 0.4, "origin": "center_of_mass"}``
        - NFW/BS: ``{"t_m": 3.0}``
    parallax_params : dict or None, optional
        Parallax parameters ``{"piEN": ..., "piEE": ...}``. If None, parallax is
        disabled. Validated by :func:`validate_parallax`.
    source_mags : dict of str to float or None, optional
        Source magnitudes per band in AB mag. Must include all requested ``bands``.
        Interpretation depends on blending mode (see ``blend_mags`` / ``blend_g``).
    blend_mags : dict of str to float or None, optional
        Blend-only magnitudes per band in AB mag (physical blending). If provided,
        must include all requested ``bands``. Mutually exclusive with ``blend_g``.
    blend_g : dict of str to float, float, or None, optional
        Blend ratio method with ``g = f_blend / f_source``. May be a scalar applied
        to all bands or a per-band dict. If provided, ``source_mags`` are interpreted
        as total baseline magnitudes (source + blend). Mutually exclusive with
        ``blend_mags``.

    Returns
    -------
    dict of str to pandas.DataFrame
        Mapping ``band -> lightcurve``. Each DataFrame contains columns:

        - ``mjd``: timestamps in MJD
        - ``flux_njy``: flux density in nJy (derived from model magnitudes)
        - ``mag``: AB magnitudes from pyLIMA

        Rows are sorted by ``mjd`` within each band.
    """
    mt = _normalize_model_type(model_type)
    if mt not in MODEL_SPECS:
        raise ValueError(describe_model_requirements(mt))

    if mt in ("NFW", "BS") and (not NFW_LOADED or not BS_LOADED):
        load_custom_extended_lens_tables()
        
    # Validate inputs
    mp = validate_model_params(mt, model_params)
    pp = validate_parallax(parallax_params)
    validate_photometry(bands=bands, source_mags=source_mags, blend_mags=blend_mags, blend_g=blend_g)

    # Build time grid (original defaults) if not provided
    if time_grid_jd is None:
        time_grid_jd = build_time_grid_jd(t0_mjd=t0_mjd, tE_days=tE)

    # Load-check for extended lens models
    if mt == "NFW" and not NFW_LOADED:
        raise RuntimeError("NFW tables not loaded. Call load_custom_extended_lens_tables(nfw_csv_path=..., boson_csv_path=...).")
    if mt == "BS" and not BS_LOADED:
        raise RuntimeError("BS tables not loaded. Call load_custom_extended_lens_tables(nfw_csv_path=..., boson_csv_path=...).")

    enable_parallax = pp is not None

    sim_ev = event.Event(ra=float(ra), dec=float(dec))
    telescope_objs: dict[str, Any] = {}

    for b in bands:
        tel = simulator.simulate_a_telescope(name=b, location="Earth", timestamps=time_grid_jd, astrometry=False)
        sim_ev.telescopes.append(tel)
        telescope_objs[b] = tel

    sim_ev.find_survey(bands[0])

    t0_jd = float(t0_mjd) + MJD_OFFSET
    parallax_arg = ["Full", t0_jd] if enable_parallax else ["None", t0_jd]

    usbl_origin = ["center_of_mass", [0, 0]]
    if mt == "USBL":
        usbl_origin = [str(mp.get("origin", "center_of_mass")), [0, 0]]

    # Instantiate model
    if mt == "PSPL":
        ev_model = PSPL_model.PSPLmodel(sim_ev, parallax=parallax_arg, blend_flux_parameter="fblend")
    elif mt == "FSPL":
        ev_model = FSPLarge_model.FSPLargemodel(sim_ev, parallax=parallax_arg, blend_flux_parameter="fblend")
    elif mt == "USBL":
        ev_model = USBL_model.USBLmodel(sim_ev, parallax=parallax_arg, origin=usbl_origin, blend_flux_parameter="fblend")
    elif mt == "NFW":
        ev_model = NFWmodel(sim_ev, parallax=parallax_arg, blend_flux_parameter="fblend")
    elif mt == "BS":
        ev_model = BSmodel(sim_ev, parallax=parallax_arg, blend_flux_parameter="fblend")
    else:
        raise ValueError(describe_model_requirements(mt))

    # Build param_pool (kept close to original)
    param_pool: dict[str, Any] = {"t0": t0_jd, "u0": float(u0), "tE": float(tE)}

    if enable_parallax:
        param_pool.update({"piEN": float(pp["piEN"]), "piEE": float(pp["piEE"])})

    # Model-specific params (canonical)
    if mt == "USBL":
        s_val = float(mp["s"])
        q_val = float(mp["q"])
        rho_val = float(mp["rho"])
        alpha_val = float(mp["alpha"])

        param_pool.update({"s": s_val, "sep": s_val, "separation": s_val})
        param_pool.update({"q": q_val, "mass_ratio": q_val})
        param_pool.update({"rho": rho_val, "alpha": alpha_val})

    elif mt == "FSPL":
        param_pool["rho"] = float(mp["rho"])

    elif mt in ("NFW", "BS"):
        param_pool["t_m"] = float(mp["t_m"])

    # Telescope flux parameters (AB mags -> nJy flux)
    assert source_mags is not None
    for b in bands:
        s_mag = float(source_mags[b])
        f_s, f_b = _compute_fsource_fblend_for_band(
            band=b,
            source_mag=s_mag,
            blend_mags=blend_mags,
            blend_g=blend_g,
        )
        param_pool[f"fsource_{b}"] = float(f_s)
        param_pool[f"fblend_{b}"] = float(f_b)

    # Build ordered parameter vector
    expected = ev_model.pyLIMA_standards_dictionnary
    ordered = [0.0] * len(expected)

    # Final safety: make sure *required* names for this model exist in pyLIMA expected list
    # and are present in param_pool. This prevents "silent 0.0" bugs.
    required_names = ["t0", "u0", "tE"]
    if enable_parallax:
        required_names += ["piEN", "piEE"]

    if mt == "FSPL":
        required_names += ["rho"]
    if mt == "USBL":
        required_names += ["separation", "mass_ratio", "rho", "alpha"]
        #required_names += ["s", "q", "rho", "alpha"]
    if mt in ("NFW", "BS"):
        required_names += ["t_m"]

    for name in required_names:
        if name not in expected:
            raise RuntimeError(
                f"Internal mismatch: pyLIMA expected params for model {mt} do not include {name!r}.\n"
                f"Expected keys: {list(expected.keys())}"
            )
        if name not in param_pool:
            raise RuntimeError(
                f"Internal error: required param {name!r} missing from param_pool for model {mt}."
            )

    for name, idx in expected.items():
        if name in param_pool:
            ordered[idx] = param_pool[name]

    py_params = ev_model.compute_pyLIMA_parameters(ordered)
    simulator.simulate_lightcurve(ev_model, py_params, add_noise=False)

    # Collect outputs
    out: dict[str, pd.DataFrame] = {}
    for b in bands:
        lc = telescope_objs[b].lightcurve
        time_jd = lc["time"].value
        mag = lc["mag"].value
        flux_njy = mag2nJy(mag)
        out[b] = (
            pd.DataFrame({"mjd": time_jd - MJD_OFFSET, "flux_njy": flux_njy, "mag": mag})
            .sort_values("mjd", ignore_index=True)
        )

    return out


# Helper functions to write and plot
def write_lightcurves_txt(
    lightcurves: dict[str, pd.DataFrame],
    output_file: str,
    *,
    meta: dict | None = None,
) -> None:
    """
    Write multi-band light curves to a single tab-delimited text file.

    Parameters
    ----------
    lightcurves : dict of str to pandas.DataFrame
        Mapping ``band -> DataFrame``. Each DataFrame must contain the columns ``"mjd"``, ``"mag"``, and ``"flux_njy"``.
    output_file : str
        Path to the output text file.
    meta : dict or None, optional
        Optional metadata to include as comment lines at the top of the file. Default is None.

    Returns
    -------
    None
        This function writes a file and returns None.
    """
    meta = {} if meta is None else dict(meta)

    rows = []
    for band, df in lightcurves.items():
        tmp = df.copy()
        tmp["filter"] = band
        rows.append(tmp)

    all_df = pd.concat(rows, axis=0, ignore_index=True).sort_values("mjd", ignore_index=True)

    with open(output_file, "w") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("# Columns: mjd\tmag\tflux_njy\tfilter\n")
        all_df[["mjd", "mag", "flux_njy", "filter"]].to_csv(f, sep="\t", index=False, header=True)


def plot_lightcurves_mag(
    lightcurves: dict[str, pd.DataFrame],
    *,
    title: str | None = None,
    invert_mag: bool = True,
):
    """
    Plot multi-band light curves in AB magnitudes versus MJD.

    Parameters
    ----------
    lightcurves : dict of str to pandas.DataFrame
        Mapping ``band -> DataFrame``. Each DataFrame must contain the columns
        ``"mjd"`` and ``"mag"``.
    title : str or None, optional
        Optional plot title. Default is None.
    invert_mag : bool, optional
        If True, invert the y-axis so smaller magnitudes plot higher. Default is True.

    Returns
    -------
    None
        Displays a Matplotlib figure and returns None.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    for band, df in lightcurves.items():
        plt.plot(df["mjd"].values, df["mag"].values, marker=".", linestyle="none", label=band)

    plt.xlabel("MJD")
    plt.ylabel("AB mag")
    if invert_mag:
        plt.gca().invert_yaxis()

    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Example USBL (requires s, q, rho; alpha/origin optional)
    lcs_usbl = simulate_perfect_event(
        model_type="USBL",
        ra=270.66,
        dec=-35.70,
        t0_mjd=62000.0,
        u0=0.08,
        tE=200.0,
        bands=("g", "r", "i", "z", "y"),
        source_mags={"g": 20.0, "r": 19.7, "i": 19.4, "z": 18.8, "y": 18.2},
        blend_mags={"g": 22.0, "r": 21.7, "i": 21.4, "z": 20.8, "y": 20.2},
        parallax_params={"piEN": 0.12, "piEE": -0.05},
        model_params={"s": 1.15, "q": 1e-3, "rho": 1e-3, "alpha": 0.4, "origin": "center_of_mass"},
    )
    write_lightcurves_txt(lcs_usbl, "perfect_pylima_USBL.txt", meta={"model": "USBL"})
    plot_lightcurves_mag(lcs_usbl, title="Perfect pyLIMA USBL (mag)")

    # Example BS (requires t_m)
    lcs_bs = simulate_perfect_event(
        model_type="BS",
        ra=270.66,
        dec=-35.70,
        t0_mjd=62000.0,
        u0=0.08,
        tE=66.0,
        bands=("g", "r", "i"),
        source_mags={"g": 20.0, "r": 19.7, "i": 19.4},
        blend_g={"g": 0.3, "r": 0.4, "i": 0.5},  # here source_mags are TOTAL baseline mags
        parallax_params=None,
        model_params={"t_m": 3},
    )

    write_lightcurves_txt(lcs_bs, "perfect_pylima_BS.txt", meta={"model": "BS"})
    plot_lightcurves_mag(lcs_bs, title="Perfect pyLIMA BS (mag)")
