#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from io import StringIO
from getpass import getpass
from astropy import units as u
from astropy import constants as const
from dl import queryClient as qc
from dl import authClient as ac


def login_datalab() -> None:
    """
    Prompt for Astro Data Lab credentials and authenticate the current session.

    This function interactively requests a username and password to obtain an authentication token (Data Lab client). 
    If the returned token is not valid, a `RuntimeError` is raised.

    Will soon replace this function 
    """
    print("Logging into Astro Data Lab...")

    token = ac.login(input("Username: "), getpass("Password: "))

    if not ac.isValidToken(token):
        raise RuntimeError("Login failed.")


def datalab_login() -> None:
    """
    Ensure an active Astro Data Lab authentication token exists.

    This function attempts to reuse an existing cached token via `ac.getToken()`.
    If no valid token is available (or token retrieval raises an exception),
    it falls back to an interactive login via `login_datalab()`.
    """
    try:
        tok = ac.getToken()

        if not ac.isValidToken(tok):
            login_datalab()

    except Exception:
        login_datalab()


def query_trilegal_stars(
    ra: float,
    dec: float,
    *,
    radius_deg: float = 0.1,
    ds_max_pc: float = 10000.0,
    limit: int = 5000,
    timeout: int = 600,
) -> pd.DataFrame:
    """
    Query a TRILEGAL star sample from Astro Data Lab (their `lsst_sim.simdr2` table) near an input sky position.

    The query uses `q3c_radial_query` to select sources within `radius_deg` of (`ra`, `dec`) and applies a 
    distance cut using the distance modulus `mu0`. Two derived columns are appended, the `distance_pc` (distance inferred from `mu0`)
    and the `mu_total` (total proper motion magnitude from `pmracosd` and `pmdec`).

    Parameters
    ----------
    ra : float
        Right ascension of the query center (degrees).
    dec : float
        Declination of the query center (degrees).
    radius_deg : float, optional
        Radial search radius (degrees). Default is 0.1 degrees.
    ds_max_pc : float, optional
        Maximum source distance (pc). Implemented via a distance modulus cut
        (`mu0 < 5*log10(ds_max_pc) - 5`). Default is 10000.0 parsecs.
    limit : int, optional
        Maximum number of rows to return. Default is 5000.
    timeout : int, optional
        Query timeout in seconds. Default is 600.

    Returns
    -------
    pandas.DataFrame
        Table of TRILEGAL sources with lowercase column names, containing:
        - Raw queried columns:
          `ra, dec, mu0, pmracosd, pmdec, umag, gmag, rmag, imag, zmag, ymag, logl, logte`
        - Added columns:
          `distance_pc, mu_total`
    """
    mu0_max = 5 * np.log10(float(ds_max_pc)) - 5

    query = f"""
        SELECT ra, dec, mu0, pmracosd, pmdec, umag, gmag, rmag, imag, zmag, ymag, logl, logte
        FROM lsst_sim.simdr2
        WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})
          AND mu0 < ({mu0_max})
        LIMIT {int(limit)}
    """

    res = qc.query(sql=query, format="csv", timeout=timeout)

    try:
        df = pd.read_csv(StringIO(res))
    except Exception:
        df = pd.read_csv(StringIO(res.decode()))

    df.columns = [c.lower() for c in df.columns]
    df["distance_pc"] = 10 ** ((df["mu0"] + 5) / 5)
    df["mu_total"] = np.sqrt(df["pmracosd"] ** 2 + df["pmdec"] ** 2)

    return df


def generate_physical_pairs(
    df: pd.DataFrame,
    n_events: int,
    *,
    min_dist_pc: float = 10.0,
    offset_pc: float = 0.1,
    random_seed: int | None = None,
    physical_vectors: bool = False,
) -> pd.DataFrame:
    """
    Construct source–lens (foreground) pairs from a TRILEGAL catalog.

    This routine selects background sources and assigns each one a randomly
    chosen foreground lens drawn from the same TRILEGAL sample, subject to a
    geometric distance ordering, such that a row is chosen as a source at distance ``D_S``
    and a lens is then drawn from objects with ``min_dist_pc < D_L < D_S - offset_pc``

    The output is an event-pair table containing source/lens distances, per-band
    magnitudes for both objects, and an estimate of relative proper motion.

    Parameters
    ----------
    df : pandas.DataFrame
        Input TRILEGAL table. Must contain (at minimum) the columns:
        ``distance_pc``, ``ra``, ``dec``, ``logl``, ``logte``,
        ``pmracosd``, ``pmdec``, ``mu_total``, and band magnitudes
        ``umag, gmag, rmag, imag, zmag, ymag``.
    n_events : int
        Number of source–lens pairs to generate (best effort; may return fewer).
    min_dist_pc : float, optional
        Minimum allowed lens distance (pc). Default is 10.0.
    offset_pc : float, optional
        Minimum separation in distance between source and lens (pc), enforced via
        ``D_L < D_S - offset_pc``. Default is 0.1.
    random_seed : int or None, optional
        Seed for the random number generator. Default is None.
    physical_vectors : bool, optional
        If True, compute the relative proper motion vector from the catalog
        components (``pmracosd``, ``pmdec``) and return a physically consistent
        trajectory angle. If False, only the magnitudes of the proper motions
        are used (``mu_total``) and a random relative angle is drawn; in this
        case ``traj_angle`` is set to NaN. Default is False.

    Returns
    -------
    pandas.DataFrame
        Table of generated pairs with one row per event. Columns include:

        - ``mu_rel`` : float
            Relative proper motion magnitude (mas/yr).
        - ``traj_angle`` : float
            Trajectory angle (radians) for the relative proper-motion vector,
            or NaN if ``physical_vectors=False``.
        - ``distance_source`` : float
            Source distance ``D_S`` (pc).
        - ``distance_blend`` : float
            Lens distance ``D_L`` (pc). (Named “blend” to match downstream usage.)
        - ``logl_source``, ``logte_source`` : float
            Source stellar parameters copied from TRILEGAL.
        - ``ra``, ``dec`` : float
            Sky position (degrees) copied from the selected source.
        - Per-band magnitudes:
            ``{u,g,r,i,z,y}_source`` and ``{u,g,r,i,z,y}_blend``.

        The returned DataFrame may contain fewer than ``n_events`` rows if the
        input catalog does not provide enough valid source–lens combinations.

    Notes
    -----
    - The input catalog is sorted by ``distance_pc``. Sources are drawn from the
      most distant 90% of objects (indices from 10% to end) to bias toward
      background sources and increase the chance of finding a foreground lens.
    - If ``physical_vectors=False``, ``mu_rel`` is computed from the scalar proper
      motion magnitudes using a random relative angle:
      ``mu_rel = sqrt(mu_S^2 + mu_L^2 - 2 mu_S mu_L cos(theta))``.
      The direction is intentionally deferred to a later prior.
    - A single foreground lens is drawn *with replacement* for each source; the
      same lens may appear in multiple pairs.

    Raises
    ------
    KeyError
        If required columns are missing from ``df``.
    """

    rng = np.random.default_rng(random_seed)
    df_sorted = df.sort_values(by="distance_pc").reset_index(drop=True)

    pairs: List[Dict[str, Any]] = []
    candidate_indices = np.arange(int(len(df_sorted) * 0.1), len(df_sorted))
    rng.shuffle(candidate_indices)

    for i in candidate_indices:
        if len(pairs) >= n_events:
            break

        source = df_sorted.iloc[i]
        d_source = float(source["distance_pc"])

        max_lens_dist = d_source - float(offset_pc)
        if max_lens_dist <= min_dist_pc:
            continue

        foreground = df_sorted[
            (df_sorted["distance_pc"] > min_dist_pc) & (df_sorted["distance_pc"] < max_lens_dist)
        ]
        if foreground.empty:
            continue

        lens = foreground.sample(n=1, random_state=int(rng.integers(1e9))).iloc[0]

        if physical_vectors:
            dmu_ra = float(source["pmracosd"] - lens["pmracosd"])
            dmu_dec = float(source["pmdec"] - lens["pmdec"])
            mu_rel = float(np.sqrt(dmu_ra**2 + dmu_dec**2))
            traj_angle = float(np.arctan2(dmu_dec, dmu_ra))
        else:
            mu_S = float(source["mu_total"])
            mu_L = float(lens["mu_total"])
            theta = float(rng.uniform(0, 2 * np.pi))
            mu_rel = float(np.sqrt(mu_S**2 + mu_L**2 - 2 * mu_S * mu_L * np.cos(theta)))
            traj_angle = np.nan

        row: Dict[str, Any] = {
            "mu_rel": mu_rel, # mas/yr
            "traj_angle": traj_angle, # rad (may be nan)
            "distance_source": d_source, # pc
            "distance_blend": float(lens["distance_pc"]),  # pc
            "logl_source": float(source["logl"]),
            "logte_source": float(source["logte"]),
            "ra": float(source["ra"]),
            "dec": float(source["dec"]),
        }

        for b in "ugrizy":
            row[f"{b}_source"] = float(source[f"{b}mag"])
            row[f"{b}_blend"] = float(lens[f"{b}mag"])

        pairs.append(row)

    return pd.DataFrame(pairs)


def calculate_trilegal_physics(
    row: dict | pd.Series,
    *,
    lens_mass_solar: float,
    angle_rad: Optional[float], # need to input this if traj_angle column is NaN
    semi_major_axis_au: Optional[float], # this is optional and if input will ensure that s_physical is returned
) -> dict | None:
    """
    Compute microlensing geometry and derived parameters for a TRILEGAL source–lens pair.

    Given a pre-constructed source–lens pairing (from the generate_physical_pairs function),
    this function computes the Einstein angle, Einstein timescale, finite-source parameter, 
    and the microlensing parallax components (optional).

    Note that in our current implementation, the lens mass is provided externally (i.e., as a prior), 
    rather than inferred from the TRILEGAL catalog (this could in principle be derived from the surface gravity and radius). 

    The trajectory angle used to decompose the parallax vector is taken from ``row["traj_angle"]`` when available (i.e., when using physical
    proper-motion vectors), otherwise it must be supplied via the ``angle_rad`` argument.
    
    The Einstein angle is computed as: ``theta_E = sqrt( (4GM/c^2) * (1/D_L - 1/D_S) )``.
    The relative parallax is calculated as: ``pi_rel = AU * (1/D_L - 1/D_S)``.
    The parallax magnitude is: ``pi_E = pi_rel / theta_E`` (dimensionless).
    The source angular radius is estimated from TRILEGAL luminosity and effective temperature via Stefan–Boltzmann to get ``R_*``, then ``theta_* = R_* / D_S``.
    
    The function will return None if the lens is not in front of the source (``(1/D_L - 1/D_S) <= 0``) or if the ``traj_angle`` is NaN/missing and ``angle_rad`` is None.

    Parameters
    ----------
    row : dict or pandas.Series
        A single pair record containing (at minimum) the keys/columns:
        ``distance_source``, ``distance_blend``, ``mu_rel``, ``logl_source``,
        ``logte_source``, and optionally ``traj_angle``.
        Distances are interpreted as pc and ``mu_rel`` as mas/yr.
    lens_mass_solar : float
        Lens mass in solar masses.
    angle_rad : float or None
        Trajectory angle in radians used to decompose the parallax magnitude into
        (piEN, piEE). Required if ``row["traj_angle"]`` is missing or NaN; ignored
        otherwise.
    semi_major_axis_au : float or None
        Physical semi-major axis (AU) used to compute a physically-motivated binary
        separation ``s_physical = theta_a / theta_E`` (dimensionless). If None,
        ``s_physical`` is returned as None.

    Returns
    -------
    dict or None
        Dictionary of derived quantities, or None if the geometry is unphysical or
        required inputs are missing. Returned keys:

        - ``tE`` : float
            Einstein timescale (days), computed as ``tE = theta_E / mu_rel``.
        - ``rho`` : float
            Finite-source size parameter ``rho = theta_* / theta_E`` (dimensionless).
        - ``piEN`` : float
            North component of microlensing parallax (dimensionless).
        - ``piEE`` : float
            East component of microlensing parallax (dimensionless).
        - ``s_physical`` : float or None
            Dimensionless binary separation computed from ``semi_major_axis_au``,
            or None if not requested.
        - ``M_L`` : float
            Lens mass (Msun).
        - ``D_S`` : float
            Source distance (pc).
        - ``D_L`` : float
            Lens distance (pc).
        - ``mu_rel`` : float
            Relative proper motion (mas/yr).
        - ``theta_E_mas`` : float
            Einstein angle (mas).
        - ``pi_rel_mas`` : float
            Relative parallax (mas).
    """
    G, c, au = const.G, const.c, const.au

    M_L = float(lens_mass_solar) * u.M_sun
    D_S = float(row["distance_source"]) * u.pc
    D_L = float(row["distance_blend"]) * u.pc
    mu_rel = float(row["mu_rel"]) * (u.mas / u.yr)

    term = (4 * G * M_L / c**2).to(u.au)
    dist_term = (1 / D_L - 1 / D_S).to(1 / u.au)
    if dist_term.value <= 0:
        return None

    theta_E_rad = np.sqrt(term * dist_term).decompose().value * u.rad
    theta_E_mas = theta_E_rad.to(u.mas)
    tE = (theta_E_mas / mu_rel).to(u.day).value

    L_S = (10 ** float(row["logl_source"])) * const.L_sun
    T_S = (10 ** float(row["logte_source"])) * u.K
    R_S = np.sqrt(L_S / (4 * np.pi * const.sigma_sb * T_S**4))

    theta_S_rad = (R_S / D_S).decompose().value * u.rad
    rho = (theta_S_rad / theta_E_rad).decompose().value

    pi_rel = (au / D_L - au / D_S).decompose()
    pi_E_mag = (pi_rel / theta_E_rad).decompose().value

    traj_angle = row.get("traj_angle", np.nan)
    if pd.notnull(traj_angle):
        ang = float(traj_angle)
    else:
        if angle_rad is None:
            # must provide the prior when physical_vectors=False
            return None
        ang = float(angle_rad)

    piEN = float(pi_E_mag * np.sin(ang))
    piEE = float(pi_E_mag * np.cos(ang))
    pi_rel_mas = float(pi_rel.to_value(u.mas, equivalencies=u.dimensionless_angles()))

    s_val = None
    if semi_major_axis_au is not None:
        a_au = float(semi_major_axis_au) * u.au
        s_rad = (a_au / D_L).decompose().value
        s_val = float(s_rad / theta_E_rad.value)

    return {
        "tE": float(tE),
        "rho": float(rho),
        "piEN": float(piEN),
        "piEE": float(piEE),
        "s_physical": s_val,
        "M_L": float(M_L.to_value(u.M_sun)),
        "D_S": float(D_S.to_value(u.pc)),
        "D_L": float(D_L.to_value(u.pc)),
        "mu_rel": float(mu_rel.to_value(u.mas / u.yr)),
        "theta_E_mas": float(theta_E_mas.value),
        "pi_rel_mas": float(pi_rel_mas),
    }


def draw_blend_g(rng: np.random.Generator, bands: str = "ugrizy") -> Dict[str, float]:
    """
    Draw per-band blend fractions for a simulated event. This is a convenience helper 
    for cases where a dedicated per-band prior object is not being used.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used to draw the blend fractions.
    bands : str, optional
        Photometric bands to generate blend fractions for. Each character is treated
        as a band label (e.g., ``"ugrizy"``). Default is ``"ugrizy"``.

    Returns
    -------
    dict
        Dictionary mapping each band label to a blend fraction in [0, 1) (e.g., ``{"g": 0.12, "r": 0.63, ...}``).
    """
    return {b: float(rng.uniform(0, 1)) for b in bands}

# The microlensing models we support at the moment
ModelType = Literal["PSPL", "FSPL", "USBL", "NFW", "BS"]

class Prior:
    """
    The base class for parameter priors, used to generate random draws from a prior distribution.

    Methods
    -------
    sample(rng, n=None)
        Draw one sample (if ``n is None``) or ``n`` samples (if ``n`` is an int).
    describe()
        Return a short string description of the prior, used for logging/debugging.
    """

    def sample(self, rng: np.random.Generator, n: Optional[int] = None) -> Any:
        """
        Draw sample(s) from the prior distribution.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to draw samples.
        n : int or None, optional
            If None, return a single draw. If an integer, return ``n`` draws.
            Default is None.

        Returns
        -------
        Any
            A single draw or a collection of draws, depending on ``n``.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """
        Return a short description of the prior.

        Returns
        -------
        str
            A human-readable name/description of the prior. By default this is
            the class name, but subclasses may override for more detail.
        """
        return self.__class__.__name__


@dataclass
class Fixed(Prior):
    """
    Deterministic prior that always returns a fixed value.

    Parameters
    ----------
    value : Any
        The constant value returned by this prior. This can be a scalar (float/int),
        a string, or any object expected by downstream code.
    """

    value: Any

    def sample(self, rng: np.random.Generator, n: Optional[int] = None) -> Any:
        """
        Return the fixed value (optionally repeated ``n`` times).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator (unused for this prior; included for API
            compatibility).
        n : int or None, optional
            If None, return a single value. If an integer, return a list of length
            ``n`` containing repeated values. Default is None.

        Returns
        -------
        Any
            ``value`` if ``n is None``; otherwise a list of repeated values.
        """
        if n is None:
            return self.value
        return [self.value for _ in range(n)]

    def describe(self) -> str:
        """
        Return a human-readable description of the prior.

        Returns
        -------
        str
            Description in the form ``"Fixed(<value>)"``.
        """
        return f"Fixed({self.value})"


@dataclass
class Uniform(Prior):
    """
    Continuous uniform prior over an interval [low, high).

    Parameters
    ----------
    low : float
        Lower bound of the uniform distribution.
    high : float
        Upper bound of the uniform distribution.
    """

    low: float
    high: float

    def sample(self, rng: np.random.Generator, n: Optional[int] = None) -> Any:
        """
        Draw sample(s) from a uniform distribution on [low, high).

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to draw samples.
        n : int or None, optional
            If None, return a single float. If an integer, return an array of
            ``n`` samples. Default is None.

        Returns
        -------
        float or numpy.ndarray
            A single draw if ``n is None``; otherwise an array of length ``n``.
        """
        return rng.uniform(self.low, self.high, size=n)

    def describe(self) -> str:
        """
        Return a human-readable description of the prior.

        Returns
        -------
        str
            Description in the form ``"Uniform(low, high)"``.
        """
        return f"Uniform({self.low}, {self.high})"


@dataclass
class LogUniform(Prior):
    """
    Log-uniform prior over an interval [low, high).

    Samples are drawn uniformly in log10-space and then exponentiated (``x = 10**U(log10(low), log10(high))``).

    Parameters
    ----------
    low : float
        Lower bound of the distribution. Must be > 0.
    high : float
        Upper bound of the distribution. Must be > 0 and > low.
    """

    low: float
    high: float

    def sample(self, rng: np.random.Generator, n: Optional[int] = None) -> Any:
        """
        Draw sample(s) from a log-uniform distribution on [low, high). Will raise an error
        if ``low`` or ``high`` are not positive, or if ``high <= low``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to draw samples.
        n : int or None, optional
            If None, return a single float. If an integer, return an array of
            ``n`` samples. Default is None.

        Returns
        -------
        float or numpy.ndarray
            A single draw if ``n is None``; otherwise an array of length ``n``.
        """
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniform requires low > 0 and high > 0.")

        if self.high <= self.low:
            raise ValueError("LogUniform requires high > low.")

        lo = np.log10(self.low)
        hi = np.log10(self.high)

        return 10 ** rng.uniform(lo, hi, size=n)

    def describe(self) -> str:
        """
        Return a human-readable description of the prior.

        Returns
        -------
        str
            Description in the form ``"LogUniform(low, high)"``.
        """
        return f"LogUniform({self.low}, {self.high})"

@dataclass
class Choice(Prior):
    """
    Discrete prior that samples from a finite set of options. If ``n`` is None, 
    returns a single element from ``options``. If ``n`` is an integer, returns an array of ``n`` draws.
   
    Parameters
    ----------
    options : tuple of Any
        The allowed values to sample from.
    """

    options: Tuple[Any, ...]

    def sample(self, rng: np.random.Generator, n: Optional[int] = None) -> Any:
        """
        Draw one or more samples from ``options``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to draw samples.
        n : int or None, optional
            If None, return a single draw. If an integer, return an array of
            ``n`` draws. Default is None.

        Returns
        -------
        Any
            A single option if ``n is None``; otherwise a NumPy array of length
            ``n`` containing sampled options.
        """
        if len(self.options) == 0:
            raise ValueError("Choice prior requires a non-empty options tuple.")

        if n is None:
            return rng.choice(self.options)
        return rng.choice(self.options, size=n)

    def describe(self) -> str:
        """
        Return a human-readable description of the prior.

        Returns
        -------
        str
            Description in the form ``"Choice([opt1, opt2, ...])"``.
        """
        return f"Choice({list(self.options)})"


@dataclass
class PerBandUniform(Prior):
    """
    Per-band uniform prior that returns a dictionary of independent draws (sampling is per event only!).

    This is intended for parameters that must be specified separately for each
    photometric band (e.g., per-band blend fractions).

    Parameters
    ----------
    low : float
        Lower bound of the uniform distribution for each band.
    high : float
        Upper bound of the uniform distribution for each band.
    bands : str, optional
        Photometric bands to sample. Each character is treated as a band label.
        Default is ``"ugrizy"``.
    """

    low: float
    high: float
    bands: str = "ugrizy"

    def sample(self, rng: np.random.Generator, n: Optional[int] = None) -> Any:
        """
        Draw a per-band dictionary of independent uniform samples.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to draw samples.
        n : int or None, optional
            Must be None. This prior is not vectorized because it returns a
            dictionary of per-band scalars.

        Returns
        -------
        dict
            Mapping ``{band: value}`` where each ``value`` is drawn uniformly
            from [low, high).
        """
        if n is not None:
            raise ValueError("PerBandUniform.sample() is not supported; sample per-event!")
        return {b: float(rng.uniform(self.low, self.high)) for b in self.bands}

    def describe(self) -> str:
        """
        Return a human-readable description of the prior.

        Returns
        -------
        str
            Description in the form ``"PerBandUniform(low, high, bands='...')"``.
        """
        return f"PerBandUniform({self.low}, {self.high}, bands='{self.bands}')"


@dataclass(frozen=True)
class GenerationConfig:
    """
    Configuration flags controlling TRILEGAL-based event-table generation.

    This class determines which physical parameters must be provided via priors,
    how relative proper motion is handled, whether parallax terms are computed, and
    how blending and (for binaries) separations are treated.

    Attributes
    ----------
    model_type : ModelType, optional
        Microlensing model family to generate parameters for. Supported values are
        ``"PSPL"``, ``"FSPL"``, ``"USBL"``, ``"NFW"``, and ``"BS"``. Default is
        ``"PSPL"``.
    enable_parallax : bool, optional
        If True, compute and store microlensing parallax components (piEN, piEE).
        If False, piEN and piEE are stored as NaN in the output table. Default is True.
    physical_vectors : bool, optional
        If True, derive relative proper motion and trajectory angle from TRILEGAL
        proper-motion components (``pmracosd``, ``pmdec``). If False, compute only
        the magnitude of relative proper motion using ``mu_total`` and treat the
        trajectory angle as an external prior. Default is False.
    custom_blending : bool, optional
        If True, blending is controlled by an external per-band blend fraction prior
        (e.g., ``blend_g``), and the stored ``source_mags`` represent baseline total
        magnitudes (source+blend). If False, store the TRILEGAL source-only and lens-only 
        magnitudes directly. Default is True.
    use_physical_s : bool, optional
        USBL-only option. If True and a semi-major axis prior is provided, compute a
        physically motivated separation ``s_physical`` and use it as ``s`` when
        available; otherwise fall back to sampling ``s`` from a prior. If False,
        always sample ``s`` from its prior. Default is True.
    """

    model_type: ModelType = "PSPL"
    enable_parallax: bool = True
    physical_vectors: bool = False
    custom_blending: bool = True

    # This is for USBL only -- if True and semi_major_axis prior provided, use s_physical, otherwise use the s prior
    use_physical_s: bool = True



def required_prior_names(cfg: GenerationConfig) -> List[str]:
    """
    Return the list of prior names required for a given generation configuration.

    This helper inspects :the GenerationConfig class and determines which priors must
    be provided to generate a valid event-parameter table.

    If ``cfg.enable_parallax`` is set to True and ``cfg.physical_vectors`` is False, the ``"traj_angle_rad"`` is 
        required to decompose the parallax magnitude into (piEN, piEE) when ``traj_angle`` is not available from TRILEGAL.

    If ``cfg.custom_blending`` is True, ``"blend_g"`` is required (per-band blend fraction prior).

    If ``cfg.model_type == "USBL"``, then ``"q"``, ``"alpha"``, and ``"origin"`` are required, plus either ``"semi_major_axis_au_or_s"`` when 
        ``cfg.use_physical_s`` is True (meaning the user must provide either ``semi_major_axis_au`` to compute a physical separation or 
        an explicit ``s`` prior fallback), or ``"s"`` when ``cfg.use_physical_s`` is False.

    If ``cfg.model_type`` is ``"NFW"`` or ``"BS"``, then ``"t_m"`` is required.

    Parameters
    ----------
    cfg : GenerationConfig
        Configuration specifying model type and which physical effects are enabled
        (parallax, physical proper-motion vectors, blending, and USBL separation mode).

    Returns
    -------
    list of str
        Names of required priors. The base set always includes t0, u0 and the lens mass ("lens_mass_solar"").
        Additional requirements depend on ``cfg``.
    """
    req = ["t0", "u0", "lens_mass_solar"]

    # angle which is only when enable_parallax=True and physical_vectors=False (since traj_angle is NaN)
    if cfg.enable_parallax and (not cfg.physical_vectors):
        req.append("traj_angle_rad")

    # blending -- only required if using custom blending
    if cfg.custom_blending:
        req.append("blend_g")

    if cfg.model_type == "USBL":
        req.extend(["q", "alpha", "origin"])
        if cfg.use_physical_s:
            # require semi_major_axis_au to compute s_physical, if missing, remember that users need to provide s!!
            req.append("semi_major_axis_au_or_s")
        else:
            req.append("s")

    if cfg.model_type in ("NFW", "BS"):
        req.append("t_m")

    return req


def default_priors(cfg: GenerationConfig) -> Dict[str, Prior]:
    """
    Construct a default set of priors consistent with the generation configuration.

    This function returns a dictionary mapping parameter names to the objects in the Prior class, and
    is intended to provide a starting point that users can override as needed.

    Parameters
    ----------
    cfg : GenerationConfig
        Configuration controlling which model type is used and whether additional
        priors are required (parallax angle, blending, USBL parameters, etc.).

    Returns
    -------
    dict of str -> Prior
        Dictionary of prior objects keyed by parameter name.

        Base priors (always included)
        - ``t0`` : Uniform(60000.0, 63650.0)
            Event time of closest approach (MJD).
        - ``u0`` : Uniform(0.0, 0.3)
            Impact parameter in units of theta_E.
        - ``lens_mass_solar`` : Uniform(0.1, 1.0)
            Lens mass in solar masses.

        Conditional priors
        - ``traj_angle_rad`` : Uniform(0.0, 2pi)
            Included only if ``cfg.enable_parallax`` is True and
            ``cfg.physical_vectors`` is False (i.e., when the trajectory angle is
            not available from TRILEGAL and must be provided as a prior).
        - ``blend_g`` : PerBandUniform(0.0, 1.0, bands="ugrizy")
            Included only if ``cfg.custom_blending`` is True.

        USBL priors (only if ``cfg.model_type == "USBL"``)
        - ``q`` : LogUniform(1e-4, 1.0)
            Binary mass ratio.
        - ``alpha`` : Uniform(0.0, 2pi)
            Binary-lens trajectory angle (model convention).
        - ``origin`` : Choice(("center_of_mass", "central_caustic", "second_caustic"))
            Reference point for the binary-lens parameterization.
        - ``semi_major_axis_au`` : Uniform(0.1, 28.0)
            Included if ``cfg.use_physical_s`` is True; used to compute a physical
            separation ``s_physical`` when possible.
        - ``s`` : LogUniform(0.4, 2.5)
            Always included for USBL as a fallback when physical separation is not
            available (e.g., if ``semi_major_axis_au`` is not provided or
            ``cfg.use_physical_s`` is False).

        NFW/BS priors (only if ``cfg.model_type`` in {"NFW", "BS"})
        - ``t_m`` : Uniform(0.1, 5.0)
            Model-specific timescale parameter.
    """

    pri: Dict[str, Prior] = {
        "t0": Uniform(60000.0, 63650.0),
        "u0": Uniform(0.0, 0.3),
        "lens_mass_solar": Uniform(0.1, 1.0),
    }

    if cfg.enable_parallax and (not cfg.physical_vectors):
        pri["traj_angle_rad"] = Uniform(0.0, 2 * np.pi)

    if cfg.custom_blending:
        pri["blend_g"] = PerBandUniform(0.0, 1.0, bands="ugrizy")

    if cfg.model_type == "USBL":
        pri["q"] = LogUniform(1e-4, 1.0)
        pri["alpha"] = Uniform(0.0, 2 * np.pi)
        pri["origin"] = Choice(("center_of_mass", "central_caustic", "second_caustic"))
        if cfg.use_physical_s:
            pri["semi_major_axis_au"] = Uniform(0.1, 28.0)
            # fallback to s in case semi_major_axis_au missing
            pri["s"] = LogUniform(0.4, 2.5)
        else:
            pri["s"] = LogUniform(0.4, 2.5)

    if cfg.model_type in ("NFW", "BS"):
        pri["t_m"] = Uniform(0.1, 5.0)

    return pri


def validate_priors(cfg: GenerationConfig, priors: Dict[str, Prior]) -> None:
    """
    Validate that a prior dictionary satisfies the requirements of a configuration.

    This function enforces that all priors required by the required_prior_names function
    are present in the provided ``priors`` mapping. A good sanity check -- it raises errors if
    priors are missing, but functions with extra priors (but does print a warning).

    Parameters
    ----------
    cfg : GenerationConfig
        Configuration specifying which priors are required (model type, parallax,
        blending mode, physical-vector handling, etc.).
    priors : dict of str -> Prior
        Dictionary mapping prior names to :class:`Prior` instances.
    """

    req = required_prior_names(cfg)

    missing: List[str] = []
    for name in req:
        if name == "semi_major_axis_au_or_s":
            if ("semi_major_axis_au" not in priors) and ("s" not in priors):
                missing.append("semi_major_axis_au OR s")
        else:
            if name not in priors:
                missing.append(name)

    if missing:
        raise ValueError(
            "Missing required priors for this configuration:" + "\n  - " + "\n  - ".join(missing))

    # helpful warning for unused priors
    used = set()
    used.update(["t0", "u0", "lens_mass_solar"])
    if cfg.enable_parallax and (not cfg.physical_vectors):
        used.add("traj_angle_rad")
    if cfg.custom_blending:
        used.add("blend_g")
    if cfg.model_type == "USBL":
        used.update(["q", "alpha", "origin"])
        if cfg.use_physical_s:
            used.update(["semi_major_axis_au", "s"])  # s is fallback
        else:
            used.add("s")
    if cfg.model_type in ("NFW", "BS"):
        used.add("t_m")

    extras = sorted([k for k in priors.keys() if k not in used])
    if extras:
        print(f"WARNING: Unused priors for cfg={cfg}: {extras}")


def describe_requirements(cfg: GenerationConfig, priors: Optional[Dict[str, Prior]] = None) -> str:
    """
    Build a human-readable summary of required (and optionally provided) priors.

    Parameters
    ----------
    cfg : GenerationConfig
        Configuration specifying which priors are required.
    priors : dict of str -> Prior, optional
        If provided, the set of priors that will be used. These are included in
        the returned summary under "Provided priors:".

    Returns
    -------
    str
        Text summary.
    """
    req = required_prior_names(cfg)
    lines = []
    lines.append(f"Configuration: {cfg}")
    lines.append("Required priors:")

    for name in req:
        if name == "semi_major_axis_au_or_s":
            lines.append("  - semi_major_axis_au OR s (USBL separation source)")
        else:
            lines.append(f"  - {name}")

    if priors is not None:
        lines.append("\nProvided priors:")
        for k, v in sorted(priors.items(), key=lambda x: x[0]):
            lines.append(f"  - {k}: {v.describe()}")

    return "\n".join(lines)


# Now can can construct the TRILEGAL table
def generate_trilegal_event_table(
    *,
    n_events: int,
    ra: float,
    dec: float,
    cfg: GenerationConfig,
    priors: Optional[Dict[str, Prior]] = None,
    random_seed: Optional[int] = None,
    # TRILEGAL query controls
    radius_deg: float = 0.2,
    query_limit: int = 50000,
    ds_max_pc: float = 10000.0,
    timeout: int = 600,
    # pairing controls
    min_dist_pc: float = 10.0,
    offset_pc: float = 0.1,
) -> pd.DataFrame:
    """
    Generate a microlensing event-parameter table using TRILEGAL stars + user priors.

    This is the main function for building the parameter DataFrame. This function first queries a 
    TRILEGAL catalog from Astro Data Lab around (ra, dec), then constructs source–lens pairs consistent with foreground lensing geometry.
    It then samples user-specified priors (t0, u0, lens mass, etc.) and computes derived microlensing physics (tE, rho, theta_E, parallax terms).
    Model-specific parameters (e.g., USBL q/alpha/origin/s) are then added, and the per-band photometry is handled according to the blending mode.
    
    Note that our pair selection can result in the same lens being used across multiple events (sampling with replacement), depending on the density of valid foreground objects.

    Parameters
    ----------
    n_events : int
        Number of events to generate (note that it may return fewer if filtering removes too many pairs).
    ra : float
        Right ascension of the TRILEGAL query center (degrees).
    dec : float
        Declination of the TRILEGAL query center (degrees).
    cfg : GenerationConfig
        Generation configuration controlling parallax, physical-vector usage,
        blending behavior, and model type.
    priors : dict of str -> Prior, optional
        Mapping from parameter name to the Prior class. If None, defaults are created
        via default_priors(cfg). The required keys depend on ``cfg`` and are
        validated by the validate_priors function.
    random_seed : int or None, optional
        Seed for the RNG used in prior sampling and (when applicable) pair selection.
        Default is None.

    radius_deg : float, optional
        TRILEGAL cone-search radius passed to the query_trilegal_stars function. Default is 0.2.
    query_limit : int, optional
        Maximum number of TRILEGAL rows retrieved from Data Lab. Default is 50000.
    ds_max_pc : float, optional
        Maximum distance (pc) used to cut TRILEGAL sources via a distance-modulus
        filter in the query_trilegal_stars function. Default is 10000.0.
    timeout : int, optional
        Data Lab query timeout in seconds. Default is 600.

    min_dist_pc : float, optional
        Minimum lens distance (pc) used when pairing sources and lenses. Default is 10.0.
    offset_pc : float, optional
        Minimum source–lens distance separation (pc), enforced as ``D_L < D_S - offset_pc``
        during pairing. Default is 0.1.

    Returns
    -------
    pandas.DataFrame
        Event table with one row per generated simulation. Currently the code includes:

        Common microlensing parameters
        - ``sim_id`` : int
            Row index (for reproducibility/debugging).
        - ``model_type`` : str
            Model identifier (from ``cfg.model_type``).
        - ``ra``, ``dec`` : float
            Sky position (degrees).
        - ``t0`` : float
            Time of closest approach (MJD; drawn from prior).
        - ``u0`` : float
            Impact parameter (theta_E units; drawn from prior).
        - ``tE`` : float
            Einstein timescale (days; derived).
        - ``rho`` : float
            Finite-source size parameter (derived).
        - ``piEN``, ``piEE`` : float
            Parallax components (derived) or NaN if ``cfg.enable_parallax=False``.
        - ``M_L`` : float
            Lens mass (in units of Msun).
        - ``D_S``, ``D_L`` : float
            Source and lens distances (pc; from TRILEGAL pairing).
        - ``mu_rel`` : float
            Relative proper motion (mas/yr).
        - ``theta_E_mas`` : float
            Einstein angle (mas).
        - ``pi_rel_mas`` : float
            Relative parallax (mas).
        - ``a_au`` : float or None
            Semi-major axis (AU; sampled only when requested/available).
        - ``s_physical`` : float or None
            Separation implied by ``a_au`` and lens distance, in theta_E units.

        Photometry, which depends on the cfg.custom_blending input:
        - If ``cfg.custom_blending=True``:
            * ``blend_g`` : dict
              Per-band blend fractions (prior-driven).
            * ``source_mags`` : dict
              Baseline total magnitudes (source+blend) in each band, intended for later splitting using ``blend_g``.
            * ``blend_mags`` : None
        - If ``cfg.custom_blending=False``:
            * ``blend_g`` : None
            * ``source_mags`` : dict
              TRILEGAL source-only magnitudes.
            * ``blend_mags`` : dict
              TRILEGAL lens-only magnitudes.

        Model-specific columns
        - For ``cfg.model_type == "USBL"``:
            ``q``, ``alpha``, ``origin``, ``s``.
        - For ``cfg.model_type`` in {``"NFW"``, ``"BS"``}:
            ``t_m``
    """
    rng = np.random.default_rng(random_seed)

    if priors is None:
        priors = default_priors(cfg)

    validate_priors(cfg, priors)
    print(describe_requirements(cfg, priors))

    print('Querying...')
    stars = query_trilegal_stars(
        ra, dec,
        radius_deg=radius_deg,
        ds_max_pc=ds_max_pc,
        limit=query_limit,
        timeout=timeout,
    )

    pairs = generate_physical_pairs(
        stars,
        n_events,
        min_dist_pc=min_dist_pc,
        offset_pc=offset_pc,
        random_seed=random_seed,
        physical_vectors=cfg.physical_vectors,
    )

    rows: List[Dict[str, Any]] = []

    for i in range(len(pairs)):
        p = pairs.iloc[i]

        # Sample the common priors
        t0 = float(priors["t0"].sample(rng))
        u0 = float(priors["u0"].sample(rng))
        lens_mass = float(priors["lens_mass_solar"].sample(rng))

        # angle (only needed when traj_angle is NaN and parallax enabled)
        angle = None
        # We need an angle for the calculation, even if this is not used for parallax later
        if not cfg.physical_vectors: 
            angle = float(priors["traj_angle_rad"].sample(rng))

        # optional semi-major axis (for USBL physical s)
        a_au = None
        if (cfg.model_type == "USBL") and cfg.use_physical_s and ("semi_major_axis_au" in priors):
            a_au = float(priors["semi_major_axis_au"].sample(rng))

        phys = calculate_trilegal_physics(
            p,
            lens_mass_solar=lens_mass,
            angle_rad=angle,
            semi_major_axis_au=a_au,
        )
        if phys is None:
            continue

        # parallax switch (storing NaN if not enabled!)
        piEN = phys["piEN"] if cfg.enable_parallax else np.nan
        piEE = phys["piEE"] if cfg.enable_parallax else np.nan

        event: Dict[str, Any] = {
            "sim_id": int(i),
            "model_type": str(cfg.model_type),
            "ra": float(ra),
            "dec": float(dec),
            "t0": t0,
            "u0": u0,
            "tE": float(phys["tE"]),
            "rho": float(phys["rho"]),
            "piEN": float(piEN),
            "piEE": float(piEE),
            "M_L": float(phys["M_L"]),
            "D_S": float(phys["D_S"]),
            "D_L": float(phys["D_L"]),
            "mu_rel": float(phys["mu_rel"]),
            "theta_E_mas": float(phys["theta_E_mas"]),
            "pi_rel_mas": float(phys["pi_rel_mas"]),
            "a_au": a_au,
            "s_physical": phys["s_physical"],
        }

        # photometry from pairs
        true_source_mags = {b: float(p[f"{b}_source"]) for b in "ugrizy"}
        true_blend_mags = {b: float(p[f"{b}_blend"]) for b in "ugrizy"}

        if cfg.custom_blending:
            # prior-driven blend_g
            bg = priors["blend_g"].sample(rng) if isinstance(priors["blend_g"], PerBandUniform) else draw_blend_g(rng)
            baseline_total_mags = {b: float(true_source_mags[b] - 2.5 * np.log10(1.0 + bg[b])) for b in "ugrizy"}

            event["blend_g"] = bg
            event["source_mags"] = baseline_total_mags
            event["blend_mags"] = None
        else:
            event["blend_g"] = None
            event["source_mags"] = true_source_mags
            event["blend_mags"] = true_blend_mags

        # These are the model-specific priors
        if cfg.model_type == "USBL":
            event["q"] = float(priors["q"].sample(rng))
            event["alpha"] = float(priors["alpha"].sample(rng))
            event["origin"] = str(priors["origin"].sample(rng))

            # s: prefer s_physical when available, else we will use the s prior
            if cfg.use_physical_s and (phys["s_physical"] is not None):
                s_val = float(phys["s_physical"])
            else:
                # require s prior when physical separation is not available
                s_val = float(priors["s"].sample(rng)) if "s" in priors else float(phys["s_physical"])
            event["s"] = float(s_val)

        elif cfg.model_type in ("NFW", "BS"):
            event["t_m"] = float(priors["t_m"].sample(rng))

        # For FSPL the rho parameter is already derived so no extra priors needed
        rows.append(event)

        if len(rows) >= n_events:
            break

    out = pd.DataFrame(rows)
    if len(out) < n_events:
        print(f"WARNING: Requested {n_events} events, but only {len(out)} were generated due to the filtering.")
    return out

import pdb; pdb.set_trace()
if __name__ == "__main__":

    datalab_login()

    cfg = GenerationConfig(
        model_type="USBL",
        enable_parallax=True,
        physical_vectors=False,
        custom_blending=False,
        use_physical_s=True,
    )

    # Best to just start with our defaults then manually override whatever we need
    pri = default_priors(cfg)

    #pri["t0"] = Uniform(61000.0, 62000.0) # Could ensure signal with adaptive t0? (within +- of some point)
    #pri["u0"] = Uniform(0.0, 0.5)
    pri["lens_mass_solar"] = Uniform(0.001, 100.0)
    #pri["q"] = LogUniform(1e-5, 1e-2) # Markus said there is a planet mass ratio function!
    #pri["semi_major_axis_au"] = Uniform(0.5, 10.0) # Markus noted that 0.1-10 is most common
    #pri["blend_g"] = PerBandUniform(0.0, 0.3) # 0-1 is fine -- 0 for NFW/Boson? 

    df = generate_trilegal_event_table(
        n_events=50000,
        ra=270.66,
        dec=-35.70,
        cfg=cfg,
        priors=pri,
        random_seed=42,
        radius_deg=0.2,
        query_limit=50000,
    )

    df.to_pickle("trilegal_event_params.pkl")

    df_csv = df.copy()
    for col in ("source_mags", "blend_mags", "blend_g"):
        df_csv[col] = df_csv[col].apply(lambda x: None if x is None else str(x))
    df_csv.to_csv("trilegal_event_params.csv", index=False)

    import pdb; pdb.set_trace()
    print(df.head())
    print(f"\nSaved {len(df)} events.")





