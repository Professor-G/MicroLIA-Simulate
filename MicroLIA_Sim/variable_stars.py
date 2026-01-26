#!/usr/bin/env python3
#This code loads a multiband RR Lyrae template, 
#randomly selects one, folds the input times to phase,
#interpolates the template per band, and applies a single magnitude 
#offset so the simulated g/r/i light curves have realistic colors and a chosen mean brightness.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from importlib import resources

def load_rr_template_txt(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load a single multiband RR Lyrae template from a .txt file.

    The file is expected to have a header 'Mag,Phase,Band' and rows like:
    Mag,Phase,Band
    17.30,0.000,g
    17.31,0.001,g
    ...

    Parameters
    ----------
    path : str or Path
        Path to the template .txt file.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'phase': mapping band -> phase array (float)
        - 'mag': mapping band  -> magnitude array (float)
        - 'bands': list of available band names.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=",")  # columns: Mag, Phase, Band

    phases_by_band: Dict[str, np.ndarray] = {}
    mags_by_band: Dict[str, np.ndarray] = {}

    for band, group in df.groupby("Band"):
        group_sorted = group.sort_values("Phase")
        phases_by_band[band] = group_sorted["Phase"].to_numpy(dtype=float)
        mags_by_band[band] = group_sorted["Mag"].to_numpy(dtype=float)

    return {
        "phase": phases_by_band,
        "mag": mags_by_band,
        "bands": sorted(phases_by_band.keys()),
    }


def pick_random_template_path(
    root_dir: str | Path,
    rr_type: str = "RRab",
    rng: Optional[np.random.Generator] = None,
) -> Path:
    """
    Pick a random template file.

    Parameters
    ----------
    root_dir : str or Path
        Root directory where the templates live.
    rr_type : {'RRab', 'RRc'}
        Type of RR Lyrae to draw.
    rng : numpy.random.Generator, optional
        RNG instance. If None, a default generator is used.

    Returns
    -------
    Path
        Path to a randomly selected .txt template.
    """
    root_dir = Path(root_dir)
    folder = root_dir / rr_type
    files = sorted(folder.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No .txt templates found in {folder}")

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, len(files))
    return files[idx]


def _interp_periodic(
    phase_templ: np.ndarray,
    mag_templ: np.ndarray,
    phase_query: np.ndarray,
) -> np.ndarray:
    """
    Interpolate template magnitudes as a function of phase in [0, 1).

    Assumes:
    - phase_templ is sorted and spans roughly [0, ~1].
    - phase_query is in [0, 1) (we enforce this with modulo 1).

    Uses linear interpolation.
    """
    # Ensure arrays
    phase_templ = np.asarray(phase_templ, dtype=float)
    mag_templ = np.asarray(mag_templ, dtype=float)
    phase_query = np.asarray(phase_query, dtype=float) % 1.0

    # phase_templ should be sorted already, but just to be safe:
    order = np.argsort(phase_templ)
    phase_sorted = phase_templ[order]
    mag_sorted = mag_templ[order]

    # np.interp requires ascending x; query points must lie within range
    # Template goes up to ~1.0–1.01, query is in [0,1), so this is fine.
    mag_interp = np.interp(phase_query, phase_sorted, mag_sorted)
    return mag_interp


def simulate_rrlyrae_multiband_from_txt(
    times_by_band: Mapping[str, np.ndarray],
    period: float,
    template_path: str | Path,
    reference_band: str = "r",
    reference_mean_mag: Optional[float] = None,
    T0: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate multiband RR Lyrae light curves using a single template .txt file.

    Parameters
    ----------
    times_by_band : mapping
        Mapping from band name to 1D array of observation times (e.g.,{'g': mjd_g, 'r': mjd_r, 'i': mjd_i}
    period : float
        RR Lyrae period in days.
    template_path : str or Path
        Path to the chosen template .txt file.
    reference_band : str, optional
        Band used to anchor the baseline magnitude.
    reference_mean_mag : float, optional
        Desired mean magnitude in the reference band. If None, the
        template's native magnitudes are used.
    T0 : float, optional
        Epoch corresponding to phase = 0 (in same units as `times_by_band`).
        If None, a random T0 is drawn uniformly in [0, period).
    rng : numpy.random.Generator, optional
        RNG used when drawing random T0. Ignored if T0 is provided.

    Returns
    -------
    dict
        Dictionary mapping band -> simulated magnitudes array.
        Keys match those in the `times_by_band` argument.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Load template
    template = load_rr_template_txt(template_path)
    phase_templ = template["phase"]
    mag_templ = template["mag"]
    available_bands: Sequence[str] = template["bands"]

    if reference_band not in available_bands:
        raise ValueError(
            f"Reference band '{reference_band}' not in template bands {available_bands}"
        )

    # Draw T0 if not provided
    if T0 is None:
        T0 = rng.uniform(0.0, period)

    # Compute offset to enforce desired mean mag in reference band
    ref_mag_template = mag_templ[reference_band]
    if reference_mean_mag is not None:
        delta_mag = reference_mean_mag - np.mean(ref_mag_template)
    else:
        delta_mag = 0.0

    # Interpolate per band
    mags_out: Dict[str, np.ndarray] = {}

    for band, t in times_by_band.items():
        if band not in available_bands:
            raise ValueError(
                f"Band '{band}' requested but not in template bands {available_bands}"
            )

        t = np.asarray(t, dtype=float)

        # Phase-folding
        phase = ((t - T0) / period) % 1.0

        band_phase_templ = phase_templ[band]
        band_mag_templ = mag_templ[band]

        mag_interp = _interp_periodic(band_phase_templ, band_mag_templ, phase)
        mags_out[band] = mag_interp + delta_mag

    return mags_out


def simulate_rrlyae(times, bailey, period, reference_band, reference_mean_mag, rng):
    """
    Generate multiband RR Lyrae (or Cepheid-like) light curves by sampling a
    template and phase-folding it onto the provided cadence. RRLyrae templates are
    from Baeza-Villagra et al. (2025). It randomly selects a template file from the package data directory
    and interpolates the template magnitudes at the phase-folded observation times. It
    then applies a single magnitude offset so the reference band has the desired mean magnitude.

    If `period` is None, it draws a period from a simple distribution based on the Bailey type (`bailey`):
        * 1 -> RRab : Normal(0.6, 0.15) days
        * 2 -> RRc : Normal(0.33, 0.10) days
        * 3 -> Cepheid-like : 10**LogNormal(0.0, 0.2) days
   
    Parameters
    ----------
    times : Mapping[str, array_like]
        Mapping from band name to 1D array of observation times, e.g.
        {'g': t_g, 'r': t_r, 'i': t_i}. Times must be in days.
    bailey : int
        Variability class selector. Expected values:
        1 (RRab), 2 (RRc), 3 (Cepheid-like).
    period : float or None
        Period in days. If None, a period is drawn based on `bailey`.
    reference_band : str
        Band used to anchor the mean magnitude (passed through to the underlying
        simulator).
    reference_mean_mag : float
        Desired mean magnitude in `reference_band` (passed through to the
        underlying simulator).
    rng : numpy.random.Generator, optional
        RNG used when drawing random T0. Ignored if T0 is provided.


    Returns
    -------
    dict
        Dictionary mapping each input band name to an array of simulated
        magnitudes evaluated at `times[band]`.
    """

    if period is None:
        if bailey == 1: #RRab
            period = np.random.normal(0.6, 0.15)
            _TYPE_ = "RRab"
        elif bailey == 2:
            period = np.random.normal(0.33, 0.1)
            _TYPE_ = "RRc"
        elif bailey == 3: #Cepheids
            period = np.random.lognormal(0., 0.2)
            period = 10**period
            _TYPE_ = "RRc"

    root = str(resources.files(__package__) / 'data') # Always use the templates which I've put in the data dir!
    tpl_path = pick_random_template_path(root, rr_type=_TYPE_, rng=rng)

    mags = simulate_rrlyrae_multiband_from_txt(
        times_by_band=times,
        period=period,
        template_path=tpl_path,
        reference_band=reference_band,
        reference_mean_mag=reference_mean_mag,
        rng=rng
    )

    return mags 


