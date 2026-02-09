.. _Microlensing_Simulations:

Microlensing Simulations
========================

This module provides the functionality for generating noise-free(“perfect”) multi-band microlensing light curves with `pyLIMA`.

The primary function is:

- :func:`~MicroLIA_Sim.microlensing.simulate_perfect_event`

.. note::

   At import time, this module overwrites ``pyLIMA.toolbox.brightness_transformation`` module-level constants to ensure consistent AB-flux conversions:

   - ``ZERO_POINT = 31.4``
   - ``EXPOSURE_TIME = 1.0``

   This is intentional because pyLIMA’s defaults are configured for Roman!


Quickstart
----------

Simulate a standard point-source/point-lens (PSPL) event in multiple bands:

.. code-block:: python

   from MicroLIA_Sim.microlensing import simulate_perfect_event

   lcs = simulate_perfect_event(
       model_type="PSPL",
       ra=270.66,
       dec=-35.70,
       t0_mjd=62000.0,
       u0=0.08,
       tE=200.0,
       bands=("g", "r", "i"),
       source_mags={"g": 20.0, "r": 19.7, "i": 19.4},
       blend_mags={"g": 22.0, "r": 21.7, "i": 21.4}, # including blending is optional
       parallax_params=None, # optional
       model_params=None, # PSPL has no extra params
   )

The returned object is a dictionary ``{band: DataFrame}``, where each DataFrame has:

- ``mjd`` (float)
- ``flux_njy`` (float)
- ``mag`` (float)

The module provides a few helper functions to save and plot the simulated event:

.. code-block:: python

   from MicroLIA_Sim.microlensing import write_lightcurves_txt, plot_lightcurves_mag

   write_lightcurves_txt(lcs, "example_event.txt", meta={"model": "PSPL"})
   plot_lightcurves_mag(lcs, title="Perfect PSPL Lightcurve")



Supported models
----------------

Model selection is controlled by ``model_type``. Model-specific parameters are passed via ``model_params`` and validated so that users know what simulation parameters are needed.


.. list-table::
   :header-rows: 1
   :widths: 12 22 50

   * - model_type
     - Required ``model_params``
     - Notes
   * - ``"PSPL"``
     - (none)
     - Point-source point-lens (standard Paczyński).
   * - ``"FSPL"``
     - ``rho``
     - Finite-source point-lens.
   * - ``"USBL"``
     - ``s, q, rho``
     - Uniform-source binary-lens. Optional: ``alpha`` (default 0.0), ``origin`` (default ``"center_of_mass"``).
   * - ``"NFW"``
     - ``t_m``
     - Extended lens. Mass-function tables are **auto-loaded** from packaged ``data/`` on first use.
   * - ``"BS"``
     - ``t_m``
     - Extended lens. Mass-function tables are **auto-loaded** from packaged ``data/`` on first use.


Photometry and blending
-----------------------

You must supply ``source_mags`` for all requested bands.

Blending can be specified in **exactly one** of the following ways:

1) Physical blending via ``blend_mags``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``source_mags`` are **source-only** AB magnitudes per band.
- ``blend_mags`` are **blend-only** AB magnitudes per band.

.. code-block:: python

   lcs = simulate_perfect_event(
       model_type="PSPL",
       ra=270.66, dec=-35.70,
       t0_mjd=62000.0, u0=0.08, tE=200.0,
       bands=("g","r","i"),
       source_mags={"g": 20.0, "r": 19.7, "i": 19.4},
       blend_mags={"g": 22.0, "r": 21.7, "i": 21.4},
   )

2) Blend-ratio method via ``blend_g``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``blend_g`` is the flux ratio ``g = f_blend / f_source``.
- If you use ``blend_g``, then ``source_mags`` are interpreted as the **total baseline** magnitudes (source + blend).
- ``blend_g`` may be a scalar (applied to all bands) or a per-band dictionary.

.. code-block:: python

   lcs = simulate_perfect_event(
       model_type="PSPL",
       ra=270.66, dec=-35.70,
       t0_mjd=62000.0, u0=0.08, tE=200.0,
       bands=("g","r","i"),
       source_mags={"g": 19.8, "r": 19.5, "i": 19.2},   # TOTAL baseline mags here
       blend_g={"g": 0.3, "r": 0.4, "i": 0.5},
   )


Parallax
--------

Parallax can be enabled by passing:

- ``parallax_params={"piEN": ..., "piEE": ...}``

Disable parallax by passing ``parallax_params=None``.


Time grids
----------

If you do not pass ``time_grid_jd``, the module generates an adaptive grid that is:

- dense near the peak (``t0 ± peak_width_factor * tE``)
- sparse in the wings (out to ``t0 ± window_size_days``)

where ``peak_width_factor`` is set to 5.0 (sets the half-width of the densely sampled core as ``peak_width_factor * tE_days``) and ``window_size_days`` is set to 4000 days -- these arguments are not currently tunable!

.. note::

   The ``simulate_perfect_event`` fuctions accepts ``t0_mjd`` but ``time_grid_jd`` is **JD**. An internal ``MJD_OFFSET = 2400000.5`` is applied. This is something I will update soon to avoid confusion...


Extended-lens models (NFW / BS)
-------------------------------

For ``model_type="NFW"`` or ``"BS"``, the enclosed-mass tables are bundled with the package under ``MicroLIA_Sim/data/`` and are loaded automatically when needed.

.. code-block:: python

   lcs = simulate_perfect_event(
       model_type="BS",
       ra=270.66, dec=-35.70,
       t0_mjd=62000.0, u0=0.08, tE=66.0,
       bands=("g","r","i"),
       source_mags={"g": 20.0, "r": 19.7, "i": 19.4},
       blend_g={"g": 0.3, "r": 0.4, "i": 0.5},
       model_params={"t_m": 3.0},
   )


Example: Different event types
-------------------------------

Below we demonstrate how to simulate a variety of event types including a stellar lens with a planetary companion and a free floating planet, showing explicitly all the options/priors that need to be set (and those that do not)

First we import the necessary modules to generate the event parameters and to do the simulations. 

.. code-block:: python

    import numpy as np
    import pandas as pd

    from MicroLIA_Sim import param_generation as pg
    from MicroLIA_Sim import microlensing as ul

    # Login to astrodatalab (although shouldn't be required after first time log in...)
    pg.login_datalab()


PSPL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configuration used for computing the event parameters
    # Note that for parallax the physical vectors are not used by default, as this will likely bias us to the TRILEGAL sample
    # I think it's best to random the geomtry randomly 
    cfg_bh = pg.GenerationConfig(
        model_type="PSPL",
        use_trilegal_mass=False, # set to True if you want code to use the actual mass of the lens from TRILEGAL
        sample_tE_directly=True, # if True then tE is input as a prior, in which case code derives the lens mass
        enable_parallax=True, # Whether to include parallax
        physical_vectors=False, # Whether to use TRILEGAL vectors, if False we sample angle (traj_angle_rad) instead (0-2pi)
        custom_blending=True, # Use 'g' flux ratio method, if False will use the lens flux to approximate the blending
        use_physical_s=False, # NOT USED FOR PSPL MODEL
    )

    # Priors
    priors_bh = {
        # Impact pram and t0 are always required
        "t0": pg.Uniform(60000.0, 63650.0),
        "u0": pg.Uniform(0.0, 0.5),
        
        # Either input tE as a prior or input the lens mass (or don't input either and set ``use_trilegal_mass``=True)
        "tE": pg.LogUniform(50.0, 300.0),
        # "lens_mass_solar": pg.Uniform(0.1, 1.0), # Unused because ``sample_tE_directly``=True

        # Parallax, need to input trajectory angle since ``physical_vectors``=False
        "traj_angle_rad": pg.Uniform(0.0, 2*np.pi), 

        # Blending, need to provide the blending factors since ``custom_blending``=True
        # Set range to (0.0, 0.0) to disable blending! 
        "blend_g": pg.PerBandUniform(0.0, 1.0, bands="ugrizy"),

        # The priors listed below are only for USBL events, do not input for PSPL
        # "q": pg.LogUniform(1e-5, 1e-2),
        # "s": pg.LogUniform(0.4, 2.0),
        # "alpha": pg.Uniform(0.0, 2*np.pi),
        # "origin": pg.Fixed("center_of_mass"),
        # "semi_major_axis_au": pg.Uniform(1.0, 5.0),

        # The prior listed below is only for NFW/BS events, do not input for PSPL
        # "t_m": pg.Uniform(1.0, 10.0),
    }

    # Run query and generate the params table
    df_bh = pg.generate_trilegal_event_table(n_events=1, ra=270.66, dec=-35.70, cfg=cfg_bh, priors=priors_bh)

    # Now use those parameters to do the simulation! Only one event was generated so query the first row
    row = df_bh.iloc[0]

    # Define the parallax components which the simulation code explicitly requires
    parallax_args = None
    if not np.isnan(row["piEN"]):
        parallax_args = {"piEN": float(row["piEN"]), "piEE": float(row["piEE"])}

    # Run the simulator, note that the PSPL model has no extra 'model_params'
    lcs_bh = ul.simulate_perfect_event(
        model_type="PSPL",
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        t0_mjd=float(row["t0"]),
        u0=float(row["u0"]),
        tE=float(row["tE"]),
        bands=("g", "r", "i", "z", "y"),
        time_grid_jd=None, # Will use the adaptive candence Rache suggested, but can be custom array of timestamps (JD).
        source_mags=row["source_mags"],
        blend_g=row["blend_g"],
        parallax_params=parallax_args,
        model_params={} # Empty for PSPL
    )
    ul.plot_lightcurves_mag(lcs_bh, title="Black Hole Simulation")

USBL (Planetary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configuration for generating event parameters
    cfg_planet = pg.GenerationConfig(
        model_type="USBL",
        use_trilegal_mass=False, # set to True if you want code to use the actual mass of the lens from TRILEGAL
        sample_tE_directly=True, # We input tE
        enable_parallax=True, # Include parallax
        physical_vectors=False, # Random angle
        custom_blending=True, # Use 'g' flux ratio
        use_physical_s=True, # Set to True to input separation in AU and derive the projected 's'.
    )

    # Set the Priors
    priors_planet = {
        # Always required
        "t0": pg.Uniform(60000.0, 63650.0),
        "u0": pg.Uniform(0.0, 0.1),
        
        # Event timescale or mass if ``sample_tE_directly``=False
        "tE": pg.LogUniform(10.0, 50.0),
        # "lens_mass_solar": pg.Uniform(0.1, 1.0), # Unused because ``sample_tE_directly``=True

        # Parallax
        "traj_angle_rad": pg.Uniform(0.0, 2*np.pi),

        # Blending
        "blend_g": pg.PerBandUniform(0.0, 1.0, bands="ugrizy"),

        # USBL specific params
        "q": pg.LogUniform(1e-5, 1e-2), # Planet Mass Ratio
        "alpha": pg.Uniform(0.0, 2*np.pi),
        "origin": pg.Fixed("center_of_mass"),
        "semi_major_axis_au": pg.Uniform(1.5, 5.2), # Physical Distance (AU), used because ``use_physical_s``=True
        # "s": pg.LogUniform(0.4, 2.0), # Only used if ``use_physical_s``=False

        # Only used for NFW/BS, do not set!
        # "t_m": pg.Uniform(1.0, 10.0),
    }

    # Run query and generate the params table
    df_planet = pg.generate_trilegal_event_table(n_events=1, ra=270.66, dec=-35.70, cfg=cfg_planet, priors=priors_planet)

    # Now use those parameters to do the simulation! Only one event was generated so query the first row
    row = df_planet.iloc[0]

    # Set the parallax components which the simulator explicitly requires
    parallax_args = None
    if not np.isnan(row["piEN"]):
        parallax_args = {"piEN": float(row["piEN"]), "piEE": float(row["piEE"])}

    # Set the USBL Params which the simulator explicitly requires
    usbl_params = {
        "q": float(row["q"]),
        "s": float(row["s"]), # the derived projected separation
        "rho": float(row["rho"]), # The finite source size, code computes this
        "alpha": float(row["alpha"]),
        "origin": str(row["origin"])
    }

    # Run the simulator
    lcs_planet = ul.simulate_perfect_event(
        model_type="USBL",
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        t0_mjd=float(row["t0"]),
        u0=float(row["u0"]),
        tE=float(row["tE"]),
        bands=("g", "r", "i", "z", "y"),
        time_grid_jd=None, # Will use the adaptive candence Rache suggested, but can be custom array of timestamps (JD).
        source_mags=row["source_mags"],
        blend_g=row["blend_g"],
        parallax_params=parallax_args,
        model_params=usbl_params # Passing the dictionary with the parallax and USBL-specific params 
    )
    ul.plot_lightcurves_mag(lcs_planet, title="USBL (Planetary System)")


FSPL (Free-floating planets)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configuration for event parameters
    cfg_ffp = pg.GenerationConfig(
        model_type="FSPL",
        sample_tE_directly=True, # We input tE
        enable_parallax=False, # Disabled right? Event likely too short for parallax to really affect the lightcurve
        physical_vectors=False, # Irrelevant now because ``enable_parallax``=False
        custom_blending=True, # Use 'g' flux ratio
        use_physical_s=False, # Not needed
        use_trilegal_mass=False, # Not needed
    )

    # Set the priors
    priors_ffp = {
        # Common priors always used
        "t0": pg.Uniform(60000.0, 63650.0),
        "u0": pg.Uniform(0.0, 0.05),         
        
        # Event timescale (Or input mass directly)
        "tE": pg.LogUniform(0.05, 2.0),
        # "lens_mass_solar": pg.Uniform(0.1, 1.0), 

        # Parallax (although in this example it is disabled -- ``enable_parallax``=False)
        # "traj_angle_rad": pg.Uniform(0.0, 2*np.pi), # Unused (Parallax False)

        # Blending
        "blend_g": pg.PerBandUniform(0.0, 1.0, bands="ugrizy"),

        # USBL-specific params (rho is computed by the code)
        # "q": pg.LogUniform(1e-5, 1e-2),
        # "s": pg.LogUniform(0.4, 2.0),
        # "alpha": pg.Uniform(0.0, 2*np.pi),
        # "origin": pg.Fixed("center_of_mass"),
        # "semi_major_axis_au": pg.Uniform(1.0, 5.0),

        # Only for NFW/BS
        # "t_m": pg.Uniform(1.0, 10.0),
    }

    # Generation params table
    df_ffp = pg.generate_trilegal_event_table(n_events=1, ra=270.66, dec=-35.70, cfg=cfg_ffp, priors=priors_ffp)

    # Simulation
    row = df_ffp.iloc[0]

    # Parallax dict is None in this case
    parallax_args = None

    # Extract FSPL Params, we only need rho
    fspl_params = {"rho": float(row["rho"])}

    # Run the simulator
    lcs_ffp = ul.simulate_perfect_event(
        model_type="FSPL",
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        t0_mjd=float(row["t0"]),
        u0=float(row["u0"]),
        tE=float(row["tE"]),
        bands=("g", "r", "i", "z", "y"),
        source_mags=row["source_mags"],
        blend_g=row["blend_g"],
        parallax_params=parallax_args,
        model_params=fspl_params
    )
    ul.plot_lightcurves_mag(lcs_ffp, title="Free-Floating Planet")

Boson Stars / Navarro–Frenk–White DM profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Configure
    cfg_bs = pg.GenerationConfig(
        model_type="BS",
        sample_tE_directly=True, # We input tE
        enable_parallax=True, # Enable parallax
        physical_vectors=False, # Sample angle
        custom_blending=True, # Use 'g' flux ratio
        use_physical_s=False, # Not needed
        use_trilegal_mass=False # Not needed
    )

    # Set the priors
    priors_bs = {
        # Common priors
        "t0": pg.Uniform(60000.0, 63650.0),
        "u0": pg.Uniform(0.0, 0.3),
        
        # Event timescale (or set the mass from some dark matter mass function?)
        "tE": pg.LogUniform(30.0, 150.0),
        # "lens_mass_solar": pg.Uniform(0.1, 1.0),

        # Parallax
        "traj_angle_rad": pg.Uniform(0.0, 2*np.pi),

        # Blending, set to 0 for an invisible lens (although there could still be blending due to crowded source field)
        "blend_g": pg.PerBandUniform(0.0, 0.0, bands="ugrizy"),

        # Need to set the t_m for BS/NFW
        "t_m": pg.Uniform(1.0, 3.0),

        # Only for USBL model, do not set
        # "q": pg.LogUniform(1e-5, 1e-2),
        # "s": pg.LogUniform(0.4, 2.0),
        # "alpha": pg.Uniform(0.0, 2*np.pi),
        # "origin": pg.Fixed("center_of_mass"),
        # "semi_major_axis_au": pg.Uniform(1.0, 5.0),
    }

    # Generate params
    df_bs = pg.generate_trilegal_event_table(n_events=1, ra=270.66, dec=-35.70, cfg=cfg_bs, priors=priors_bs)

    # Do the simulation
    row = df_bs.iloc[0]

    # Define the parallax components
    parallax_args = None
    if not np.isnan(row["piEN"]):
        parallax_args = {"piEN": float(row["piEN"]), "piEE": float(row["piEE"])}

    # Extract the Extended Model Params (only t_m)
    bs_params = {"t_m": float(row["t_m"])}

    # Run the simulator
    lcs_bs = ul.simulate_perfect_event(
        model_type="BS",
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        t0_mjd=float(row["t0"]),
        u0=float(row["u0"]),
        tE=float(row["tE"]),
        bands=("g", "r", "i", "z", "y"),
        source_mags=row["source_mags"],
        blend_g=row["blend_g"],
        parallax_params=parallax_args,
        model_params=bs_params
    )
    ul.plot_lightcurves_mag(lcs_bs, title="Boson Star")


