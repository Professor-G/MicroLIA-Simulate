.. _Parameter_Generation:

Parameter Generation
====================

This module generates physically motivated microlensing event-parameter tables by combining a TRILEGAL stellar catalog query from Astro Data Lab (the ``lsst_sim.simdr2``, single stars only, but there is also a binary table!), and a user-defined priors for parameters not provided (or not yet inferred) from TRILEGAL.

The main function is :func:`MicroLIA_Sim.param_generation.generate_trilegal_event_table`, which returns a :class:`pandas.DataFrame` with one row per simulated event.

These event parameters can then be used to simulate the microlensing events using the microlensing module!

Workflow overview
-----------------

The procedure for generating the events dataframe is as follows:

1. Authenticate with Astro Data Lab (it is interactive so not yet cluster-friendly, will work on this!).
2. Query a TRILEGAL star sample near (RA, Dec) with optional distance-modulus cut.
3. Construct source–lens pairs with the required distance ordering (foreground lens).
4. Draw user priors (e.g., ``t0``, ``u0``, and optionally the lens mass).
5. Compute derived microlensing quantities (``tE``, ``rho``, ``theta_E``, parallax).
6. Attach per-band photometry (and blending mags if blending is enabled) and model-specific parameters.


Core concepts
-------------

Generation configuration
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~MicroLIA_Sim.param_generation.GenerationConfig` controls how the table is built:

- ``model_type``: one of ``"PSPL"``, ``"FSPL"``, ``"USBL"``, ``"NFW"``, ``"BS"``
- ``enable_parallax``: store ``piEN``/``piEE`` or store NaNs
- ``physical_vectors``: use catalog proper-motion components (physically consistent trajectory angle)
- ``custom_blending``: store baseline total magnitudes + per-band blend ratios, or store source+blend mags directly
- ``use_physical_s`` (USBL only): compute a physical separation from a semi-major axis prior when provided
- ``use_trilegal_mass``: if ``True``, use the actual mass of the lens star from the TRILEGAL catalog; if ``False``, sample from a ``lens_mass_solar`` prior.
- ``sample_tE_directly``: if ``True``, sample the timescale ``tE`` from a prior and derive the physical lens mass; if ``False`` (default), use the lens mass (from prior or catalog) to derive ``tE``.

Priors
^^^^^^

Priors are objects that generate random draws for simulation parameters. These are provided as a dictionary mapping ``parameter_name -> Prior`` (e.g., ``"u0" -> Uniform(...)``).

Available prior classes include:

- :class:`~MicroLIA_Sim.param_generation.Fixed` — always returns a constant value.
- :class:`~MicroLIA_Sim.param_generation.Uniform` — uniform on ``[low, high)``.
- :class:`~MicroLIA_Sim.param_generation.LogUniform` — log-uniform on ``[low, high)`` (uniform in log10-space).
- :class:`~MicroLIA_Sim.param_generation.Choice` — discrete draw from a finite set of options.
- :class:`~MicroLIA_Sim.param_generation.PerBandUniform` — independent per-band draws (returns a dict; intended for per-event sampling).



Required priors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The base required priors are always:

- ``t0`` (MJD)
- ``u0`` (impact parameter in units of :math:`\theta_E`)

Additional requirements depend on the configuration:

- If ``use_trilegal_mass=False`` (default), need to input ``lens_mass_solar`` (Msun)
- If ``sample_tE_directly=True``, need to input ``tE`` (days)
- If ``sample_tE_directly=False`` (default) and ``use_trilegal_mass=False``, need to input ``lens_mass_solar`` (Msun)
- If ``enable_parallax=True`` and ``physical_vectors=False``, need to input ``traj_angle_rad``
- If ``custom_blending=True``, need to input ``blend_g`` (per-band blend flux ratio prior)
- If ``model_type="USBL"``, need to input ``q``, ``alpha``, ``origin``, and either:
  - ``semi_major_axis_au`` **or** ``s``
- If ``model_type in {"NFW","BS"}``, need to input ``t_m``

TRILEGAL query and pairing
--------------------------

TRILEGAL query
^^^^^^^^^^^^^^

:func:`~MicroLIA_Sim.param_generation.query_trilegal_stars` queries ``lsst_sim.simdr2`` within a cone (using ``q3c_radial_query``) and applies a distance modulus cut via ``mu0``. It appends:

- ``distance_pc`` (from ``mu0``)
- ``mu_total`` (from ``pmracosd`` and ``pmdec``)

Pair construction
^^^^^^^^^^^^^^^^^

:func:`~MicroLIA_Sim.param_generation.generate_physical_pairs` constructs (source, lens) pairs such that:

- the source is at distance :math:`D_S`
- the lens is drawn from foreground objects satisfying
  :math:`D_L < D_S - \Delta D` and :math:`D_L > D_\mathrm{min}`

The same foreground lens may appear in multiple events (sampling with replacement).

Derived microlensing quantities
-------------------------------

:func:`~MicroLIA_Sim.param_generation.calculate_trilegal_physics` computes:

- Einstein angle :math:`\theta_E`
- Einstein timescale :math:`t_E = \theta_E / \mu_\mathrm{rel}`
- finite-source parameter :math:`\rho = \theta_\star/\theta_E`
- relative parallax and microlensing parallax components (optional)
- optional physical binary separation (if a semi-major axis prior is supplied)

.. note::

   The lens mass can be supplied via a prior (``lens_mass_solar``), inferred directly from the TRILEGAL catalog
   (``use_trilegal_mass = True``), or can be derived from a timescale prior (``sample_tE_directly = True``).

Blending convention
-------------------

When ``custom_blending=True``, the table stores:

- ``blend_g``: per-band blend **flux ratio** (drawn from a prior)
- ``source_mags``: **baseline total** magnitudes (source+blend)

The conversion used is:

.. math::

   m_\mathrm{base} = m_\mathrm{source} - 2.5\log_{10}\left(1 + f_\mathrm{blend}\right),

where :math:`f_\mathrm{blend} = F_\mathrm{blend} / F_\mathrm{source}`.

When ``custom_blending=False``, blending is computed from the source/lens fluxes and the table stores TRILEGAL magnitudes directly:

- ``source_mags``: TRILEGAL source-only magnitudes
- ``blend_mags``: TRILEGAL lens-only magnitudes

Examples
--------

PSPL example with custom lens mass prior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from MicroLIA_Sim.param_generation import (
       datalab_login,
       GenerationConfig,
       default_priors,
       generate_trilegal_event_table,
       Uniform, LogUniform, PerBandUniform
   )

   # Authenticate with DataLab (should only need to do once...)
   datalab_login()

   cfg = GenerationConfig(
       model_type="PSPL",
       enable_parallax=True,
       physical_vectors=False,
       custom_blending=True,
   )

   # Best to just start with our defaults then manually override whatever we need
   priors = default_priors(cfg)

   # Manual overrides
   priors["t0"] = Uniform(61000.0, 62000.0) 
   priors["u0"] = Uniform(0.0, 0.5)
   priors["lens_mass_solar"] = Uniform(0.001, 100.0)
   priors["blend_g"] = PerBandUniform(0.0, 0.3) 

   df = generate_trilegal_event_table(
       n_events=5000,
       ra=270.66,
       dec=-35.70,
       cfg=cfg,
       priors=priors,
       random_seed=1909,
       radius_deg=0.2,
       query_limit=5000,
   )

Sampling tE directly and deriving the lens mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reproduce specific timescale distributions (e.g., to match survey sensitivity) rather than a mass function, you can sample ``tE`` directly by enabling the ``sample_tE_directly`` argument. The simulation will invert the physics to find the required lens mass.

.. code-block:: python

    cfg = GenerationConfig(
        model_type="PSPL",
        sample_tE_directly=True, # Enable direct tE sampling
        enable_parallax=True,
        use_trilegal_mass=False # Ignored when sample_tE_directly=True
    )

    priors = default_priors(cfg)

    # Need to provide 'tE' instead of 'lens_mass_solar'
    priors["tE"] = LogUniform(1.0, 100.0)

    df = generate_trilegal_event_table(
        n_events=1000,
        ra=270.66,
        dec=-35.70,
        cfg=cfg,
        priors=priors
    )

    # The output df["M_L"] column will now contain the derived masses!

Using TRILEGAL catalog masses (No Mass Prior)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can force the simulation to use the physical mass of the selected lens star from the catalog. In this case, you do not provide a ``lens_mass_solar`` prior.

.. code-block:: python

    cfg = GenerationConfig(
        model_type="PSPL",
        enable_parallax=True,
        use_trilegal_mass=True # Enable usage of TRILEGAL 'mass' column
    )

    # Note: default_priors(cfg) will NOT include 'lens_mass_solar' now
    priors = default_priors(cfg)

    df = generate_trilegal_event_table(
        n_events=1000,
        ra=270.66,
        dec=-35.70,
        cfg=cfg,
        priors=priors
    )

USBL with physical separation (semi-major axis prior)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np

   from MicroLIA_Sim.param_generation import (
       GenerationConfig,
       default_priors,
       generate_trilegal_event_table,
       Uniform,
   )

   cfg = GenerationConfig(
       model_type="USBL",
       enable_parallax=True,
       physical_vectors=False,
       custom_blending=False,
       use_physical_s=True,
   )

   # Best to just start with our default priors then manually override whatever we need
   priors = default_priors(cfg)

   # Manual override
   priors["lens_mass_solar"] = Uniform(0.001, 100.0)

   # Provide a semi-major axis prior (AU) so s_physical can be computed
   priors["semi_major_axis_au"] = Uniform(0.1, 10.0)

   df = generate_trilegal_event_table(
       n_events=5000,
       ra=270.66,
       dec=-35.70,
       cfg=cfg,
       priors=priors,
       random_seed=1909,
   )

Turning off parallax
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from MicroLIA_Sim.param_generation import GenerationConfig, default_priors, generate_trilegal_event_table

   cfg = GenerationConfig(
       model_type="PSPL",
       enable_parallax=False, # will store piEN/piEE as NaN
       physical_vectors=False,
       custom_blending=True,
   )

   priors = default_priors(cfg)

   df = generate_trilegal_event_table(
       n_events=1000,
       ra=10.0,
       dec=-10.0,
       cfg=cfg,
       priors=priors,
       random_seed=1909,
   )

Output schema
-------------

The event table includes:

- Common microlensing fields: ``t0, u0, tE, rho, piEN, piEE, theta_E_mas, pi_rel_mas, mu_rel``
- Geometry: ``D_S, D_L, M_L``
- Model metadata: ``sim_id, model_type, ra, dec``
- Optional separation: ``a_au, s_physical`` (USBL when requested)
- Photometry fields:
  - if ``custom_blending=True``: ``blend_g`` (dict), ``source_mags`` (dict), ``blend_mags=None``
  - else: ``source_mags`` (dict), ``blend_mags`` (dict), ``blend_g=None``
- Model-specific fields:
  - USBL: ``q, alpha, origin, s``
  - NFW/BS: ``t_m``

API reference
-------------

.. automodule:: MicroLIA_Sim.param_generation
   :members:
   :undoc-members:
   :show-inheritance:
