.. _Parameter_Generation:

Parameter Generation
====================

This module generates **physically motivated microlensing event-parameter tables** by combining a TRILEGAL stellar catalog query from **Astro Data Lab** (``lsst_sim.simdr2``, single stars only), and a user-defined **priors** for parameters not provided (or not yet inferred) from TRILEGAL.

The main function is :func:`MicroLIA_Sim.param_generation.generate_trilegal_event_table`, which returns a :class:`pandas.DataFrame` with one row per simulated event.

These event parameters can then be used to simulate the microlensing events!

Workflow overview
-----------------

The procedure for generating the events dataframe is as follows:

1. Authenticate with Astro Data Lab (token reuse if possible, it is interactive so not yet cluster-friendly).
2. Query a TRILEGAL star sample near (RA, Dec) with optional distance-modulus cut.
3. Construct source–lens pairs with the required distance ordering (foreground lens).
4. Draw user priors (e.g., ``t0``, ``u0``, lens mass).
5. Compute derived microlensing quantities (``tE``, ``rho``, ``theta_E``, optional parallax).
6. Attach per-band photometry (optionally using a blending prior) and model-specific parameters.

Authentication
--------------

The helper :func:`~MicroLIA_Sim.param_generation.datalab_login` attempts to reuse a cached token via ``dl.authClient.getToken()`` and falls back to interactive login if needed.

.. warning::

   Interactive login (username/password prompts) is not suitable for non-interactive batch jobs. For cluster use, ensure a valid token is available on disk (or pre-authenticate before submitting). Will make this more flexible in the future!

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
- ``lens_mass_solar`` (Msun)

Additional requirements depend on the configuration:

- If ``enable_parallax=True`` and ``physical_vectors=False``: require ``traj_angle_rad``
- If ``custom_blending=True``: require ``blend_g`` (per-band blend flux ratio prior)
- If ``model_type="USBL"``: require ``q``, ``alpha``, ``origin``, and either:
  - ``semi_major_axis_au`` **or** ``s``
- If ``model_type in {"NFW","BS"}``: require ``t_m``

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

   The lens mass is currently supplied via a prior (``lens_mass_solar``), rather than inferred
   from TRILEGAL stellar parameters -- this may change soon as in principle we could compute masses from the surface gravity which is not currently saved during the query.

Blending convention
-------------------

When ``custom_blending=True``, the table stores:

- ``blend_g``: per-band blend **flux ratio** (drawn from a prior)
- ``source_mags``: **baseline total** magnitudes (source+blend)

The conversion used is:

.. math::

   m_\mathrm{base} = m_\mathrm{source} - 2.5\log_{10}\left(1 + f_\mathrm{blend}\right),

where :math:`f_\mathrm{blend} = F_\mathrm{blend} / F_\mathrm{source}`.

When ``custom_blending=False``, the table stores TRILEGAL magnitudes directly:

- ``source_mags``: TRILEGAL source-only magnitudes
- ``blend_mags``: TRILEGAL lens-only magnitudes

Examples
--------

Minimal PSPL example
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from MicroLIA_Sim.param_generation import (
       datalab_login,
       GenerationConfig,
       default_priors,
       generate_trilegal_event_table,
   )

   # Authenticate
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
   pri["t0"] = Uniform(61000.0, 62000.0) # Could ensure signal with adaptive t0? (within +- of some point)
   pri["u0"] = Uniform(0.0, 0.5)
   pri["lens_mass_solar"] = Uniform(0.001, 100.0)
   pri["q"] = LogUniform(1e-5, 1e-2) # Markus said there is a planet mass ratio function!
   pri["semi_major_axis_au"] = Uniform(0.5, 10.0) # Markus noted that 0.1-10 is most common
   pri["blend_g"] = PerBandUniform(0.0, 0.3) # 0-1 is fine -- 0 for NFW/Boson? 

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
   pri = default_priors(cfg)

   # Manual override
   pri["lens_mass_solar"] = Uniform(0.001, 100.0)

   # Provide a semi-major axis prior (AU) so s_physical can be computed
   pri["semi_major_axis_au"] = Uniform(0.1, 10.0)

   df = generate_trilegal_event_table(
       n_events=20000,
       ra=270.66,
       dec=-35.70,
       cfg=cfg,
       priors=pri,
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

   pri = default_priors(cfg)

   df = generate_trilegal_event_table(
       n_events=1000,
       ra=10.0,
       dec=-10.0,
       cfg=cfg,
       priors=pri,
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
