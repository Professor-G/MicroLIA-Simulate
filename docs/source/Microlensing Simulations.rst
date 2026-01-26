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
   plot_lightcurves_mag(lcs, title="Perfect pyLIMA PSPL")



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





