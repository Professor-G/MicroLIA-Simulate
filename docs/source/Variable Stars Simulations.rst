.. _Variable_Stars_Simulations:

Variable Stars Simulations
==========================

Overview
--------

This module simulates multiband variable-star light curves in ``g/r/i`` by sampling empirical templates, phase-folding the input observation times, and
interpolating the template in each band. A single magnitude offset is applied to all bands so that the simulated light curves preserve the template's intrinsic
colors while matching a desired mean magnitude in a chosen reference band.

The RR Lyrae template set used here is from `BaezaVillagra et al 2025 <https://www.aanda.org/articles/aa/full_html/2025/02/aa53129-24/aa53129-24.html>`_.

Quick start
-----------

The main function is ``simulate_rrlyae()``. It returns a dict mapping each band to simulated magnitudes evaluated at the provided observation times.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from MicroLIA_Sim.variable_stars import simulate_rrlyae

   rng = np.random.default_rng(1909)

   # Example observation times (days; e.g., MJD)
   t_g = 7000.0 + np.sort(rng.uniform(0, 365, size=200))
   t_r = 7000.0 + np.sort(rng.uniform(0, 365, size=200))
   t_i = 7000.0 + np.sort(rng.uniform(0, 365, size=200))

   times = {"g": t_g, "r": t_r, "i": t_i}

   mags = simulate_rrlyae(
       times=times,
       bailey=1, # 1=RRab, 2=RRc, 3=Cepheid-like
       period=None, # days; if set, bailey is ignored
       reference_band="i",
       reference_mean_mag=18.0,
       rng=rng, # controls random phase zero-point (T0) and template
   )

   plt.plot(t_g, mags["g"], "o", label="g")
   plt.plot(t_r, mags["r"], "o", label="r")
   plt.plot(t_i, mags["i"], "o", label="i")
   plt.gca().invert_yaxis()
   plt.legend()
   plt.show()

Bailey-type priors
^^^^^^^^^^^^^^^^^^

If ``period`` is not provided:

* ``bailey == 1`` (RRab): :math:`P \sim \mathcal{N}(0.6, 0.15)` days
* ``bailey == 2`` (RRc): :math:`P \sim \mathcal{N}(0.33, 0.10)` days
* ``bailey == 3`` (Cepheid-like): :math:`P = 10^{X}`, where :math:`X \sim \mathrm{LogNormal}(0.0, 0.2)`

Reproducibility
^^^^^^^^^^^^^^^

There are two random elements, the template selection and the phase zero-point. Passing a seeded ``rng`` makes these reproducible.

.. note::

   In the current implementation, ``bailey == 3`` uses the RRc templates and is intended as an approximation.
