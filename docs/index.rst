.. MicroLIA-Simulate documentation master file, created by
   sphinx-quickstart on Sun Jan 25 20:20:20 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MicroLIA-Simulate's documentation!
=============================================

.. admonition:: Documentation status (last updated |today|)
   :class: note

   This documentation is actively being developed!

MicroLIA-Simulate is the dedicated simulation API for the MicroLIA framework, serving as a standalone engine for generating synthetic lightcurves for astronomical datasets.

Quick Start
===========

Installation
------------

MicroLIA-Simulate requires **Python 3.12+**. Install the latest stable release via pip:

.. code-block:: bash

    pip install MicroLIA-Simulate

Alternatively, install the development version from GitHub:

.. code-block:: bash

    git clone https://github.com/Professor-G/MicroLIA-Simulate.git
    cd MicroLIA-Simulate
    pip install -e .

User Guide
==========

Detailed guides on how to generate parameters and simulate light curves.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   source/Parameter Generation
   source/Microlensing Simulations
   source/Variable Stars Simulations

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   source/MicroLIA_Sim
