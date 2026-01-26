.. MicroLIA-Simulate documentation master file, created by
   sphinx-quickstart on Sun Jan 25 20:20:20 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MicroLIA-Simulate's documentation!
=============================================

.. admonition:: Documentation status (last updated |today|)
   :class: note

   This documentation is actively being developed!

**MicroLIA-Simulate** is the dedicated simulation API for the **MicroLIA** framework. It serves as a standalone engine for generating synthetic astronomical datasets, designed as a lightweight sister package to the main analysis tools.

Quick Start
===========

Installation
------------

MicroLIA-Simulate requires **Python 3.10+**. Install the latest stable release via pip:

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
   :maxdepth: 2
   :caption: User Guide

   source/Microlensing Simulations
   source/Parameter Generation

API Reference
=============

Complete documentation of the modules and functions.

.. toctree::
   :maxdepth: 2
   :caption: Python API

   source/MicroLIA_Sim
