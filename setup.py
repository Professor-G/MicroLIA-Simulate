# -*- coding: utf-8 -*-
"""
Created on Fri June 6 3 16:40:44 2022

@author: danielgodinez
"""
from setuptools import setup, find_packages

setup(
    name="MicroLIA-Simulate",
    version="0.1.0",
    description="A standalone simulation engine for the MicroLIA framework, designed as a lightweight sister package for generating synthetic astronomical datasets.",
    author="Daniel Godines",
    author_email="dan@gmail.com",
    url="https://github.com/Professor-G/MicroLIA-Simulate",
    license='GPL-3.0',    
    classifiers=[
        'Development Status :: 4 - Beta',  # will change to '5 - Stable' after Meet's review
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
    packages=find_packages(), 
    install_requires=[
        'numpy',
        'pandas',
        'scipy', # Used for interpolation and optimization (for the root finding)
        'astropy',# Used for units and astro constants
        'matplotlib',
        'pyLIMA==1.9.8', # The core microlensing simulation engine
        'astro-datalab>=2.22.1', # To query TRILEGAL 
    ],   
    python_requires='>=3.12',
    include_package_data=True,
)