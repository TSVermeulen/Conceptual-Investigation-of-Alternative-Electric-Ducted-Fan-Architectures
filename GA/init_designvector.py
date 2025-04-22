"""
design_vector_init
==================

Description
-----------
This module defines the DesignVector class, which is used to construct the design vector for the pymoo framework 
optimization problem. The design vector supports mixed-variable optimization, including real and integer variables.

Classes
-------
DesignVector
    Class for constructing the design vector based on configuration toggles.

Examples
--------
>>> from init_designvector import DesignVector
>>> from config import cfg
>>> dv = DesignVector()
>>> design_vector = dv._construct_vector(cfg)

Notes
-----
This module integrates with the pymoo framework for optimization. Ensure that the configuration module (cfg) is 
properly set up with the required toggles and parameters for the design vector construction.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial implementation. Extracted from the problem_definition.py file for better modularity and readability.
"""

import numpy as np
from pymoo.core.variable import Real, Integer
from types import ModuleType

class DesignVector():
    """
    This class is used to construct the design vector for the optimisation problem.
    """


    def __init__(self) -> None:
        """
        Initialisation for the DesignVector class.
        """


    def _construct_vector(self,
                          cfg: ModuleType) -> dict:
        """
        Initialize the pymoo design vector based on the toggles in config.

        Parameters
        ----------
        - cfg : ModuleType
            The config module containing the design vector configuration.
        
        Returns
        -------
        - dict
            A dictionary containing the design vector variables and their bounds.
        """

        # Initialize variable list with variable types.
        # This is required to handle the mixed-variable nature of the optimisation, where the blade count is an integer
        vars = []
        if cfg.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, initialise the variable types
            vars.append(Real(bounds=(0, 1)))  # mapping variable for b_8
            vars.append(Real(bounds=(0, 1)))  # b_15
            vars.append(Real(bounds=(0.1, 1)))  # x_t
            vars.append(Real(bounds=(0, 0.25)))  # y_t
            vars.append(Real(bounds=(0, 0.05)))  # dz_TE
            vars.append(Real(bounds=(-0.1, 0)))  # r_LE
            vars.append(Real(bounds=(0, 0.5)))  # trailing_wedge_angle
            vars.append(Real(bounds=(0.25, 4)))  # Chord Length
        if cfg.OPTIMIZE_DUCT:
            # If the duct is to be optimised, intialise the variable types
            vars.append(Real(bounds=(0, 1)))  # b_0
            vars.append(Real(bounds=(0, 0.5)))  # b_2
            vars.append(Real(bounds=(0, 1)))  # mapping variable for b_8
            vars.append(Real(bounds=(0, 1)))  # b_15
            vars.append(Real(bounds=(0, 1)))  # b_17
            vars.append(Real(bounds=(0.1, 1)))  # x_t
            vars.append(Real(bounds=(0, 0.25)))  # y_t
            vars.append(Real(bounds=(0.05, 1)))  # x_c
            vars.append(Real(bounds=(0, 0.1)))  # y_c
            vars.append(Real(bounds=(0, 0.2)))  # z_TE
            vars.append(Real(bounds=(0, 0.05)))  # dz_TE
            vars.append(Real(bounds=(-0.1, 0)))  # r_LE
            vars.append(Real(bounds=(0, 0.5)))  # trailing_wedge_angle
            vars.append(Real(bounds=(0, 0.5)))  # trailing_camberline_angle
            vars.append(Real(bounds=(0, 0.5)))  # leading_edge_direction
            vars.append(Real(bounds=(0.25, 2.5)))  # Chord Length
            vars.append(Real(bounds=(-0.5, 0.5)))  # Leading Edge X-Coordinate

        for i in range(cfg.NUM_STAGES):
            # If (any of) the rotor/stator stage(s) are to be optimised, initialise the variable types
            if cfg.OPTIMIZE_STAGE[i]:
                for _ in range(cfg.NUM_RADIALSECTIONS):
                    vars.append(Real(bounds=(0, 1)))  # b_0
                    vars.append(Real(bounds=(0, 0.5)))  # b_2
                    vars.append(Real(bounds=(0, 1)))  # mapping variable for b_8
                    vars.append(Real(bounds=(0, 1)))  # b_15
                    vars.append(Real(bounds=(0, 1)))  # b_17
                    vars.append(Real(bounds=(0.1, 1)))  # x_t
                    vars.append(Real(bounds=(0, 0.25)))  # y_t
                    vars.append(Real(bounds=(0.05, 1)))  # x_c
                    vars.append(Real(bounds=(0, 0.1)))  # y_c
                    vars.append(Real(bounds=(0, 0.2)))  # z_TE
                    vars.append(Real(bounds=(0, 0.05)))  # dz_TE
                    vars.append(Real(bounds=(-0.1, 0)))  # r_LE
                    vars.append(Real(bounds=(0, 0.5)))  # trailing_wedge_angle
                    vars.append(Real(bounds=(0, 0.5)))  # trailing_camberline_angle
                    vars.append(Real(bounds=(0, 0.5)))  # leading_edge_direction

        for i in range(cfg.NUM_STAGES):
            if cfg.OPTIMIZE_STAGE[i]:
                vars.append(Real(bounds=(0.1, 0.4)))  # root_LE_coordinate
                vars.append(Integer(bounds=(3, 20)))  # blade_count
                vars.append(Real(bounds=(-np.pi/4, np.pi/4)))  # ref_blade_angle
                vars.append(Real(bounds=(0, 1.5)))  # blade radius

                for _ in range(cfg.NUM_RADIALSECTIONS): 
                    vars.append(Real(bounds=(0.05, 0.5)))  # chord length
                for _ in range(cfg.NUM_RADIALSECTIONS): 
                    vars.append(Real(bounds=(0, np.pi/3)))  # sweep_angle
                for _ in range(cfg.NUM_RADIALSECTIONS): 
                    vars.append(Real(bounds=(-np.pi/4, np.pi/4)))  # blade_angle

        # For a mixed-variable problem, PyMoo expects the vars to be a dictionary, so we convert vars to a dictionary.
        # Note that all variables are given a name xi.
        vars = {f"x{i}": var for i, var in enumerate(vars)}

        return vars