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

# Import standard libraries
import numpy as np
from types import ModuleType

# Import 3rd party libraries
from pymoo.core.variable import Real, Integer

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

        # Define helper function with the default 15-parameter profile definition values to keep the method cleaned up
        def profile_section_vars() -> list:
            """ Return the standard 15-var profile section definition """
            return [Real(bounds=(0, 1)),  # b_0
                    Real(bounds=(0, 0.5)),  # b_2
                    Real(bounds=(0.05, 1)),  # mapping variable for b_8
                    Real(bounds=(0, 1)),  # b_15
                    Real(bounds=(0, 1)),  # b_17
                    Real(bounds=(0.1, 0.9)),  # x_t
                    Real(bounds=(0.0125, 0.25)),  # y_t
                    Real(bounds=(0.05, 1)),  # x_c
                    Real(bounds=(0, 0.1)),  # y_c
                    Real(bounds=(0, 0.2)),  # z_TE
                    Real(bounds=(0, 0.02)),  # dz_TE
                    Real(bounds=(-0.1, -0.001)),  # r_LE
                    Real(bounds=(0.01, np.pi/3)),  # trailing_wedge_angle
                    Real(bounds=(0.01, np.pi/3)),  # trailing_camberline_angle
                    Real(bounds=(0.01, np.pi/3))]  # leading_edge_direction

        # Initialize variable list with variable types.
        # This is required to handle the mixed-variable nature of the optimisation, where the blade count is an integer
        vector = []
        if cfg.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, initialise the variable types
            complete_profile = profile_section_vars()
            centerbody_var_indices = [2, 3, 5, 6, 10, 11, 12]
            profile = [complete_profile[i] for i in centerbody_var_indices]

            vector.extend(profile)
            vector.append(Real(bounds=(0.25, 4)))  # Chord Length
        if cfg.OPTIMIZE_DUCT:
            # If the duct is to be optimised, intialise the variable types
            vector.extend(profile_section_vars())
            vector.append(Real(bounds=(0.25, 2.5)))  # Chord Length
            vector.append(Real(bounds=(-0.5, 0.5)))  # Leading Edge X-Coordinate

        for i in range(cfg.NUM_STAGES):
            # If (any of) the rotor/stator stage(s) are to be optimised, initialise the variable types
            if cfg.OPTIMIZE_STAGE[i]:
                for _ in range(cfg.NUM_RADIALSECTIONS[i]):
                    vector.extend(profile_section_vars())                  

        for i in range(cfg.NUM_STAGES):
            if cfg.OPTIMIZE_STAGE[i]:
                vector.append(Real(bounds=(0., 0.4)))  # root_LE_coordinate
                vector.append(Real(bounds=(-np.pi/4, np.pi/4)))  # ref_blade_angle
                vector.append(Integer(bounds=(2, 20)))  # blade_count
                if cfg.ROTATING[i]:
                    vector.append(Real(bounds=(20, 42)))  # blade RPS
                vector.append(Real(bounds=(1.0, 3.0)))  # blade diameter

                for _ in range(cfg.NUM_RADIALSECTIONS[i]): 
                    vector.append(Real(bounds=(0.1, 0.75)))  # chord length
                for _ in range(cfg.NUM_RADIALSECTIONS[i]): 
                    vector.append(Real(bounds=(0, np.pi/3)))  # sweep_angle
                for _ in range(cfg.NUM_RADIALSECTIONS[i]): 
                    vector.append(Real(bounds=(0, np.pi/3)))  # blade_angle

        # For a mixed-variable problem, PyMoo expects the vector to be a dictionary, so we convert vector to a dictionary.
        # Note that all variables are given a name xi.
        vector = {f"x{i}": var for i, var in enumerate(vector)}

        return vector
    

if __name__ == "__main__":
    import config
    # config.OPTIMIZE_DUCT = False
    test = DesignVector()
    vector = test._construct_vector(config)
    print(vector)