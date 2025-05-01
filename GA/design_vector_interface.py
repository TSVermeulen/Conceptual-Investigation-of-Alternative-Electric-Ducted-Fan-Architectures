"""
design_vector_interface
=======================

Description
-----------
This module provides utilities for efficient access and manipulation of design vectors used in optimization problems.

Classes
-------
DesignVectorAccessor
    Class to provide efficient access to design vector elements without repeated string formatting.

Examples
--------
>>> x_dict = {"var1": 10, "var2": 20, "var3": 30}
>>> x_keys = ["var1", "var2", "var3"]
>>> accessor = DesignVectorAccessor(x_dict, x_keys)
>>> accessor.get(1)  # Returns 20

Notes
-----
This module is designed to streamline the handling of design vectors in optimization workflows. It ensures efficient
access to elements and provides error handling for missing keys or indices.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial implementation.
"""

import sys
import numpy as np
from pathlib import Path

import config

# Add the parent and submodels paths to the system path if they are not already in the path
parent_path = str(Path(__file__).resolve().parent.parent)
submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

if parent_path not in sys.path:
    sys.path.append(parent_path)

if submodels_path not in sys.path:
    sys.path.append(submodels_path)

class DesignVectorInterface:
    """ 
    Simple class to provide efficient access to design vector elements without repeated string formatting, 
    and to deconstruct the design vector dictionary x into the expected "sub"-dictionaries required to use the MTFLOW interface """
    
    def __init__(self,
                 x_dict: dict[str, float|int]) -> None:
        """
        Initialization of the DesignVectorInterface class

        Parameters
        ----------
        - x_dict : dict[str, float|int]
            The Pymoo design vector dictionary

        Returns
        -------
        None
        """

        self.x_dict = x_dict
        self.x_keys = list(x_dict.keys())

        # Import control variables
        self.num_radial = config.NUM_RADIALSECTIONS
        self.num_stages = config.NUM_STAGES
        self.optimize_stages = config.OPTIMIZE_STAGE

    
    def GetValueFromVector(self,
            base_idx: int,
            offset: int = 0,
            default=None) -> float|int:
        """ 
        Simple function to extract the variable at base_idx + offset in the pymoo design vector dictionary.

        Parameters
        ----------
        - base_idx : int
            The base index of the start of the "section" being extracted
        - offset : int, optional
            An optional offset from the base index. Defaults to zero if no value is provided.
        - default : optional
            An optional value to use as default in case the value cannot be extracted from the dictionary

        Returns
        -------
        - float | int
            The value extracted from the design vector dictionary
        """

        try:
            key = self.x_keys[base_idx + offset]
            return self.x_dict[key]
        except (IndexError, KeyError) as err:
            if default is not None:
                return default
            raise KeyError(f"Design vector key at position {base_idx + offset} missing") from err
        
    
    def DeconstructDesignVector(self) -> tuple:
        """
        Decompose the design vector x into dictionaries of all the design variables to match the expected input formats for 
        the MTFLOW code interface. 
        The design vector has the standard format: [centerbody, duct, blades]

        Returns
        -------
        tuple
            A tuple containing the decomposed design variables:
            - centerbody_variables: dict
            - duct_variables: dict
            - blade_design_parameters: list
            - blade_blading_parameters: list
            - blade_diameters: list
            - Lref: float
        """

        # Create a local alias for self.GetValueFromVector - this is marginally quicker per call
        vget = self.GetValueFromVector  

        # Define a helper function to compute parameter b_8 using the mapping design variable
        def Getb8(b_8_map: float, 
                  r_le: float, 
                  x_t: float, 
                  y_t: float) -> float:
            """
            Helper function to compute the bezier parameter b_8 using the mapping parameter 0 <= b_8_map <= 1
            """

            term = -2 * r_le * x_t / 3
            sqrt_term = 0 if term <= 0 else np.sqrt(term)

            return b_8_map * min(y_t, sqrt_term)

        # Define a pointer to count the number of variable parameters
        idx = 0
        centerbody_designvar_count = 8
        duct_designvar_count = 17
        if config.OPTIMIZE_CENTERBODY:
            centerbody_start = 0
            duct_start = centerbody_designvar_count
            stage_start = centerbody_designvar_count + (duct_designvar_count if config.OPTIMIZE_DUCT else 0)
        else:
            duct_start = 0
            stage_start = duct_designvar_count if config.OPTIMIZE_DUCT else 0

        # Deconstruct the centerbody values if it's variable.
        # If the centerbody is constant, read in the centerbody values from config.
        # Note that if the centerbody is variable, we keep the LE coordinate fixed, as the LE coordinate of the duct would already be free to move. 
        if config.OPTIMIZE_CENTERBODY:
            idx = centerbody_start
            centerbody_variables = {"b_0": 0,
                                    "b_2": 0, 
                                    "b_8": Getb8(vget(idx), vget(idx, 5), vget(idx, 2), vget(idx, 3)),
                                    "b_15": vget(idx, 1),
                                    "b_17": 0,
                                    "x_t": vget(idx, 2),
                                    "y_t": vget(idx, 3),
                                    "x_c": 0,
                                    "y_c": 0,
                                    "z_TE": 0,
                                    "dz_TE": vget(idx, 4),
                                    "r_LE": vget(idx, 5),
                                    "trailing_wedge_angle": vget(idx, 6),
                                    "trailing_camberline_angle": 0,
                                    "leading_edge_direction": 0, 
                                    "Chord Length": vget(idx, 7),
                                    "Leading Edge Coordinates": (0, 0)}
        else:
            centerbody_variables = config.CENTERBODY_VALUES

        # Deconstruct the duct values if it's variable.
        # If the duct is constant, read in the duct values from config.
        if config.OPTIMIZE_DUCT:
            idx = duct_start
            duct_variables = {"b_0": vget(idx),
                              "b_2": vget(idx, 1), 
                              "b_8": Getb8(vget(idx, 2), vget(idx, 11), vget(idx, 5), vget(idx, 6)),
                              "b_15": vget(idx, 3),
                              "b_17": vget(idx, 4),
                              "x_t": vget(idx, 5),
                              "y_t": vget(idx, 6),
                              "x_c": vget(idx, 7),
                              "y_c": vget(idx, 8),
                              "z_TE": vget(idx, 9),
                              "dz_TE": vget(idx, 10),
                              "r_LE": vget(idx, 11),
                              "trailing_wedge_angle": vget(idx, 12),
                              "trailing_camberline_angle": vget(idx, 13),
                              "leading_edge_direction": vget(idx, 14), 
                              "Chord Length": vget(idx, 15),
                              "Leading Edge Coordinates": (vget(idx, 16), 0)}
        else:
            duct_variables = config.DUCT_VALUES
                
        # Deconstruct the rotorblade parameters if they are variable.
        # If the rotorblade parameters are constant, read in the parameters from config.
        blade_design_parameters = []
        idx = stage_start
        for i in range(self.num_stages):
            # Initiate empty list for each stage
            stage_design_parameters = []
            if self.optimize_stages[i]:
                # If the stage is to be optimized, read in the design vector for the blade profiles
                for _ in range(self.num_radial[i]):
                    # Loop over the number of radial sections and append each section to stage_design_parameters
                    section_parameters = {"b_0": vget(idx),
                                        "b_2": vget(idx, 1), 
                                        "b_8": Getb8(vget(idx, 2), vget(idx, 11), vget(idx, 5), vget(idx, 6)), 
                                        "b_15": vget(idx, 3),
                                        "b_17": vget(idx, 4),
                                        "x_t": vget(idx, 5),
                                        "y_t": vget(idx, 6),
                                        "x_c": vget(idx, 7),
                                        "y_c": vget(idx, 8),
                                        "z_TE": vget(idx, 9),
                                        "dz_TE": vget(idx, 10),
                                        "r_LE": vget(idx, 11),
                                        "trailing_wedge_angle": vget(idx, 12),
                                        "trailing_camberline_angle": vget(idx, 13),
                                        "leading_edge_direction": vget(idx, 14)}
                    idx += 15
                    stage_design_parameters.append(section_parameters)
            else:
                # If the stage is meant to be constant, read it in from config. 
                stage_design_parameters = config.STAGE_DESIGN_VARIABLES[i]
            # Write the stage nested list to blade_design_parameters
            blade_design_parameters.append(stage_design_parameters)

        blade_blading_parameters = []
        blade_diameters = []
        for i in range(self.num_stages):
            # Initiate empty list for each stage
            stage_blading_parameters = {}
            if self.optimize_stages[i]:
                # If the stage is to be optimized, read in the design vector for the blading parameters
                stage_blading_parameters["root_LE_coordinate"] = vget(idx)
                stage_blading_parameters["ref_blade_angle"] = vget(idx, 2)
                stage_blading_parameters["reference_section_blade_angle"] = config.REFERENCE_SECTION_ANGLES[i]
                stage_blading_parameters["blade_count"] = int(round(vget(idx, 1)))
                stage_blading_parameters["radial_stations"] = np.linspace(0, 1, self.num_radial[i]) * vget(idx, 3)  # Radial stations are defined as fraction of blade radius * local radius
                blade_diameters.append(vget(idx, 3) * 2)

                # Initialize sectional blading parameter lists
                stage_blading_parameters["chord_length"] = [None] * self.num_radial[i]
                stage_blading_parameters["sweep_angle"] = [None] * self.num_radial[i]
                stage_blading_parameters["blade_angle"] = [None] * self.num_radial[i]

                base_idx = idx + 4
                for j in range(self.num_radial[i]):
                    # Loop over the number of radial sections and write their data to the corresponding lists
                    stage_blading_parameters["chord_length"][j]= vget(base_idx, j)
                    stage_blading_parameters["sweep_angle"][j] = vget(base_idx, self.num_radial[i] + j)
                    stage_blading_parameters["blade_angle"][j] = vget(base_idx, self.num_radial[i] * 2 + j)
                idx = base_idx + 3 * self.num_radial[i]               
            else:
                stage_blading_parameters = config.STAGE_BLADING_PARAMETERS[i]
                blade_diameters.append(config.BLADE_DIAMETERS[i])
            
            # Append the stage blading parameters to the main list
            blade_blading_parameters.append(stage_blading_parameters)
        
        # Write the reference length for MTFLOW
        Lref = blade_diameters[0]

        return centerbody_variables, duct_variables, blade_design_parameters, blade_blading_parameters, blade_diameters, Lref