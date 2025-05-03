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
from scipy import interpolate

# Add the parent and submodels paths to the system path if they are not already in the path
parent_path = str(Path(__file__).resolve().parent.parent)
submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

if parent_path not in sys.path:
    sys.path.append(parent_path)

if submodels_path not in sys.path:
    sys.path.append(submodels_path)

import config
from Submodels.Parameterizations import AirfoilParameterization

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

        # Initialize the AirfoilParameterization class for slightly better memory usage
        self.Parameterization = AirfoilParameterization()

    
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


    def ComputeDuctRadialLocation(self,
                                  duct_variables: dict,
                                  blade_blading_parameters: list[dict]) -> tuple[dict, list]:
        """
        Compute the y-coordinate of the LE of the duct based on the design variables. 

        Parameters
        ----------
        - duct_variables : dict
            A dictionary containing the duct variables.
        - blade_blading_parameters : list[dict]
            The blading parameters for the turbomachinery stage(s).

        Returns
        -------
        - tuple:
            - duct_variables : dict
                The updated duct variables dictionary containing the updated LE y coordinate.
            - blade_blading_parameters : list[dict]
                THe updated blading parameters containing the updated radii of the stator stage(s).
        """

        # Initialize data array for the radial duct coordinates
        radial_duct_coordinates = np.zeros(self.num_stages)

        # Compute the duct x,y coordinates. Note that we are only interested in the lower surface.
        _, _, lower_x, lower_y = self.Parameterization.ComputeProfileCoordinates([duct_variables["b_0"],
                                                                                  duct_variables["b_2"],
                                                                                  duct_variables["b_8"],
                                                                                  duct_variables["b_15"],
                                                                                  duct_variables["b_17"]],
                                                                                  duct_variables)
        lower_x = lower_x * duct_variables["Chord Length"]
        lower_y = lower_y * duct_variables["Chord Length"]

        # Shift the duct x coordinate to the correct location in space
        lower_x += duct_variables["Leading Edge Coordinates"][0]

        # Construct cubic spline interpolant of the duct surface
        duct_interpolant = interpolate.CubicSpline(lower_x,
                                                   np.abs(lower_y),  # Take absolute value of y-coordinates since we need the distance, not the actual coordinate
                                                   extrapolate=False) 

        rot_flags = config.ROTATING
        x_min, x_max = lower_x[0], lower_x[-1]
        tip_gap = config.tipGap
        for i in range(self.num_stages):
            if not rot_flags[i]:
                continue
            
            # Extract blading and blade radius
            blading = blade_blading_parameters[i]
            y_tip = blading["radial_stations"][-1]
            print(y_tip)

            # Compute the LE and TE x-coordinates of the tip section
            sweep = np.tan(blading["sweep_angle"][-1])
            x_tip_LE = blading["root_LE_coordinate"] + sweep * y_tip
            projected_chord = blading["chord_length"][-1] * np.cos(np.pi/2 - 
                                                                   (blading["blade_angle"][-1] + blading["ref_blade_angle"] - blading["reference_section_blade_angle"]))
            x_tip_TE = x_tip_LE + projected_chord

            # Compute the offsets for the LE and TE of the blade tip
            LE_offset = float(duct_interpolant(x_tip_LE)) if x_min <= x_tip_LE <= x_max else 0  # Set to 0 if duct does not lie above LE
            TE_offset = float(duct_interpolant(x_tip_TE)) if x_min <= x_tip_TE <= x_max else 0  # Set to 0 if duct does not lie above TE

            # Compute the radial location of the duct
            radial_duct_coordinates[i] = y_tip + tip_gap + max(LE_offset, TE_offset)

        # The LE y coordinate of the duct is then the maximum of the computed coordinates to enforce the minimum tip gap everywhere
        LE_coordinate_duct = np.max(radial_duct_coordinates)

        # Update the duct variables
        duct_variables["Leading Edge Coordinates"] = (duct_variables["Leading Edge Coordinates"][0],
                                                      LE_coordinate_duct)
        
        # Set the radius of all stators equal to this y coordinate to avoid miss-matches in stator sizes. 
        for i in range(self.num_stages):
            if not rot_flags[i]:
                r_old = np.max(blade_blading_parameters[i]["radial_stations"])
                blade_blading_parameters[i]["radial_stations"] = blade_blading_parameters[i]["radial_stations"] / r_old * LE_coordinate_duct    
    
        # Return the updated data
        return duct_variables, blade_blading_parameters
    
    
    def DeconstructDesignVector(self) -> tuple:
        """
        Decompose the design vector x into dictionaries of all the design variables to match the expected input formats for 
        the MTFLOW code interface. 

        Returns
        -------
        tuple
            A tuple containing the decomposed design variables:
            - centerbody_variables: dict
            - duct_variables: dict
            - blade_design_parameters: list
            - blade_blading_parameters: list
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
        for i in range(self.num_stages):
            # Initiate empty list for each stage
            stage_blading_parameters = {}
            if self.optimize_stages[i]:
                # If the stage is to be optimized, read in the design vector for the blading parameters
                stage_blading_parameters["root_LE_coordinate"] = vget(idx)
                stage_blading_parameters["ref_blade_angle"] = vget(idx, 1)
                stage_blading_parameters["reference_section_blade_angle"] = config.REFERENCE_BLADE_ANGLES[i]
                stage_blading_parameters["blade_count"] = int(round(vget(idx, 2)))
                stage_blading_parameters["radial_stations"] = np.linspace(0, 0.5 * vget(idx, 3), self.num_radial[i])  # Radial stations are defined as fraction of blade radius * local radius

                # Initialize sectional blading parameter lists
                stage_blading_parameters["chord_length"] = [None] * self.num_radial[i]
                stage_blading_parameters["sweep_angle"] = [None] * self.num_radial[i]
                stage_blading_parameters["blade_angle"] = [None] * self.num_radial[i]

                base_idx = idx + 4
                for j in range(self.num_radial[i]):
                    # Loop over the number of radial sections and write their data to the corresponding lists
                    stage_blading_parameters["chord_length"][j]= vget(base_idx, j)
                base_idx += self.num_radial[i]
                for j in range(self.num_radial[i]):
                    stage_blading_parameters["sweep_angle"][j] = vget(base_idx, j)
                base_idx += self.num_radial[i]
                for j in range(self.num_radial[i]):    
                    stage_blading_parameters["blade_angle"][j] = vget(base_idx, j)
                base_idx += self.num_radial[i] 

                # Update the index to correctly point to the next stage
                idx = base_idx              
            else:
                stage_blading_parameters = config.STAGE_BLADING_PARAMETERS[i]
            
            # Append the stage blading parameters to the main list
            blade_blading_parameters.append(stage_blading_parameters)

        # Compute the updated duct and blading parameters
        print(blade_blading_parameters)
        duct_variables, blade_blading_parameters = self.ComputeDuctRadialLocation(duct_variables=duct_variables,
                                                                                  blade_blading_parameters=blade_blading_parameters)
        print(blade_blading_parameters)
        print("=========================================================")
        # Write the reference length for MTFLOW
        Lref = blade_blading_parameters[0]["radial_stations"][-1]

        return centerbody_variables, duct_variables, blade_design_parameters, blade_blading_parameters, Lref