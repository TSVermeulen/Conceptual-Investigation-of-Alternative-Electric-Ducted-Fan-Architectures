"""
file_handling
============

Description
-----------
A brief description of what this module does.

Functions
---------
function1(param1, param2)
    Description of function1.

function2(param1, param2)
    Description of function2.

Classes
-------
ClassName
    Description of ClassName.

Examples
--------
Examples should be written in doctest format, and should illustrate how to
use the function or class.

>>> example_function(1, 2)
3

Notes
-----
Any additional notes about the module.

References
----------
Any references used in the module.

"""

import numpy as np
import os
from Parameterizations import AirfoilParameterization

class fileHandling:
    """
    
    """

    def __init__(self, *args,) -> None:
        """
        Initialize the fileHandling class.
        
        This method sets up the initial state of the class.
        The fileHandling class is, in effect, a grouping class, 
        and therefore does not take any inputs.

        Returns
        -------
        None
        """
        

    class fileHandlingMTSET:
        """
        
        """

        def __init__(self, *args,) -> None:
            """
            Initialize the fileHandlingMTSET class.
        
            This method sets up the initial state of the class.

            Returns
            -------
            None

            """

            # Extract center body and duct parameterization parameters and Ducted fan design parameters
            # Write them to self
            # All inputs must be dictionaries
            params_CB, params_duct, ducted_fan_design_params, case_name = args 
            
            self.centerbody_params: dict = params_CB
            self.ducted_fan_design_params: dict = ducted_fan_design_params
            self.duct_params: dict = params_duct
            self.case_name: str = case_name
            

        def GetGridSize(self):
            """
            Determine grid size - x & y coordinates of grid boundary. 
            Ideally find some way to automatically determine suitable values for YTOP and XFRONT/REAR based on operating condition

            This would imply some dependency on the variation of (inviscid) flow variables

            Can use the output from an inviscid MTSOL run at cruising conditions to determine the grid boundaries 
            (variation in Mach at outermost streamline < 1E-3)

            Use a default initial grid of:
            (Note this is in reference to the outer edges of the ducted fan itself)

            Y_top = CONST_1 * max diameter (2.5?)
            X_FRONT = CONST_2 * MAX LENGTH (1?)
            X_AFT = CONST_3 * MAX LENGTH (3?)
            """

            pass


        def GetProfileCoordinates(self,
                                  x: dict,
                                  LE_coordinates: tuple[float, float] = (0, 0),
                                  ) -> tuple[np.ndarray[float], np.ndarray[float]]:
            """
            Compute the profile coordinates of an airfoil based on given design parameters and leading edge coordinates.
            Parameters:
            -----------
            x : dict
                Dictionary of design parameters for the profile parameterization.
            LE_coordinates : tuple[float, float], optional
                Tuple containing the leading edge coordinates (dX, dY) to shift the profile coordinates within the domain.
                Default is (0, 0), i.e. no offset
            Returns:
            --------
            tuple[np.ndarray[float], np.ndarray[float]]
                A tuple containing two numpy arrays: the x-coordinates and y-coordinates of the airfoil profile. 
            """
            
            # Calculate coordinates of profile based on parameterization
            parameterization = np.array([x["b_8"], 
                                         x["b_15"], 
                                         x["b_0"], 
                                         x["b_2"], 
                                         x["b_17"], 
                                         x["r_LE"], 
                                         x["trailing_wedge_angle"], 
                                         x["trailing_camberline_angle"], 
                                         x["leading_edge_direction"],
                                        ])
            print(parameterization)
            upper_x, upper_y, lower_x, lower_y = AirfoilParameterization.ComputeProfileCoordinates(parameterization)
            
            # Multiply with chord length to get correct profile dimensions
            upper_x = upper_x * x["Chord Length"]
            lower_x = upper_x * x["Chord Length"]
            upper_y = upper_y * x["Chord Length"]
            lower_y = lower_y * x["Chord Length"]

            # Offset airfoil using the (dX, dY) input to shift the profile coordinates to the appropriate 
            # location within the domain 
            upper_x += LE_coordinates[0]
            lower_x += LE_coordinates[0]
            upper_y += LE_coordinates[1]
            lower_y += LE_coordinates[1]

            # Construct overall profile coordinates data, 
            # in-line with required format of MTFLOW            
            # Check if arrays are sorted
            upper_x_sorted = np.all(np.diff(upper_x) >= 0)
            lower_x_sorted = np.all(np.diff(lower_x) >= 0)

            # Determine x and y based on the sorting checks
            if upper_x_sorted and lower_x_sorted:
                x = np.concatenate((np.flip(upper_x), lower_x), axis=0)
                y = np.concatenate((np.flip(upper_y), lower_y), axis=0)
            elif upper_x_sorted and not lower_x_sorted:
                x = np.concatenate((np.flip(upper_x), np.flip(lower_x)), axis=0)
                y = np.concatenate((np.flip(upper_y), np.flip(lower_y)), axis=0)
            elif not upper_x_sorted and lower_x_sorted:
                x = np.concatenate((upper_x, lower_x), axis=0)
                y = np.concatenate((upper_y, lower_y), axis=0)
            else:
                x = np.concatenate((upper_x, np.flip(lower_x)), axis=0)
                y = np.concatenate((upper_y, np.flip(lower_y)), axis=0)

            return np.vstack(x, y).T
        
        
        def GenerateMTSETInput(self,
                               domain_boundaries: np.ndarray[float]) -> None:
            """
            
            """

            # Get profiles of centerbody and duct
            xy_centerbody = self.GetProfileCoordinates(self.centerbody_params)
            xy_duct = self.GetProfileCoordinates(self.duct_params,
                                                 self.ducted_fan_design_params["Duct Leading Edge Coordinates"])
            
            # Generate walls.xxx input data structure
            walls = np.array([])
            walls = np.append(walls, [f"{self.case_name}"])  # First line of the input file contains the case name
            walls = np.append(walls, [domain_boundaries])  # Second line contains the domain boundaries [XINL XOUT YBOT YTOP]
            walls = np.append(walls, [xy_centerbody])  # Third item contains the centerbody profile coordinates
            walls = np.append(walls, [999., 999.])  # Elements are separated by a line containing 999. 999.
            walls = np.append(walls, [xy_duct])  # Fifth item contains the duct profile coordinates

            print(walls)


            return


if __name__ == "__main__":

    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 1.0}
    n24112_coeff = {"b_0": -0.00010682822799885562, "b_2": 0.04399164709308923, "b_8": 0.04206606421133544, "b_15": 0.749999789203726, "b_17": 0.6999996858370111, "r_LE": -0.01732927766979101, "trailing_wedge_angle": 0.1384158688629374, "trailing_camberline_angle": -0.002415556195819632, "leading_edge_direction": 0.8454188491190011, "Chord Length": 1.0}
    design_params = {"Duct Leading Edge Coordinates": (0, 2)}


    call_class = fileHandling.fileHandlingMTSET(n24112_coeff, n2415_coeff, design_params, "test_case")
    call_class.GenerateMTSETInput([0, 1, 0, 1])
    pass