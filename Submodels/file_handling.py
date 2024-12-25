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
            params_CB, params_duct, DF_design_params = args 
            
            self.centerbody_params: dict = params_CB
            self.duct_params: dict = params_duct
            self.ducted_fan_design_params: dict = DF_design_params
            

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
                                  x: np.ndarray,
                                  design_params: dict,
                                  LE_coordinates: tuple[float, float] = (0, 0),
                                  ) -> tuple[np.ndarray[float], np.ndarray[float]]:
            """
            Compute the profile coordinates of an airfoil based on given design parameters and leading edge coordinates.
            Parameters:
            -----------
            x : np.ndarray
                Array of x-coordinates for the airfoil parameterization.
            design_params : dict
                Dictionary containing design parameters, including "Chord Length".
            LE_coordinates : tuple[float, float], optional
                Tuple containing the leading edge coordinates (dX, dY) to shift the profile coordinates within the domain.
                Default is (0, 0).
            Returns:
            --------
            tuple[np.ndarray[float], np.ndarray[float]]
                A tuple containing two numpy arrays: the x-coordinates and y-coordinates of the airfoil profile. 
            """
            
            # Calculate coordinates of profile based on parameterization 
            upper_x, upper_y, lower_x, lower_y = AirfoilParameterization.ComputeProfileCoordinates(x)
            
            # Multiply with chord length to get correct profile dimensions
            upper_x = upper_x * design_params["Chord Length"]
            lower_x = upper_x * design_params["Chord Length"]
            upper_y = upper_y * design_params["Chord Length"]
            lower_y = lower_y * design_params["Chord Length"]

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


if __name__ == "__main__":
    pass