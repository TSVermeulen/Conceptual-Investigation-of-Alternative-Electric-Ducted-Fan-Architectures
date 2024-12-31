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

            # Calculate Y-domain boundaries based on ducted fan design parameters. 
            # Y_bottom is always 0 as it is the symmetry line
            Y_TOP = 2.5 * self.ducted_fan_design_params["Duct Outer Diameter"]
            Y_BOT = 0

            # Calculate X-domain boundaries based on ducted fan design parameters.
            # X_FRONT is taken as 1m ahead of the leading edge of the duct.
            X_FRONT = self.ducted_fan_design_params["Duct Leading Edge Coordinates"][0] - 1
            X_AFT = self.ducted_fan_design_params["Duct Leading Edge Coordinates"][0] + self.duct_params["Chord Length"] + 3

            return [X_FRONT, X_AFT, Y_BOT, Y_TOP]


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
            
            # Restructure the input dictionary to a numpy array for the airfoil parameterization and a parameterization dictionary
            b_coeff = np.array([x["b_0"], x["b_2"], x["b_8"], x["b_15"], x["b_17"]])

            parameterization = {"x_t": x["x_t"],
                                "y_t": x["y_t"],
                                "x_c": x["x_c"],
                                "y_c": x["y_c"], 
                                "z_TE": x["z_TE"],
                                "dz_TE": x["dz_TE"],
                                "r_LE": x["r_LE"], 
                                "trailing_wedge_angle": x["trailing_wedge_angle"], 
                                "trailing_camberline_angle": x["trailing_camberline_angle"], 
                                "leading_edge_direction": x["leading_edge_direction"],
                                }

            airfoil_class = AirfoilParameterization()
            upper_x, upper_y, lower_x, lower_y = airfoil_class.ComputeProfileCoordinates(b_coeff,
                                                                                         parameterization,
                                                                                         )
            
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
            # Check if arrays are sorted and construct the profile coordinates if so
            upper_x_sorted = np.all((np.diff(upper_x) >= 0)[1:])
            lower_x_sorted = np.all((np.diff(lower_x) >= 0)[1:])

            if not upper_x_sorted:
                raise ValueError(f"Upper x-coordinates are not sorted. This indicates an error in the profile generation!")
            if not lower_x_sorted:
                raise ValueError("Lower x-coordinates are not sorted. This indicates an error in the profile generation!")
            
            x = np.concatenate((np.flip(upper_x), lower_x), axis=0)
            y = np.concatenate((np.flip(upper_y), lower_y), axis=0)

            return np.vstack((x, y)).T
        
        
        def GenerateMTSETInput(self,
                               ) -> None:
            """
            Write the MTSET input file walls.xxx for the given case. 

            Parameters:
            -----------
            domain_boundaries : np.ndarray[float]
                Array containing the domain boundaries in the format [XFRONT, XREAR, YBOT, YTOP]

            Returns:
            --------
            None
                Output of function is the input file to MTSET, walls.xxx, where xxx is equal to self.case_name
            """

            domain_boundaries = self.GetGridSize()

            # Get profiles of centerbody and duct
            xy_centerbody = self.GetProfileCoordinates(self.centerbody_params)
            xy_duct = self.GetProfileCoordinates(self.duct_params,
                                                 self.ducted_fan_design_params["Duct Leading Edge Coordinates"])
            
            # Generate walls.xxx input data structure
            file_name = "walls." + self.case_name
            with open(file_name, "w") as file:
                # Write opening lines of the file
                file.write(self.case_name + '\n')
                file.write('    '.join(map(str, domain_boundaries)) + '\n')

                # Write centerbody profile coordinates, using a tab delimiter
                for row in xy_centerbody:
                    file.write('    '.join(map(str, row)) + '\n')
                
                # Write separator line, using a tab delimiter
                file.write('    '.join(map(str, [999., 999.])) + '\n')

                # Write duct profile coordinates, using a tab delimiter
                for row in xy_duct:
                    file.write('    '.join(map(str, row)) + '\n')

            return None


    class fileHandlingMTFLO:

        def __init__(self):
            pass

        def GetBladeParameters(self,
                               design_params: dict,
                               ) -> dict:
            """

            Based on the blade design parameters, determine the blade geometry parameters needed
            


            """

            # Initialize the airfoil parameterization class
            profileParameterizationClass = AirfoilParameterization()

            # Extract the profile parameters from the design parameters
            # Restructure the input dictionary to a numpy array for the airfoil parameterization and a parameterization dictionary
            b_coeff = np.array([design_params["b_0"], design_params["b_2"], design_params["b_8"], design_params["b_15"], design_params["b_17"]])

            parameterization = {"x_t": design_params["x_t"],
                                "y_t": design_params["y_t"],
                                "x_c": design_params["x_c"],
                                "y_c": design_params["y_c"], 
                                "z_TE": design_params["z_TE"],
                                "dz_TE": design_params["dz_TE"],
                                "r_LE": design_params["r_LE"], 
                                "trailing_wedge_angle": design_params["trailing_wedge_angle"], 
                                "trailing_camberline_angle": design_params["trailing_camberline_angle"], 
                                "leading_edge_direction": design_params["leading_edge_direction"],
                                }
            
            # Calculate the thickness and blade slope distributions along the blade profiles. 
            # All parameters are nondimensionalized by the chord length, so they are then multiplied by the chord length to get the correct dimensions
            # Additionally, offsets the profiles to the correct spatial coordinate (x,r)
            thickness_distr, thickness_data_points, geometric_blade_slope, blade_slope_points = profileParameterizationClass.ComputeProfileCoordinates(b_coeff,
                                                                                                                                                       parameterization)
            thickness_distr = thickness_distr * design_params["Chord Length"]
            thickness_data_points = thickness_data_points * design_params["Chord Length"]
            geometric_blade_slope = geometric_blade_slope * design_params["Chord Length"]
            blade_slope_points = blade_slope_points * design_params["Chord Length"]
            
                        




            pass

        def ConstructBlades(self,
                            blading_params: dict,
                            design_params: np.ndarray[dict]):
            """

            interpolate between the airfoils to create the blade geometry inputs needed to create the MTFLO input file
            Uses a rectangular bivariate spine interpolation on (r,x) to obtain the blade geometry at each radial, and axial station 
    
            """

            # Collect blade geometry at each of the radial stations
            blade_geometry = np.zeros(len(design_params))
            for station in range(len(design_params)):
                blade_geometry[station] = self.GetBladeParameters(design_params[station])



            pass

        def GenerateMTFLOInput(self,
                               operating_conditions: dict,
                               ) -> None:
            """

            Write the MTFLO input file for the given case

            """

            return None



if __name__ == "__main__":

    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017094989769520816, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 1.0}
    design_params = {"Duct Leading Edge Coordinates": (0, 2)}

    call_class = fileHandling.fileHandlingMTSET(n2415_coeff, n2415_coeff, design_params, "test_case")
    call_class.GenerateMTSETInput([0, 1, 0, 3])
