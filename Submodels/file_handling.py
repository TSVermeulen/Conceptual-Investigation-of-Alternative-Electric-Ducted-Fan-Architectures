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
from scipy import interpolate
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

        def __init__(self, *args) -> None:
            """
            
            """

            stage_count, case_name = args 

            self.stage_count: int = stage_count
            self.case_name: str = case_name


        def GetBladeParameters(self, 
                               design_params: dict,
                               ) -> dict:
            """
            Calculate the thickness and blade slope distributions along the blade profiles based on the given design parameters.

            Parameters:
            -----------
            design_params : dict
                A dictionary containing the design parameters for the blade. The dictionary should include the following keys:
                - "b_0", "b_2", "b_8", "b_15", "b_17": Coefficients for the airfoil parameterization.
                - "x_t", "y_t", "x_c", "y_c": Coordinates for the airfoil parameterization.
                - "z_TE", "dz_TE": Trailing edge parameters.
                - "r_LE": Leading edge radius.
                - "trailing_wedge_angle": Trailing wedge angle.
                - "trailing_camberline_angle": Trailing camberline angle.
                - "leading_edge_direction": Leading edge direction.
                - "Chord Length": The chord length of the blade.

            Returns:
            --------
            dict
                A dictionary containing the following keys:
                - "thickness_distr": Thickness distribution along the blade profile.
                - "thickness_data_points": Thickness data points along the blade profile.
                - "geometric_blade_slope": Geometric blade slope distribution.
                - "blade_slope_points": Blade slope data points.
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
            # All parameters are nondimensionalized by the chord length
            thickness_distr, thickness_data_points, geometric_blade_slope, blade_slope_points = profileParameterizationClass.GetBladeParameters(b_coeff,
                                                                                                                                                parameterization)
            thickness_distr = thickness_distr
            thickness_data_points = thickness_data_points
            geometric_blade_slope = geometric_blade_slope
            blade_slope_points = blade_slope_points
            
            # Construct output dictionary
            # Output dictionary contains cubic spline interpolated functions of the thickness and blade slope. 
            # Cubic spline extrapolation is allowed, but not recommended. Extrapolation is only used to handle round off errors (i.e. if x=0.99, to obtain the value at x=1)
            thickness_distribution = interpolate.CubicSpline(thickness_data_points, 
                                                             thickness_distr, 
                                                             extrapolate=True)
            blade_slope_distribution = interpolate.CubicSpline(blade_slope_points, 
                                                               geometric_blade_slope, 
                                                               extrapolate=True)
            
            blade_geometry = {"thickness_distr": thickness_distribution,
                              "blade_slope_distribution": blade_slope_distribution,
                              }
            
            return blade_geometry

        def ConstructBlades(self,
                            blading_params: dict,
                            design_params: np.ndarray[dict],
                            ) -> dict:
            """

            interpolate between the airfoils to create the blade geometry inputs needed to create the MTFLO input file
            Uses a rectangular bivariate spine interpolation on (r,x) to obtain the blade geometry at each radial, and axial station 

            TO DO:
            - Implement entropy calculation

            """

            # Collect blade geometry at each of the radial stations
            # Note that the blade geometry is a dictionary containing the thickness and blade slope interpolated distributions
            blade_geometry = [None] * len(design_params)
            for station in range(len(design_params)):
                blade_geometry[station] = self.GetBladeParameters(design_params[station])
            
            # Construct the chord length distribution
            chord_distribution = interpolate.CubicSpline(blading_params["radial_stations"], 
                                                         blading_params["chord_length"], 
                                                         extrapolate=True,
                                                         )
            
            # Construct the sweep angle distribution
            sweep_distribution = interpolate.CubicSpline(blading_params["radial_stations"],
                                                         blading_params["sweep_angle"],
                                                         extrapolate=True,
                                                         )

            # Construct the twist angle distribution
            twist_distribution = interpolate.CubicSpline(blading_params["radial_stations"],
                                                         blading_params["twist_angle"],
                                                         extrapolate=True,
                                                         )
            
            # Construct thickness and camber distribution rectangular data grid
            # Use a cosine spacing of data points to increase resolution at the leading and trailing edges
            # This is done to ensure that each profile is sampled at the same nondimensionalised points x/c, 
            # enabling the creation of a rectangular grid for the bivariate spline interpolation.
            # Uses 100 points to ensure an initial smooth interpolation. 
            n_points = 100
            cosine_space = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2

            # Loop over the different radial stations and compute the thickness and camber distribution for each station
            thickness_profile_distribution = np.zeros((len(design_params), n_points))
            slope_profile_distribution = np.zeros((len(design_params), n_points))
            for station in range(len(thickness_profile_distribution)):
                thickness_profile_distribution[station] = blade_geometry[station]["thickness_distr"](cosine_space)
                slope_profile_distribution[station] = blade_geometry[station]["blade_slope_distribution"](cosine_space)

            # Construct the thickness and slope bivariate spline interpolations
            # First determine the appropriate interpolation method based on the minimum dimension of the data input
            # Then use the regulargridinterpolator to create the interpolation function
            method = 'quintic'
            if 4 <= min(min(thickness_profile_distribution.shape), cosine_space.size) < 6:
                method = 'cubic'
            elif min(min(thickness_profile_distribution.shape), cosine_space.size) < 4:
                method = 'linear'

            thickness_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                          cosine_space),
                                                                         thickness_profile_distribution,
                                                                         method=method,
                                                                         ) 
            
            # Determine the appropriate interpolation method for the slope distribution
            method = 'quintic'
            if 4 <= min(min(slope_profile_distribution.shape), cosine_space.size) < 6:
                method = 'cubic'
            elif min(min(slope_profile_distribution.shape), cosine_space.size) < 4:
                method = 'linear'

            slope_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                      cosine_space),
                                                                     slope_profile_distribution,
                                                                     method=method,
                                                                     )  

            # CONSTRUCT DUMMY ENTROPY DISTRIBUTION
            # This needs to be replaced with a proper entropy calculation, which is currently not implemented.
            # Current implementation is a placeholder to allow for the construction of the MTFLO input file
            entropy_profile_distribution = np.zeros((len(blading_params["radial_stations"]), len(cosine_space)))
            entropy_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                       cosine_space),
                                                                      entropy_profile_distribution,
                                                                      method=method,
                                                                      )     

            # Construct output data
            constructed_blade = {"chord_distribution": chord_distribution,
                                 "sweep_distribution": sweep_distribution,
                                 "twist_distribution": twist_distribution,
                                 "thickness_distribution": thickness_distribution,
                                 "slope_distribution": slope_distribution,
                                 "entropy_distribution": entropy_distribution,
                                 }
            
            return constructed_blade

        def GenerateMTFLOInput(self,
                               blading_params: np.ndarray[dict],
                               design_params: np.ndarray[dict],
                               ) -> None:
            """

            Write the MTFLO input file for the given case

            """

            filename = "tflow." + self.case_name
            with open(filename, "w") as file:
                # Write the case name to the file
                file.write('NAME\n')
                file.write(self.case_name + '\n')
                file.write('END\n \n')

                # Loop over the number of stages and write the data for each stage
                for stage in range(self.stage_count):
                    # First write the "generic" data for the stage
                    # This includes the number of blades, the rotational rate, and the data types to be provided
                    # Formatting is in-line with the user guide from the MTFLOW documentation

                    # Start a stage block
                    file.write('STAGE\n \n')

                    # Write the number of blades
                    file.write('NBLADE\n')
                    file.write(blading_params[stage]["blade_count"] + '\n')
                    file.write('END\n \n')

                    # Write the rotational rate in units of omega * L / V
                    file.write('OMEGA\n')
                    file.write(blading_params[stage]["rotational_rate"] + '\n')
                    file.write('END\n \n')

                    # Write the data types to be provided for the stage
                    file.write('DATYPE \n')
                    file.write('x    r    T    Sr    DS\n')  # Use the x,r coordinates, together with thickness, blade slope, and entropy
                    multipliers = [1., 1., 1., 1., 1.]  # Add multipliers for each data type
                    additions = [0., 0., 0., 0., 0.]  # Add additions for each data type
                    file.write('*' + '    '.join(map(str, multipliers)) + '\n')
                    file.write('+' + '    '.join(map(str, additions)) + '\n')
                    file.write('END\n \n')

                    # Collect the blade geometry interpolations
                    blade_geometry: dict = self.ConstructBlades(blading_params[stage], 
                                                                design_params[stage])
                    
                    # Generate interpolated data to construct the file geometry
                    n_axial_points = 50
                    n_radial_points = 20
                    axial_points = np.linspace(0, 1, n_axial_points)
                    radial_points = np.linspace(0, 1, n_radial_points)

                    # Loop over the radial points and construct the data for each radial point
                    # Each radial point is defined as a "section" within the input file
                    for i in range(n_radial_points): 
                        # Create a section in the input file
                        file.write('SECTION\n')

                        # Loop over the streamwise points and construct the data for each streamwise point
                        # Each data point consists of the data [x, r, T, sRel, and dS]
                        for j in range(len(axial_points)):     
                            row = np.array([axial_points[j],
                                            radial_points[i],
                                            blade_geometry["thickness_distribution"]((radial_points[i], axial_points[j])),
                                            blade_geometry["slope_distribution"]((radial_points[i], axial_points[j])),
                                            blade_geometry["entropy_distribution"]((radial_points[i], axial_points[j])),
                                            ])
                            
                            # Write the row to the file
                            file.write('    '.join(map(str, row)) + '\n')

                        # End the radial section
                        file.write('END\n')
                        

                    





            return None



if __name__ == "__main__":

    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017094989769520816, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 1.0}
    design_params = {"Duct Leading Edge Coordinates": (0, 2), "Duct Outer Diameter": 1.0}

    call_class = fileHandling.fileHandlingMTSET(n2415_coeff, n2415_coeff, design_params, "test_case")
    call_class.GenerateMTSETInput()


    call_class = fileHandling.fileHandlingMTFLO(1, "test_case")
    test = call_class.ConstructBlades({"radial_stations": [0, 1], "chord_length": [0.2, 0.2], "sweep_angle":[0, 0], "twist_angle": [0, 0]}, [n2415_coeff, n2415_coeff])
    
    generate_test = call_class.GenerateMTFLOInput([{"stage_count": 1, "blade_count": 18, "radial_stations": [0, 1], "chord_length": [0.2, 0.2], "sweep_angle":[0, 0], "twist_angle": [0, 0]}], [n2415_coeff, n2415_coeff])
    print(generate_test)
