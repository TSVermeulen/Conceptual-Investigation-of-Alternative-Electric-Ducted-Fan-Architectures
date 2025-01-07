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
            Initialize the fileHandlingMTFLO class.

            This method sets up the initial state of the class.

            Returns
            -------
            None
            """

            stage_count, case_name = args 

            self.stage_count: int = stage_count
            self.case_name: str = case_name


        def GetBladeParameters(self, 
                               design_params: dict,
                               ) -> dict:
            """
            Calculate the thickness and blade slope distributions along the blade profile based on the given design parameters.

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

            Returns:
            --------
            blade_geometry : dict
                A dictionary containing the following keys:
                - "thickness_distr": An array of the thickness distribution along the blade profile.
                - "thickness_data_points": An array of the thickness data points along the blade profile.
                - "camber_distr": An array of the camber distribution along the blade profile.
                - "camber_data_points": An array of the camber data points along the blade profile.
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
            thickness_distr, thickness_data_points, camber_distr, camber_data_points = profileParameterizationClass.GetBladeParameters(b_coeff,
                                                                                                                                       parameterization,
                                                                                                                                       )

            # Construct output dictionary
            # Output dictionary contains the data points for the thickness and camber distributions
            # No interpolation is done yet, to avoid needing to resample. 
            blade_geometry = {"thickness_points": thickness_data_points, 
                              "thickness_data": thickness_distr,
                              "camber_points": camber_data_points, 
                              "camber_data": camber_distr,
                              }
            
            return blade_geometry

        def ConstructBlades(self,
                            blading_params: dict,
                            design_params: np.ndarray[dict],
                            ) -> dict:
            """
            Construct interpolants for the blade geometry using the x, r, thickness, camber, and entropy distributions.
            Uses the principle of superposition to split out the blade design into separate interpolations of parameters. 

            Parameters:
            -----------
            - blading_params : dict
                Dictionary containing the blading parameters for the blade. The dictionary should include the following keys:
                    - "rotational_rate": The rotational rate of the blade.
                    - "blade_count": Integer of the number of blades.
                    - "radial_stations": Numpy array of the radial stations along the blade span.
                    - "chord_length": Numpy array of the chord length distribution along the blade span.
                    - "sweep_angle": Numpy array of the sweep angle distribution along the blade span.
                    - "twist_angle": Numpy array of the twist angle distribution along the blade span.
            - design_params : np.ndarray[dict]
                Numpy array containing an equal number of dictionary entries as there are radial stations. Each dictionary must contain 
                the following keys:
                    - "b_0", "b_2", "b_8", "b_15", "b_17": Coefficients for the airfoil parameterization.
                    - "x_t", "y_t", "x_c", "y_c": Coordinates for the airfoil parameterization.
                    - "z_TE", "dz_TE": Trailing edge parameters.
                    - "r_LE": Leading edge radius.
                    - "trailing_wedge_angle": Trailing wedge angle.
                    - "trailing_camberline_angle": Trailing camberline angle.
                    - "leading_edge_direction": Leading edge direction.

            Returns:
            --------
            - constructed_blade : dict
                Dictionary containing the constructed blade geometry. The dictionary includes the following keys:
                    - "chord_distribution": Cubic spline interpolant for the chord length distribution along the blade span.
                    - "sweep_distribution": Cubic spline interpolant for the sweep angle distribution along the blade span.
                    - "thickness_distribution": Bivariate spline interpolant for the circumferential thickness distribution along the blade profile.
                    - "camber_distribution": Bivariate spline interpolant for the camber distribution along the blade profile.
                    - "entropy_distribution": Bivariate spline interpolant for the entropy distribution along the blade profile.
            """

            # Collect blade geometry at each of the radial stations
            # Note that the blade geometry is a dictionary containing the thickness and camber distributions
            # Splits out the blade_geometry dictionary into separate lists
            # Thickness and camber distributions are also divided by the cosine of the twist angle to account for blade twisting. 
            blade_geometry = [None] * len(design_params)
            thickness_profile_distributions = [None] * len(design_params)
            thickness_data_points = [None] * len(design_params)
            camber_profile_distributions = [None] * len(design_params)
            camber_data_points = [None] * len(design_params)

            for station in range(len(design_params)):
                blade_geometry[station] = self.GetBladeParameters(design_params[station])
    
                thickness_profile_distributions[station] = blade_geometry[station]["thickness_data"] / np.cos(blading_params["twist_angle"][station])
                thickness_data_points[station] = blade_geometry[station]["thickness_points"] 
                camber_profile_distributions[station] = blade_geometry[station]["camber_data"] / np.cos(blading_params["twist_angle"][station])
                camber_data_points[station] = blade_geometry[station]["camber_points"]
                
            # Construct the chord length distribution
            chord_distribution = interpolate.CubicSpline(blading_params["radial_stations"], 
                                                         blading_params["chord_length"], 
                                                         extrapolate=False,
                                                         )
            
            # Construct the sweep distribution
            # Note that this is not the sweep angle, but rather the leading edge offset, measured from the root
            sweep_distribution = interpolate.CubicSpline(blading_params["radial_stations"],
                                                         blading_params["root_LE_coordinate"] + blading_params["radial_stations"] * np.sin(blading_params["sweep_angle"]),
                                                         extrapolate=False,
                                                         )
                        
            # Construct the thickness and camber bivariate spline interpolations
            # First determine the appropriate interpolation method based on the minimum dimension of the data input
            # Then use the RegularGridInterpolator to create the interpolation function
            method = 'quintic'
            if 4 <= len(thickness_data_points) < 6:
                method = 'cubic'
            elif len(thickness_data_points) < 4:
                method = 'linear'

            thickness_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                          thickness_data_points[0]),
                                                                         thickness_profile_distributions,
                                                                         method=method,
                                                                         bounds_error=True,
                                                                         ) 
    
            # Determine the appropriate interpolation method for the camber distribution
            method = 'quintic'
            if 4 <= len(camber_data_points) < 6:
                method = 'cubic'
            elif len(camber_data_points) < 4:
                method = 'linear'

            camber_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                       camber_data_points[0]),
                                                                      camber_profile_distributions,
                                                                      method=method,
                                                                      bounds_error=True,
                                                                      )  

            # CONSTRUCT DUMMY ENTROPY DISTRIBUTION
            # This needs to be replaced with a proper entropy calculation, which is currently not implemented.
            # Current implementation is a placeholder to allow for the construction of the MTFLO input file
            # The interpolated entropy distribution is simply zero at all points on the domain
            entropy_profile_distribution = np.zeros_like(camber_profile_distributions)
            entropy_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                      camber_data_points[0]),
                                                                     entropy_profile_distribution,
                                                                     method=method,
                                                                     bounds_error=True,
                                                                     )  

            # Construct output data
            constructed_blade = {"chord_distribution": chord_distribution,
                                 "sweep_distribution": sweep_distribution,
                                 "thickness_distribution": thickness_distribution,
                                 "camber_distribution": camber_distribution,
                                 "entropy_distribution": entropy_distribution,
                                 }
            
            return constructed_blade
        

        def GenerateMTFLOInput(self,
                               blading_params: np.ndarray[dict],
                               design_params: np.ndarray[dict],
                               ) -> None:
            """
            Write the MTFLO input file for the given case.

            Parameters:
            -----------
            - blading_params : np.ndarray[dict]
                Array containing the blading parameters for each stage. Each dictionary should include the following keys:
                - "root_LE_coordinate": The leading edge coordinate at the root of the blade.
                - "rotational_rate": The rotational rate of the blade.
                - "blade_count": The number of blades.
                - "radial_stations": Numpy array of the radial stations along the blade span.
                - "chord_length": Numpy array of the chord length distribution along the blade span.
                - "sweep_angle": Numpy array of the sweep angle distribution along the blade span.
                - "twist_angle": Numpy array of the twist angle distribution along the blade span.
            - design_params: np.ndarray[dict]
                Array containing an equal number of dictionary entries as there are stages. Each dictionary must contain the following keys:
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
            None
            """

            # Open the tflow.xxx file and start writing the required input data to it
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
                    file.write(str(blading_params[stage]["blade_count"]) + '\n')
                    file.write('END\n \n')

                    # Write the rotational rate in units of omega * L / V
                    file.write('OMEGA\n')
                    file.write(str(blading_params[stage]["rotational_rate"]) + '\n')
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
                    # The MTFLO code cannot accept an input file with more than 16x16 points in the streamwise and radial directions
                    # Hence n_points=16
                    # The axial points are spaced using a cosine spacing for increased resolution at the LE and TE
                    # The radial points are spaced using constant spacing. 
                    n_points = 16
                    axial_points = (1 - np.cos(np.linspace(0, np.pi, n_points))) / 2
                    radial_points = np.linspace(min(blading_params[stage]["radial_stations"]), 
                                                max(blading_params[stage]["radial_stations"]), 
                                                n_points,
                                                )

                    # Loop over the radial points and construct the data for each radial point
                    # Each radial point is defined as a "section" within the input file
                    for i in range(n_points): 
                        # Create a section in the input file
                        file.write('SECTION\n')

                        # Compute chord length of blade at radial station from the provided interpolant
                        local_chord = blade_geometry["chord_distribution"](radial_points[i])
                        axial_coordinates = axial_points * local_chord

                        # Compute the geometric blade slope dtheta/dm' in the blade-to-blade plane 
                        # Camber is denormalised using the local chord length
                        camber_distribution = blade_geometry["camber_distribution"]((radial_points[i], axial_points)) * local_chord

                        # Compute the circumferential angle and local streamsurface radius distributions
                        theta = np.atan2(camber_distribution, radial_points[i])
                        r = np.sqrt(camber_distribution ** 2 + radial_points[i] ** 2)
                        
                        # Compute the m' coordinate distribution. 
                        m_prime = np.zeros_like(r)
                        for i in range(len(m_prime)):
                            if i != 0:
                                # Initial coordinate m_prime[0] is arbitrary, and merely shifts the profile to the origin
                                # Use trapezoidal integration to compute the m_prime coordinates
                                m_prime[i] = m_prime[i - 1] + 2 / (r[i] + r[i - 1]) * np.sqrt((r[i] - r[i-1]) ** 2 + (axial_coordinates[i] - axial_coordinates[i - 1]) ** 2)

                        # Compute the blade slope distribution, defined as dtheta/dm'. Use a second order scheme at the domain edges for improved accuracy. 
                        blade_slope_distribution = np.gradient(theta, m_prime, edge_order=2)

                        # Compute the local sweep (i.e. LE offset) at the radial station from the provided interpolant
                        sweep = blade_geometry["sweep_distribution"](radial_points[i]) 

                        # Compute the thickness distribution at the radial station from the provided interpolant
                        # Thickness is denormalised using the local chord length
                        thickness_distribution = blade_geometry["thickness_distribution"]((radial_points[i], axial_points)) * local_chord   

                        # Compute the entropy distribution at the radial station from the provided interpolant
                        # Entropy is denormalised using the local chord length
                        entropy_distribution = blade_geometry["entropy_distribution"]((radial_points[i], axial_points)) * local_chord

                        # Loop over the streamwise points and construct the data for each streamwise point
                        # Each data point consists of the data [x, r, T, sRel, dS]
                        for j in range(n_points):  
                            # Write data to row
                            row = np.array([axial_points[j] * local_chord + sweep,
                                            radial_points[i],
                                            thickness_distribution[j],
                                            blade_slope_distribution[j],
                                            entropy_distribution[j],
                                            ])
    
                            # Write the row to the file
                            file.write('    '.join(map(str, row)) + '\n')
                        # End the radial section
                        file.write('END\n')
                    # End the stage 
                    file.write('\nEND\n \n')
                # End the input file
                file.write('END\n')
                        
            return None


if __name__ == "__main__":

    import time

    # Perform test generation of walls.xxx file using dummy inputs
    # Creates a dummy duct with a naca 2415 profile for the centerbody and duct
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017094989769520816, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 1.0}
    design_params = {"Duct Leading Edge Coordinates": (0, 2), "Duct Outer Diameter": 1.0}

    starttime = time.time()
    call_class = fileHandling.fileHandlingMTSET(n2415_coeff, n2415_coeff, design_params, "test_case")
    call_class.GenerateMTSETInput()
    endtime = time.time()
    print("Execution of GenerateMTSETInput() took", endtime - starttime, "seconds")

    # Perform test generation of tflow.xxx file using dummy inputs
    # Creates an input file using 2 stages, a rotor and a stator
    blading_parameters = [{"root_LE_coordinate": 0., "rotational_rate": 10, "blade_count": 18, "radial_stations": [0.1, 1], "chord_length": [0.2, 0.2], "sweep_angle":[np.pi/4, np.pi/4], "twist_angle": [0, np.pi / 3]},
                          {"root_LE_coordinate": 2., "rotational_rate": 0, "blade_count": 10, "radial_stations": [0.1, 1], "chord_length": [0.2, 0.2], "sweep_angle":[np.pi/4, np.pi/4], "twist_angle": [0, np.pi/8]}]
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    starttime = time.time()
    call_class = fileHandling.fileHandlingMTFLO(2, "test_case")
    call_class.GenerateMTFLOInput(blading_parameters, 
                                  design_parameters)
    endtime = time.time()
    print("Execution of GenerateMTFLOInput() took", endtime - starttime, "seconds")
    
