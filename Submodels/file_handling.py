"""
file_handling
=============

Description
-----------
This module provides classes and methods for handling file operations related to 
the conceptual investigation of alternative electric ducted fan architectures. 
It includes functionalities for generating the input files for the MTSET and MTFLO tools, forming 
part of the MTFLOW software suite from MIT.

Classes
-------
fileHandling
    A grouping class for file handling operations.

fileHandling.fileHandlingMTSET
    Handles the generation of the MTSET input file, walls.xxx.
    This input file contains the axisymmetric bodies present within the domain.

fileHandling.fileHandlingMTFLO
    Handles the generation of the MTFLO input file, tflow.xxx.
    This input file contains the forcing field terms corresponding to the blade rows (rotors and stators).


Examples
--------
>>> n2415_coeff = {"b_0": 0.203, "b_2": 0.319, "b_8": 0.042, "b_15": 0.750, "b_17": 0.679, 
...                "x_t": 0.299, "y_t": 0.060, "x_c": 0.405, "y_c": 0.020, "z_TE": -0.00034, 
...                "dz_TE": 0.0017, "r_LE": -0.024, "trailing_wedge_angle": 0.167, "trailing_camberline_angle": 0.065, 
...                "leading_edge_direction": 0.094, "Chord Length": 1.0}
>>> design_params = {"Duct Leading Edge Coordinates": (0, 2), "Duct Outer Diameter": 1.0}
>>> call_class = fileHandling.fileHandlingMTSET(n2415_coeff, n2415_coeff, design_params, "test_case")
>>> call_class.GenerateMTSETInput()

>>> blading_parameters = [
...     {
...         "root_LE_coordinate": 0.0,
...         "rotational_rate": 10,
...         "blade_count": 18,
...         "radial_stations": np.array([0.1, 1.0]),
...         "chord_length": np.array([0.2, 0.2]),
...         "sweep_angle": np.array([np.pi / 4, np.pi / 4]),
...         "blade_angle": np.array([0, np.pi / 3]),
...     },
...     {
...         "root_LE_coordinate": 2.0,
...         "rotational_rate": 0,
...         "blade_count": 10,
...         "radial_stations": np.array([0.1, 1.0]),
...         "chord_length": np.array([0.2, 0.2]),
...         "sweep_angle": np.array([np.pi / 4, np.pi / 4]),
...         "blade_angle": np.array([0, np.pi / 8]),
...     },
... ]
>>> design_parameters = [[n2415_coeff, n2415_coeff],
...                      [n2415_coeff, n2415_coeff],
...                      ]
>>> call_class = fileHandling.fileHandlingMTFLO(2, "test_case")
>>> call_class.GenerateMTFLOInput(blading_parameters, design_parameters)

Notes
-----
This module is designed to work with the BP3434 profile parameterization defined in the Parameterizations.py file
Ensure that the input dictionaries are correctly formatted. For details on the specific inputs needed, see the 
different method docstrings.

When executing the file as a standalone, it uses the inputs and calls contained within the if __name__ == "__main__" section. 
This part also imports the time module to measure the time needed to perform each file generation call. This is beneficial in runtime optimization.

References
----------
The coordinate transformation from cartesian space to the developed coordinates m'-theta used to calculate 
s_rel in fileHandling.fileHandlingMTFLO.GenerateMTFLOInput() is documented in the MISES user manual:
https://web.mit.edu/drela/Public/web/mises/mises.pdf

The required input data, limitations, and structures are documented within the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

The blade coordinate transformation, and order of calculations, is based on the implementation found in the BladeX module:
https://github.com/mathLab/BladeX 

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.3
Date [dd-mm-yyyy]: 06-04-2025

Changelog
---------
- V1.0: Initial working version
- V1.1: Updated test values. Added leading edge coordinate control of centrebody. Added floating point precision of 3 decimals for domain size. Updated input validation logic.
- V1.1.5: Fixed import logic of the Parameterizations module to handle local versus global file execution. 
- V1.2.0: Updated class initialization logic and function inputs to enable existing geometry inputs for debugging/validation
- V1.2.1: Fixed duplicate leading edge coordinate in fileHandling.fileHandlingMTSET.GetProfileCoordinates(). Implemented nondimensionalisation of geometric parameters for both MTSET and MTFLO input files. 
- V1.3: Significant reworks to help solve bugs and issues found in validation against the X22A ducted propeller case. Added the grid size as optional input in fileHandlingMTSET. Code now automatically determines degree of bivariate interpolants based on number of radial stations provided in input data. Factorized the GenerateMTFLOInput function. Fixed transformation from planar to cylindrical coordinate system based on the implementation found in the BladeX module. Fixed implementation of circumferential blade thickness and blade slope. 
"""

import numpy as np
from scipy import interpolate
from pathlib import Path
from typing import Optional

# Handle local versus global execution of the file with imports
if __name__ == "__main__":
    from Parameterizations import AirfoilParameterization
else:
    from .Parameterizations import AirfoilParameterization


class fileHandling:
    """
    This class contains all methods needed to generate the required input files walls.xxx and tflow.xxx for an MTFLOW analysis. 
    """


    def __init__(self) -> None:
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
        Class for handling the generation of the MTSET input file (walls.xxx).
        
        This class provides methods to generate the input file containing the axisymmetric 
        bodies present within the domain. It handles the calculation of grid sizes and 
        profile coordinates for both the center body and duct.
        """


        def __init__(self, 
                     params_CB: dict,
                     params_duct: dict,
                     case_name: str,
                     ref_length: float,
                     external_input : bool = False,
                     domain_boundaries : Optional[list[float, float, float, float]] = None,
                     ) -> None:
            """
            Initialize the fileHandlingMTSET class.
        
            This method sets up the initial state of the class.

            Parameters
            ----------
            - params_CB : dict
                Dictionary containing parameters for the centerbody.
            - params_duct : dict
                Dictionary containing parameters for the duct.
            - case_name : str
                Name of the case being handled.
            - ref_length : float
                The reference length used by MTFLOW to non-dimensionalise all the dimensions.
            - external_input : bool, optional
                A control boolean to bypass the input of the "proper" centerbody and duct dictionaries. This is 
                useful when debugging or running cases where pre-existing geometry is to be used, rather than parameterized geometry. 
            - domain_boundaries : list[float, float, float, float], optional
                A list containing the grid boundaries in the format [XFRONT, XREAR, YBOT, YTOP]. Note that these boundaries must already be non-dimensionalised by the reference length!

            Returns
            -------
            None
            """

            # Input validation
            # Required keys for parameter dictionaries
            required_keys = {"Leading Edge Coordinates",
                             "Chord Length",
                             "b_0", "b_2", "b_8", "b_15", "b_17",
                             "x_t", "y_t", "x_c", "y_c",
                             "z_TE", "dz_TE", "r_LE",
                             "trailing_wedge_angle",
                             "trailing_camberline_angle",
                             "leading_edge_direction",
                             }
            
            if domain_boundaries is None:
                domain_boundaries = [None, None, None, None]
            self.domain_boundaries = domain_boundaries

            # Only perform complete input validation if the input dictionaries contain data
            keys_to_check = {"Leading Edge Coordinates", "Chord Length"}
            if not external_input:
                keys_to_check = required_keys
                  
            for params, name in [(params_CB, "params_CB"), (params_duct, "params_duct")]:
                missing_keys = keys_to_check - set(params.keys())
                if missing_keys:
                    raise ValueError(f"Missing required keys in {name}: {missing_keys}")

            if not isinstance(case_name, str):
                raise TypeError("case_name must be a string")
            
            if ref_length <= 0:
                raise ValueError("ref_length must be a positive float")
            
            self.centerbody_params = params_CB
            self.duct_params = params_duct
            self.case_name = case_name
            self.ref_length = ref_length
            self.external_input = external_input

            # Define the Grid size calculation constants
            self.DEFAULT_Y_TOP = 1.0
            self.Y_TOP_MULTIPLIER = 2.5
            self.X_FRONT_OFFSET = 2
            self.X_AFT_OFFSET = 2

            # Define key paths/directories
            self.parent_dir = Path(__file__).resolve().parent.parent
            self.submodels_path = self.parent_dir / "Submodels"
                        

        def GetGridSize(self) -> list[float, float, float, float]:
            """
            Determine grid size - x & y coordinates of grid boundary. 
            Non-dimensionalises all geometry based on the reference length, in accordance with the MTFLOW documentation. 

            Returns
            -------
            - list[float, float, float, float]
                A list containing the grid boundaries in the format [XFRONT, XREAR, YBOT, YTOP]
            """

            # If all boundaries are provided, return them directly
            if all(entry is not None for entry in self.domain_boundaries):
                return self.domain_boundaries
            
            X_FRONT = self.domain_boundaries[0]
            X_AFT = self.domain_boundaries[1]
            Y_TOP = self.domain_boundaries[3]
            Y_BOT = 0.

            # Only computes the domain boundaries if they are not provided as an input
            if X_FRONT is None:
                # Calculate X-domain front boundary
                X_FRONT = round((min(self.duct_params["Leading Edge Coordinates"][0], 
                                self.centerbody_params["Leading Edge Coordinates"][0]) - self.X_FRONT_OFFSET) / self.ref_length,
                                3)
            if X_AFT is None:
                # Calculate X-domain aft boundary
                X_AFT = round((max(self.duct_params["Leading Edge Coordinates"][0] + self.duct_params["Chord Length"], 
                              self.centerbody_params["Leading Edge Coordinates"][0] + self.centerbody_params["Chord Length"]) + self.X_AFT_OFFSET) / self.ref_length,
                              3)
            if Y_TOP is None:
                # Calculate upper Y-domain boundary
                Y_TOP = round(max(self.DEFAULT_Y_TOP, 
                                 self.Y_TOP_MULTIPLIER * self.duct_params["Leading Edge Coordinates"][1] / self.ref_length),
                              3)
            
            return [X_FRONT, X_AFT, Y_BOT, Y_TOP]


        def GetProfileCoordinates(self,
                                  x: dict,
                                  ) -> tuple[np.ndarray[float], np.ndarray[float]]:
            """
            Compute the profile coordinates of an airfoil based on given design parameters and leading edge coordinates.
            Note that the outputs are still dimensional, and are yet to be non-dimensionalised!

            Parameters:
            -----------
            x : dict
                Dictionary of design parameters for the profile parameterization.
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
            lower_x = lower_x * x["Chord Length"]
            upper_y = upper_y * x["Chord Length"]
            lower_y = lower_y * x["Chord Length"]

            # Offset airfoil using the (dX, dY) input to shift the profile coordinates to the appropriate 
            # location within the domain 
            upper_x += x["Leading Edge Coordinates"][0]
            lower_x += x["Leading Edge Coordinates"][0]
            upper_y += x["Leading Edge Coordinates"][1]
            lower_y += x["Leading Edge Coordinates"][1]
            
            # Combine the upper and lower profile coordinates to get the complete coordinate set
            x = np.concatenate((np.flip(upper_x), lower_x[1:]), axis=0) 
            y = np.concatenate((np.flip(upper_y), lower_y[1:]), axis=0) 

            return np.vstack((x, y)).T
        
        
        def GenerateMTSETInput(self,
                               xy_centerbody: tuple[np.ndarray[float], np.ndarray[float]] = None,
                               xy_duct: tuple[np.ndarray[float], np.ndarray[float]] = None,
                               ) -> None:
            """
            Write the MTSET input file walls.xxx for the given case. 

            Parameters:
            -----------
            xy_centerbody : tuple[np.ndarray[float], np.ndarray[float]], optional
                Tuple containing the x and y coordinates of the centerbody profile.
            xy_duct : tuple[np.ndarray[float], np.ndarray[float]], optional
                Tuple containing the x and y coordinates of the duct profile.

            Returns:
            --------
            None
                Output of function is the input file to MTSET, walls.xxx, where xxx is equal to self.case_name
            """

            domain_boundaries = self.GetGridSize()

            # Get profiles of centerbody and duct
            # These can be optionally input into the function to bypass the airfoil parameterization routines. This is 
            # useful if existing geometry is being used, for which no parameterization has been generated. 
            if xy_centerbody is None and not self.external_input:
                xy_centerbody = self.GetProfileCoordinates(self.centerbody_params)
            if xy_duct is None and not self.external_input:
                xy_duct = self.GetProfileCoordinates(self.duct_params)
            
            # Non-dimensionalise the profile coordinates
            xy_centerbody = xy_centerbody / self.ref_length

            # Generate walls.xxx input data structure
            file_path = self.submodels_path / "walls.{}".format(self.case_name)
            with open(file_path, "w") as file:
                # Write opening lines of the file
                file.write(self.case_name + '\n')
                file.write('    '.join(map(str, domain_boundaries)) + '\n')

                # Write centerbody profile coordinates, using a tab delimiter
                for row in xy_centerbody:
                    file.write('    '.join(map(str, row)) + '\n')
        
                # Write duct profile coordinates, using a tab delimiter
                # Only write it if a duct is defined, otherwise skip this step
                if xy_duct is not None:
                    # Write separator line, using a tab delimiter
                    file.write('    '.join(map(str, [999., 999.])) + '\n')

                    # Non-dimensionalise the duct coordinates
                    xy_duct = xy_duct / self.ref_length

                    for row in xy_duct:
                        file.write('    '.join(map(str, row)) + '\n')


    class fileHandlingMTFLO:
        """
        Class for handling the generation of the MTFLO input file (tflow.xxx).
        
        This class provides methods to generate the input file containing the blade rows (rotors and stators).
        """

        SYMMETRIC_LIMIT = 1E-3

        def __init__(self, 
                     case_name: str,
                     ref_length: float,
                     centerbody_rotor_thickness: float = 0.1,
                     ) -> None:
            """
            Initialize the fileHandlingMTFLO class.
            This method sets up the initial state of the class.

            Parameters
            ----------
            - case_name : str
                Name of the case being handled.
            - ref_length : float
                The reference length used by MTFLOW to non-dimensionalise all the dimensions.
            - centerbody_rotor_thickness : float, optional
                The cutoff radius in meters below which we do not check the circumferential thickness limit to avoid numerical false triggers. 

            Returns
            -------
            None
            """

            self.case_name= case_name
            self.ref_length = ref_length
            self.CENTERBODY_ROTOR_THICKNESS = centerbody_rotor_thickness

            # Define key paths/directories
            self.parent_dir = Path(__file__).resolve().parent.parent
            self.submodels_path = self.parent_dir / "Submodels"


        def ValidateBladeThickness(self, 
                                   local_thickness: float, 
                                   local_radius: float, 
                                   blade_count: int) -> None:
            """
            Validate that blade thickness doesn't exceed the complete blockage limit.
            If radius is less than self.CENTERBODY_ROTOR_THICKNESS, the function does nothing. 

            Parameters
            ----------
            local_thickness : float
                The local profile thickness
            local_radius : float
                The local radius of the blade-to-blade plane
            blade_count : int
                The total number of blades in the blade-to-blade plane
            
            Returns
            -------
            None
            """

            thickness_limit = 2 * np.pi * local_radius / blade_count
            if local_thickness >= thickness_limit and local_radius > self.CENTERBODY_ROTOR_THICKNESS:
                raise ValueError(f"The cumulative blade thickness exceeds the complete blockage limit of 2PIr at r={local_radius}")


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
            profileParameterizationClass = AirfoilParameterization(symmetric_limit=self.SYMMETRIC_LIMIT)

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
            thickness_distr, thickness_data_points, camber_distr, camber_data_points = profileParameterizationClass.ComputeBezierCurves(b_coeff,
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
            Construct interpolants for the blade geometry using the x, r, thickness and camber distributions.
            Uses the principle of superposition to split out the blade design into separate interpolations of parameters. 

            Parameters:
            -----------
            - blading_params : dict
                Dictionary containing the blading parameters for the blade. The dictionary should include the following keys:
                    - "rotational_rate": The rotational rate of the blade.
                    - "blade_count": Integer of the number of blades.
                    - "reference_section_blade_angle": The blade angle at the reference section of the blade span. This is used as the value on which the other blade angles are computed. 
                    - "ref_blade_angle": The set angle of the blades. 
                    - "radial_stations": Numpy array of the radial stations along the blade span.
                    - "chord_length": Numpy array of the chord length distribution along the blade span.
                    - "sweep_angle": Numpy array of the sweep angle distribution along the blade span.
                    - "blade_angle": Numpy array of the blade angle distribution along the blade span.
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
                    - "LE_distribution": Cubic spline interpolant for the leading edge x-coordinate distribution along the blade span.
                    - "thickness_distribution": Bivariate spline interpolant for the circumferential thickness distribution along the blade profile.
                    - "camber_distribution": Bivariate spline interpolant for the camber distribution along the blade profile.
            """

            # Collect blade geometry at each of the radial stations
            # Note that the blade geometry is a dictionary containing the thickness and camber distributions
            # Splits out the blade_geometry dictionary into separate lists
            blade_geometry = [None] * len(design_params)
            thickness_profile_distributions = [None] * len(design_params)
            camber_profile_distributions = [None] * len(design_params)

            for station in range(len(design_params)):
                blade_geometry[station] = self.GetBladeParameters(design_params[station])
                thickness_profile_distributions[station] = blade_geometry[station]["thickness_data"]
                camber_profile_distributions[station] = blade_geometry[station]["camber_data"]
            thickness_data_points = blade_geometry[0]["thickness_points"]
            camber_data_points = blade_geometry[0]["camber_points"]

            # Construct the chord length distribution
            chord_distribution = interpolate.CubicSpline(blading_params["radial_stations"], 
                                                         blading_params["chord_length"], 
                                                         extrapolate=False,
                                                         )
        
            # Construct the leading edge distribution, measured from the origin
            LE_distribution = interpolate.CubicSpline(blading_params["radial_stations"],
                                                      blading_params["root_LE_coordinate"] + blading_params["radial_stations"] * np.tan(blading_params["sweep_angle"]),
                                                      extrapolate=False,
                                                      )
            
            # Construct the blade angle (pitch) distribution
            pitch_distribution = interpolate.CubicSpline(blading_params["radial_stations"],
                                                         blading_params["blade_angle"],
                                                         extrapolate=False,
                                                         )
                        
            # Construct the thickness and camber bivariate spline interpolations
            # First determine the appropriate interpolation method based on the number of datapoints provided. 
            if len(blading_params["radial_stations"]) < 4:
                method = 'slinear'
            else:
                method = 'cubic'

            thickness_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                          thickness_data_points),
                                                                         thickness_profile_distributions,
                                                                         method=method,
                                                                         bounds_error=True,
                                                                         ) 

            camber_distribution = interpolate.RegularGridInterpolator((blading_params["radial_stations"],
                                                                       camber_data_points),
                                                                      camber_profile_distributions,
                                                                      method=method,
                                                                      bounds_error=True,
                                                                      )  

            # Construct output data
            constructed_blade = {"chord_distribution": chord_distribution,
                                 "leading_edge_distribution": LE_distribution,
                                 "thickness_distribution": thickness_distribution,
                                 "camber_distribution": camber_distribution,
                                 "pitch_distribution": pitch_distribution,
                                 }
            
            return constructed_blade
        

        def RotateProfile(self, 
                          pitch: float,
                          x_u: np.ndarray[float],
                          x_l: np.ndarray[float],
                          y_u: np.ndarray[float],
                          y_l: np.ndarray[float],
                          ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
            """
            Rotate a set of x,y coordinates counter-clockwise over the specified angle.

            Parameters
            ----------
            - pitch : float
                The blade pitch angle in radians.
            - x_u : np.ndarray[float]
                The upper surface x-coordinates.
            - x_l : np.ndarray[float]
                The lower surface x-coordinates.
            - y_u : np.ndarray[float]
                The upper surface y-coordinates.
            - y_l : np.ndarray[float]
                The lower surface y-coordinates

            Returns
            -------
            - rotated_upper_x : np.ndarray[float]
                The x-coordinates of the rotated upper surface.
            - rotated_upper_y : np.ndarray[float]
                The y-coordinates of the rotated upper surface.
            - rotated_lower_x : np.ndarray[float]
                The x-coordinates of the rotated lower surface.
            - rotated_lower_y : np.ndarray[float]
                The y-coordinates of the rotated lower surface.
            """

            # Construct the rotation matrix
            rotation_angle = np.pi / 2 - pitch
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                        [np.sin(rotation_angle), np.cos(rotation_angle)]])
            
            # Center the coordinates at the mid x and y points
            mid_x = (x_u[-1] + x_u[0]) / 2
            mid_y = ((y_u + y_l) / 2)[len(y_u) // 2]
            shifted_upper_x = x_u - mid_x
            shifted_lower_x = x_l - mid_x
            shifted_upper_y = y_u - mid_y
            shifted_lower_y = y_l - mid_y

            # Perform rotation.
            rotated_upper_points = np.dot(np.column_stack((shifted_upper_x, shifted_upper_y)), rotation_matrix.T)
            rotated_lower_points = np.dot(np.column_stack((shifted_lower_x, shifted_lower_y)), rotation_matrix.T)

            # Shift the rotated coordinates back to the original coordinate system
            rotated_upper_x = rotated_upper_points[:,0] + mid_x
            rotated_upper_y = rotated_upper_points[:,1] + mid_y
            rotated_lower_x = rotated_lower_points[:,0] + mid_x
            rotated_lower_y = rotated_lower_points[:,1] + mid_y
            
            return rotated_upper_x, rotated_upper_y, rotated_lower_x, rotated_lower_y 


        def PlanarToCylindrical(self,
                                y_u: np.ndarray[float],
                                y_l: np.ndarray[float],
                                r: float,
                                ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
            """
            Convert the planar airfoil coordinates to cylindrical coordinates.

            Parameters
            ----------
            - y_u : np.ndarray[float]
                The y-coordinates of the upper surface.
            - y_l : np.ndarray[float]
                The y-coordinates of the lower surface.
            - r : float
                The radius of the cylindrical surface.
            
            Returns
            -------
            - y_section_upper : np.ndarray[float]
                The y-coordinates of the upper surface in cylindrical coordinates.
            - y_section_lower : np.ndarray[float]
                The y-coordinates of the lower surface in cylindrical coordinates.
            - y_camber : np.ndarray[float]
                The y-coordinates of the camber line in cylindrical coordinates.
            - z_section_upper : np.ndarray[float]
                The z-coordinates of the upper surface in cylindrical coordinates.
            - z_section_lower : np.ndarray[float]
                The z-coordinates of the lower surface in cylindrical coordinates.
            - z_camber : np.ndarray[float]
                The z-coordinates of the camber line in cylindrical coordinates.
            """

            # Compute camber from the upper and lower surfaces
            camber = (y_u + y_l) / 2
                
            # Compute the theta angles
            if r == 0:
                # Ensure proper handling of calculations at the centerline where the radius is zero. 
                theta_up = y_u
                theta_low = y_l
                theta_camber = camber
            else:
                theta_up = y_u / r
                theta_low = y_l / r
                theta_camber = camber / r

            y_section_upper = r * np.sin(theta_up)
            y_section_lower = r * np.sin(theta_low)
            y_camber = r * np.sin(theta_camber)

            z_section_upper = r * np.cos(theta_up)
            z_section_lower = r * np.cos(theta_low)
            z_camber = r * np.cos(theta_camber)

            return y_section_upper, y_section_lower, y_camber, z_section_upper, z_section_lower, z_camber
        

        def CircumferentialThickness(self, 
                                     y_u: np.ndarray[float], 
                                     z_u: np.ndarray[float], 
                                     y_l: np.ndarray[float], 
                                     z_l: np.ndarray[float],
                                     r: float) -> np.ndarray[float]:
            """
            Compute the circumferential blade thickness distribution

            Parameters
            ----------
            - y_u : np.ndarray[float]
                The y-coordinates of the upper surface.
            - z_u : np.ndarray[float]
                The z-coordinates of the upper surface.
            - y_l : np.ndarray[float]
                The y-coordinates of the lower surface.
            - z_l : np.ndarray[float]
                The z-coordinates of the lower surface.
            - r : float
                The radius of the cylindrical surface.

            Returns
            -------
            - r * angle_range : np.ndarray[float]
                An array representing the circumferential thickness distribution along the blade profile, 
                calculated as the arc length subtended by the angular separation between the upper and lower surfaces.
            """
            
            # Compute subtended angles of the upper and lower surfaces
            theta_upper = np.atan2(y_u, z_u)
            theta_lower = np.atan2(y_l, z_l)

            # Compute the angle range by substracting the upper and lower angle distributions
            angle_range = np.abs(theta_upper - theta_lower)

            # Return the circumferential blade thickness, i.e. the azimuthal blade thickness, using the arc length formula
            return r * angle_range


        def GeometricBladeSlope(self,
                                y_camber: np.ndarray[float],
                                x_camber: np.ndarray[float],
                                z_camber: np.ndarray[float],
                                ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
            """
            Compute the geometric blade slope dtheta/dm' in the m'-theta coordinate system. 

            Parameters
            ----------
            - y_camber : np.ndarray[float]
                Array of y-coordinates of the (rotated) camber distribution in the cylindrical coordinate system.
            - x_camber : np.ndarray[float]
                Array of x-coordinates of the (rotated) camber distribution in the cylindrical coordinate system.
            - z_camber : np.ndarray[float]
                Array of z-coordinates of the (rotated) camber distribution in the cylindrical coordinate system.

            Returns
            -------
            - blade_slope : np.ndarray[float]
                An array containing the geometric blade slope at every x-coordinate along the blade profile. 
            - m_prime : np.ndarray[float]
                An array containing the m' coordinates at which the blade slope is defined. 
            - theta : np.ndarray[float]
                An array containing the circumferential angles theta of the camber line along the x-coordinates of the blade profile. 
            """               
            
            # Compute the circumferential angle theta
            theta = np.atan2(y_camber, z_camber)
            
            # Compute the radius of the meridional plane
            r = np.sqrt(np.square(y_camber) + np.square(z_camber))

            # Compute the m' coordinate. 
            m_prime = np.zeros_like(r) # Initialize m_prime as array of zeros. 
            dr = np.diff(r) 
            dx = np.diff(x_camber)
            radius_factor = r[:-1] + r[1:] # Equivalent to r[j] + r[j-1] 
            distance = np.sqrt(np.square(dr) + np.square(dx)) 

            if np.all(r == 0):
                # Handle the case where r=0, i.e. the centerline.
                m_prime = x_camber
            else:
                m_prime[1:] = np.cumsum(2 / radius_factor * distance)

            # Construct a cubic spline of the theta-m' curve
            # The blade slope is then found simply by evaluating the gradient of the spline. 
            spline = interpolate.make_splrep(m_prime,
                                             theta,
                                             k=3,
                                             s=0) 
                                             
            blade_slope = spline(m_prime,
                                 nu=1)
            
            return blade_slope, m_prime, theta
    

        def plot_blade_data(self,
                            stage: int,
                            radial_point: float,
                            x_points: np.ndarray[float],
                            rotated_upper_x: np.ndarray[float],
                            rotated_upper_y: np.ndarray[float],
                            rotated_lower_x: np.ndarray[float],
                            rotated_lower_y: np.ndarray[float],
                            circumferential_thickness: np.ndarray[float],
                            blade_slope: np.ndarray[float],
                            m_prime: np.ndarray[float],
                            theta: np.ndarray[float]) -> None:
            """
            Generate plots visualising the blade geometry. 

            Parameters
            ----------
            - stage : int
                The stage number of the blade row.
            - radial_point : float
                The radial point along the blade span.
            - x_points : np.ndarray[float]
                The x-coordinates of the blade profile.
            - rotated_upper_x : np.ndarray[float]
                The x-coordinates of the upper surface of the blade profile.
            - rotated_upper_y : np.ndarray[float] 
                The y-coordinates of the upper surface of the blade profile.
            - rotated_lower_x : np.ndarray[float]
                The x-coordinates of the lower surface of the blade profile.
            - rotated_lower_y : np.ndarray[float]
                The y-coordinates of the lower surface of the blade profile.
            - circumferential_thickness : np.ndarray[float]
                The circumferential thickness distribution along the blade profile.
            - blade_slope : np.ndarray[float]
                The geometric blade slope distribution along the blade profile.
            - m_prime : np.ndarray[float]
                The m' coordinates along the blade profile.
            - theta : np.ndarray[float]
                The circumferential angles theta of the camber line along the blade profile.

            Returns
            -------
            None
            """

            import matplotlib.pyplot as plt

            plt.figure(1)
            plt.xlabel("Axial coordinate [m]")
            plt.ylabel("Y coordinate [m]")
            plt.title(f"Rotated profile sections for stage {stage}")
            plt.plot(np.concatenate((rotated_upper_x, np.flip(rotated_lower_x)), axis=0), np.concatenate((rotated_upper_y, np.flip(rotated_lower_y)), axis=0), label=f"R={round(radial_point, 2)} m")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout()

            plt.figure(2)
            plt.xlabel("Axial coordinate [m]")
            plt.ylabel("Circumferential Blade Thickness $T_\\theta$ [m]")
            plt.title(f"Circumferential thickness distributions for stage {stage}")
            plt.plot(x_points, circumferential_thickness, label=f"R={round(radial_point, 2)} m")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout()

            plt.figure(3)
            plt.xlabel("Normalised meridional coordinate $||m'||=\\frac{m'}{max(m')}$ [-]")
            plt.ylabel("Blade angle $\\beta$ [deg]")
            plt.title(f"Blade angle distributions for stage {stage}")
            plt.plot(m_prime / m_prime[-1], np.degrees(np.atan(blade_slope)), label=f"R={round(radial_point, 2)} m")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout()

            plt.figure(4)
            plt.xlabel("Normalised meridional coordinate $||m'||=\\frac{m'}{max(m')}$ [-]")
            plt.ylabel("Circumferential angle $\\theta$ [deg]")
            plt.title(f"Circumferential angle distributions for stage {stage}")
            plt.plot(m_prime / m_prime[-1], np.degrees(theta), label=f"R={round(radial_point, 2)} m")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout()

            plt.figure(5)
            plt.xlabel("Normalised meridional coordinate $||m'||=\\frac{m'}{max(m')}$ [-]")
            plt.ylabel("Camberline blade slope $\\frac{d \\theta}{dm'}$ [-]")
            plt.title(f"Blade slope distributions for stage {stage}")
            plt.plot(m_prime / m_prime[-1], blade_slope, label=f"R={round(radial_point, 2)} m")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout()
            
        
        def GenerateMTFLOInput(self,
                               blading_params: np.ndarray[dict],
                               design_params: np.ndarray[dict],
                               plot: bool = False,
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
                - "ref_blade_angle": The set angle of the blades.
                - "reference_section_blade_angle": The blade angle at the reference section of the blade span. This is used as the value on which the other blade angles are computed.
                - "chord_length": Numpy array of the chord length distribution along the blade span.
                - "sweep_angle": Numpy array of the sweep angle distribution along the blade span.
                - "blade_angle": Numpy array of the twist angle distribution along the blade span.
            - design_params: np.ndarray[dict]
                Nested array containing an equal number of nests as there are stages. Each nested array has dictionary entries equal to the amount of radial stations considered. 
                Each dictionary must contain the following keys:
                - "b_0", "b_2", "b_8", "b_15", "b_17": Coefficients for the airfoil parameterization.
                - "x_t", "y_t", "x_c", "y_c": Coordinates for the airfoil parameterization.
                - "z_TE", "dz_TE": Trailing edge parameters.
                - "r_LE": Leading edge radius.
                - "trailing_wedge_angle": Trailing wedge angle.
                - "trailing_camberline_angle": Trailing camberline angle.
                - "leading_edge_direction": Leading edge direction.
                - "Chord Length": The chord length of the blade.
            - plot : bool, optional
                An optional controlling boolean to decide if plots are to be created of the parameters of interest. Default value is False. 

            Returns:
            --------
            None
            """

            # Open the tflow.xxx file and start writing the required input data to it
            file_path = self.submodels_path / "tflow.{}".format(self.case_name)
            with open(file_path, "w") as file:
                # Write the case name to the file
                file.write('NAME\n')
                file.write(f"{str(self.case_name)}\n")
                file.write('END\n \n')

                # Loop over the number of stages and write the data for each stage
                for stage in range(len(blading_params)):
                    # First write the "generic" data for the stage
                    # This includes the number of blades, the rotational rate, and the data types to be provided
                    # Formatting is in-line with the user guide from the MTFLOW documentation

                    # Start a stage block
                    file.write('STAGE\n \n')

                    # Write the number of blades
                    file.write('NBLADE\n')
                    file.write(str(blading_params[stage]["blade_count"]) + '\n')
                    file.write('END\n \n')

                    # Write the rotational rate in units of RPS * 2pi * L / V
                    file.write('OMEGA\n')
                    file.write(str(blading_params[stage]["rotational_rate"]) + '\n')
                    file.write('END\n \n')

                    # Write the data types to be provided for the stage
                    file.write('DATYPE \n')
                    file.write('x    r    T    Sr\n')  # Use the x,r coordinates, together with thickness and blade slope
                    multipliers = [1, 1, 1, 1]  # Add multipliers for each data type
                    additions = [0, 0, 0, 0]  # Add additions for each data type
                    file.write('*' + '    '.join(map(str, multipliers)) + '\n')
                    file.write('+' + '    '.join(map(str, additions)) + '\n')
                    file.write('END\n \n')

                    # Collect the blade geometry interpolations
                    blade_geometry: dict = self.ConstructBlades(blading_params[stage], 
                                                                design_params[stage])
                    
                    # Generate interpolated data to construct the file geometry
                    # The MTFLO code cannot accept an input file with more than 16x16 points in the streamwise and radial directions for each stage
                    # Hence n_points=16
                    # The axial points are spaced using a cosine spacing for increased resolution at the LE and TE
                    # The radial points are spaced using constant spacing. 
                    # Routine assumed at least 120 chord-wise points were used to construct the initial input curves from which the interpolants were constructed
                    n_points_axial = 16
                    n_points_radial = 16
                    n_data = 120
                    axial_points = (1 - np.cos(np.linspace(0, np.pi, n_data))) / 2
                    radial_points = np.linspace(blading_params[stage]["radial_stations"][0], 
                                                blading_params[stage]["radial_stations"][-1], 
                                                n_points_radial,
                                                )                   

                    # Loop over the radial points and construct the data for each radial point
                    # Each radial point is defined as a "section" within the input file
                    for i in range(len(radial_points)): 
                        # Create a section in the input file
                        file.write('SECTION\n')

                        # All parameters are normalised using the local chord length, so we need to obtain the local chord in order to obtain the dimensional parameters
                        local_chord = blade_geometry["chord_distribution"](radial_points[i])
                        axial_coordinates = axial_points * local_chord
                       
                        # Create complete airfoil representation from the camber and thickness distributions
                        camber_distribution = blade_geometry["camber_distribution"]((radial_points[i], axial_points)) * local_chord
                        thickness_distribution = blade_geometry["thickness_distribution"]((radial_points[i], axial_points)) * local_chord
                        upper_x, upper_y, lower_x, lower_y = AirfoilParameterization().ConvertBezier2AirfoilCoordinates(axial_coordinates, 
                                                                                                                        thickness_distribution, 
                                                                                                                        axial_coordinates, 
                                                                                                                        camber_distribution)

                        # Rotate the airfoil profile to the correct angle
                        # The blade pitch is defined with respect to the blade pitch angle at the reference radial station, and thus is corrected accordingly. 
                        blade_pitch = (blade_geometry["pitch_distribution"](radial_points[i]) + blading_params[stage]["ref_blade_angle"] - blading_params[stage]["reference_section_blade_angle"])
                        rotated_upper_x, rotated_upper_y, rotated_lower_x, rotated_lower_y  = self.RotateProfile(blade_pitch,
                                                                                                                 upper_x,
                                                                                                                 lower_x,
                                                                                                                 upper_y,
                                                                                                                 lower_y)

                        # Compute the local leading edge offset at the radial station from the provided interpolant
                        # Use it to offset the x-coordinates of the upper and lower surfaces to the correct position
                        LE_coordinate = blade_geometry["leading_edge_distribution"](radial_points[i])
                        rotated_upper_x += LE_coordinate
                        rotated_lower_x += LE_coordinate

                        # Transform the 2D planar airfoils into 3D cylindrical sections
                        y_section_upper, y_section_lower, y_camber, z_section_upper, z_section_lower, z_camber = self.PlanarToCylindrical(rotated_upper_y,
                                                                                                                                          rotated_lower_y,
                                                                                                                                          radial_points[i])
                        
                        # Compute the circumferential blade thickness
                        if radial_points[i] == 0:
                            # Handle the case at the centerline, where we define the thickness to be zero. 
                            circumferential_thickness = np.zeros_like(axial_points)
                        else:
                            circumferential_thickness = self.CircumferentialThickness(y_section_upper,
                                                                                      z_section_upper,
                                                                                      y_section_lower,
                                                                                      z_section_lower,
                                                                                      radial_points[i])
                            
                            # Perform check that thickness does not exceed limit of complete blockage (T=2pir/N)
                            # If thickness exceeds limit, raises a ValueError
                            self.ValidateBladeThickness(max(circumferential_thickness), radial_points[i], blading_params[stage]["blade_count"])

                        # Compute the blade slope in the m'-theta plane. 
                        # Uses the average of the upper and lower x-coordinates to evaluate against. 
                        x_points = (rotated_lower_x + rotated_upper_x) / 2  
                        blade_slope, m_prime, theta = self.GeometricBladeSlope(y_camber,
                                                                               x_points,
                                                                               z_camber)       

                        if plot:
                            self.plot_blade_data(stage,
                                                 radial_points[i],
                                                 x_points,
                                                 rotated_upper_x,
                                                 rotated_upper_y,
                                                 rotated_lower_x,
                                                 rotated_lower_y,
                                                 circumferential_thickness,
                                                 blade_slope,
                                                 m_prime,
                                                 theta)

                        # Compute the sampling indices for the axial points with higher densities near the LE and TE
                        angle = np.linspace(0, np.pi, n_points_axial)
                        # Use cosine distribution to concentrate points near ends
                        normalized_indices = (1 - np.cos(angle)) / 2
                        sampling_indices = np.floor(normalized_indices * (len(x_points) - 1)).astype(int)
                                                  
                        # Loop over the streamwise points and construct the data for each streamwise point
                        # Each data point consists of the data [x / Lref, r / Lref, T / Lref, Srel]
                        for j in range(n_points_axial):  
                            # Write data to row
                            row = np.array([round((x_points[sampling_indices][j]) / self.ref_length, 5),
                                            round(radial_points[i] / self.ref_length, 5),
                                            round(circumferential_thickness[sampling_indices][j] / self.ref_length, 5),
                                            round(blade_slope[sampling_indices][j], 5),
                                            ])
                            
                            # Write the row to the file
                            file.write('    '.join(map(str, row)) + '\n')

                        # End the radial section
                        file.write('END\n')
                    
                    if plot:
                        # Only display plots if the plot boolean is True
                        import matplotlib.pyplot as plt
                        plt.show() 
                     
                    # End the stage 
                    file.write('\nEND\n \n')
                # End the input file
                file.write('END\n')


if __name__ == "__main__":
    import time

    # Perform test generation of walls.xxx file using dummy inputs
    # Creates a dummy geometry for the centerbody and duct
    centre_body_coeff = {"b_0": 0., "b_2": 0., "b_8": 2.63935800e-02, "b_15": 7.62111322e-01, "b_17": 0, 'x_t': 0.2855061027842137, 'y_t': 0.07513718500645125, 'x_c': 0.5, 'y_c': 0, 'z_TE': -2.3750854491940602e-33, 'dz_TE': 0.0019396795056937765, 'r_LE': -0.01634872585955984, 'trailing_wedge_angle': 0.15684435833921387, 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 2, "Leading Edge Coordinates": (0.3, 0)}
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 2.5, "Leading Edge Coordinates": (0, 1.2)}
    n6409_coeff = {"b_0": 0.07979831, "b_2": 0.20013347, "b_8": 0.02901246, "b_15": 0.74993802, "b_17": 0.78496242, 'x_t': 0.30429947838135246, 'y_t': 0.0452171520304373, 'x_c': 0.4249653844429819, 'y_c': 0.06028051002570214, 'z_TE': -0.0003886462495685791, 'dz_TE': 0.0004425237127035188, 'r_LE': -0.009225474218611841, 'trailing_wedge_angle': 0.10293203348896998, 'trailing_camberline_angle': 0.21034003141636426, 'leading_edge_direction': 0.26559481057525414, "Chord Length": 2.5, "Leading Edge Coordinates": (0, 2)}
    
    starttime = time.time()
    call_class = fileHandling()
    call_class_MTSET = call_class.fileHandlingMTSET(centre_body_coeff, n6409_coeff, "test_case", 2)
    call_class_MTSET.GenerateMTSETInput()
    endtime = time.time()
    print("Execution of GenerateMTSETInput() took", endtime - starttime, "seconds")

    # Perform test generation of tflow.xxx file using dummy inputs
    # Creates an input file using 2 stages, a rotor and a stator
    blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 1., "ref_blade_angle": np.deg2rad(19), "reference_section_blade_angle": np.deg2rad(34), "blade_count": 18, "radial_stations": [0.1, 1.8], "chord_length": [0.2, 0.2], "sweep_angle":[np.pi/16, np.pi/16], "blade_angle": [np.pi / 3, np.pi / 3]},
                          {"root_LE_coordinate": 1., "rotational_rate": 0., "ref_blade_angle": np.deg2rad(19), "reference_section_blade_angle": np.deg2rad(34), "blade_count": 10, "radial_stations": [0.1, 1], "chord_length": [0.2, 0.2], "sweep_angle":[np.pi/8, np.pi/8], "blade_angle": [np.pi / 3, np.pi/8]}]
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    starttime = time.time()
    call_class_MTFLO = call_class.fileHandlingMTFLO("test_case", 2)
    call_class_MTFLO.GenerateMTFLOInput(blading_parameters, 
                                        design_parameters)
    endtime = time.time()
    print("Execution of GenerateMTFLOInput() took", endtime - starttime, "seconds")