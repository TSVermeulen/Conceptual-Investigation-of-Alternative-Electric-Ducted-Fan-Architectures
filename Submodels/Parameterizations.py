"""
Parameterizations
=================

Description
-----------
This module provides methods for calculating the profile coordinates (x,y) 
based on a Bezier-Parsec (BP) 3434 parameterization. It includes methods for 
calculating the leading edge and trailing edge thickness and camber distributions, 
obtaining reference thickness and camber distributions from a reference airfoil shape, 
and extracting key parameters from these distributions. 

Additionally, a non-linear least-squares minimization method is included to obtain the parameterization 
for a given input coordinate file. 

Classes
-------
AirfoilParameterization
    A class to calculate airfoil parameterizations using Bezier curves. This module provides a class for airfoil 
    parameterization using Bezier curves. Contains full parameterization, including least-squares estimation. 

Examples
--------
>>> inputfile = r'Test Airfoils\n0015.dat'
>>> call_class = AirfoilParameterization()
>>> coefficients = call_class.FindInitialParameterization(inputfile,
                                                          plot=True)

Notes
-----
The generation of a BP3434 parameterization from a reference airfoil is prone to errors/invalid parameterizations. 
It is important to (visually) check the obtained parameterization, and to confirm the control points fall within 0 <= x/c <= 1. 

When executing the file as a standalone, obtaining the initial parameterization will take anywhere between 30 second to a minute, 
depending on the closeness of the initial guess. 

References
----------
The BP 3434 parameterization is documented in https://www.sciencedirect.com/science/article/abs/pii/S0965997810000529.
The derivation of the method, including reasoning for each of the bounds/inputs, is included in the PhD Thesis of Tim Rogalsky:
"Acceleration of Differential Evolution for Aerodynamic Design", dated March 2004

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID 4995309
Version: 1.2
Date [dd-mm-yyyy]: 27-02-2025

Changelog:
- V1.1: Updated with comments from coderabbitAI. 
- V1.0: Adapted version of a parameterization using only the bezier coefficients to give an improved fit to the reference data. 
- V1.1: Updated docstring, and working implementation for symmetric profiles (i.e. zero camber)
- V1.2: Updated FindInitialParameterization method to use SLSQP optimization rather than least squares to enable correct constraint handling. 
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import interpolate, optimize

class AirfoilParameterization:
    """
    This class calculates airfoil parameterizations using Bezier curves.
    
    It provides methods to calculate the leading edge and trailing edge thickness and camber distributions using Bezier curves. 
    It also includes methods to obtain reference thickness and camber distributions from a reference airfoil shape and to extract 
    key parameters from these distributions.
    
    Methods
    -------
    - BezierCurve3(coeff, u)
        Calculate a 3rd degree Bezier curve.
    - BezierCurve4(coeff, u)
        Calculate a 4th degree Bezier curve.
    - GetCamberAngleDistribution(X, Y)
        Calculate the camber angle distribution over the length of the airfoil.
    - GetReferenceThicknessCamber(reference_file)
        Obtain the thickness and camber distributions from the reference airfoil shape.
    - GetReferenceParameters()
        Extract key parameters of the reference profile from the thickness and camber distributions.
    - GetThicknessControlPoints(b_8, b_15, r_LE, trailing_wedge_angle)
        Calculate the control points for the thickness distribution Bezier curves.
    - GetCamberControlPoints(b_0, b_2, b_17, leading_edge_direction, trailing_camberline_angle)
        Calculate the control points for the camber distribution Bezier curves.
    - ConvertBezier2AirfoilCoordinates(thickness_x, thickness, camber_x, camber)
        Convert Bezier curves to airfoil coordinates.
    - FindInitialParameterization(reference_file)
        Find the initial parameterization for the airfoil.
    """

    def __init__(self,
                 symmetric_limit: float = 1E-3) -> None:
        """
        Initialize the AirfoilParameterization class.
        
        This method sets up the initial state of the class.

        Parameters
        ----------
        - symmetric_limit : float, optional
            The triggering value of camber below which an airfoil is treated as being symmetric. Default is 1E-3. This is needed to avoid issues with cotangent calculations. 

        Returns
        -------
        None
        """

        # Define the minimum camber level which triggers handling the airfoil profile as being symmetric
        self.symmetric_limit = symmetric_limit
        

    def BezierCurve3(self,
                     coefficients: list[float], 
                     u: float,
                     ) -> np.ndarray|float:
        """
        Calculate a 3rd degree Bezier curve.

        Parameters
         ----------
        coefficients : list[float]
            List of 4 control points for the Bezier curve.
        u : float
            Parameter ranging from 0 to 1.

        Returns
        -------
        y : float
            Value of the Bezier curve at parameter u.
        """

        #Input checking
        if len(coefficients) != 4:
            raise ValueError(f"Coefficient list must contain exactly 4 elements. Coefficient list contains {len(coefficients)} elements")
    
        # Calculate the value of y at u using a 3rd degree Bezier curve
        return coefficients[0] * (1 - u) ** 3 + 3 * coefficients[1] * u * (1 - u) ** 2 + 3 * coefficients[2] * u ** 2 * (1 - u) + coefficients[3] * u ** 3
        

    def BezierCurve4(self,
                     coefficients: list[float], 
                     u: float,
                     ) -> np.ndarray|float:
        """
        Calculate a 4th degree Bezier curve.

        Parameters
        ----------
        coefficients : list[float]
            List of 5 control points for the Bezier curve.
        u : float
            Parameter ranging from 0 to 1.

        Returns
        -------
        y : float
            Value of the Bezier curve at parameter u.
        """

        # Input checking
        if len(coefficients) != 5:
            raise ValueError(f"Coefficient list must contain exactly 5 elements. Coefficient list contains {len(coefficients)} elements.")     

        return coefficients[0] * (1 - u) ** 4 + 4 * coefficients[1] * u * (1 - u) ** 3 + 6 * coefficients[2] * u ** 2 * (1 - u) ** 2 + 4 * coefficients[3] * u ** 3 * (1 - u) + coefficients[4] * u ** 4


    def GetCamberAngleDistribution(self,
                                   x: np.ndarray|float,
                                   y: np.ndarray|float,
                                   ) -> np.ndarray|float:
        """
        Calculate the camber angle distribution over the length of the airfoil.

        Parameters
        ----------
        x : np.ndarray
            Array of x-coordinates along the airfoil.
        y : np.ndarray
            Array of camber values corresponding to the x-coordinates.

        Returns
        -------
        theta : np.ndarray
            Array of camber gradient angles at each x-coordinate.
        """

        camber_gradient = np.gradient(y, x)
        

        return np.arctan(camber_gradient)
    

    def GetReferenceThicknessCamber(self, 
                                    reference_file: str,
                                    ) -> None:
        """
        Obtain the thickness and camber distributions from the reference profile shape.

        Parameters
        ----------
        reference_file : str
            Path to the file containing the reference profile coordinates.
        
        Returns
        -------
        None
        """    

        # Load in the reference profile shape from the reference_file. 
        # Assumes coordinates are sorted counter clockwise from TE of the Upper
        # surface.
        # Profile shape must be provided in input file with unit chord length
        # Skips the first row of the profile file as it contains the profile name
        try:
            reference_coordinates = np.genfromtxt(reference_file, dtype=float, skip_header=1)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"The data input file {reference_file} does not exist in the current working directory.") from err
         

        # Find index of LE coordinate and compute the camber and thickness distributions. 
        #If number of coordinate points is even, use the middle of the array as the LE coordinate
        if len(reference_coordinates[:,0]) % 2 == 0:  
            self.idx_LE = len(reference_coordinates[:,0]) // 2
        else:  # LE coordinate index must be the index for x,y = 0,0.
            self.idx_LE = (np.abs(reference_coordinates[:,0] - 0)).argmin() 

        # Calculate camber x & y, and slope of the camber line
        x_camber = (np.flip(reference_coordinates[:self.idx_LE + 1, 0]) + reference_coordinates[self.idx_LE:, 0]) / 2
        y_camber = (np.flip(reference_coordinates[:self.idx_LE + 1, 1]) + reference_coordinates[self.idx_LE:, 1]) / 2
        theta = self.GetCamberAngleDistribution(x_camber,
                                                y_camber)  # Calculate gradient of camber
        
        # Calculate thickness distribution
        y_thickness = (np.flip(reference_coordinates[:self.idx_LE + 1, 1]) - reference_coordinates[self.idx_LE:, 1]) / (2 * np.cos(theta))    

        # Calculate x-coordinates of thickness distribution
        x_upper = x_camber - y_thickness * np.sin(theta)

        # Write parameters to self
        self.x_points_camber = x_camber
        self.x_points_thickness = x_upper
        self.camber_distribution = y_camber
        self.camber_gradient_distribution = theta
        self.thickness_gradient_distribution = self.GetCamberAngleDistribution(x_upper,
                                                                               y_thickness)
        self.thickness_distribution = y_thickness
        self.reference_data = reference_coordinates
    

    def GetReferenceParameters(self) -> dict:
        """
        Extract key parameters of the reference profile from the thickness and camber distributions.

        Calculates the leading edge radius, leading edge direction, 
        trailing edge wedge angle, and trailing edge camber line angle.
        
        Parameters
        ----------
        None

        Returns
        -------
        output_dict: dict
            Dictionary containing the following parameters:
                x_t : float
                    X-coordinate of maximum thickness.
                y_t : float
                    Maximum thickness.
                x_c : float
                    X-coordinate of maximum camber.
                y_c : float
                    Maximum camber.
                z_TE : float
                    Trailing edge vertical displacement.
                dz_TE : float
                    Half thickness of the trailing edge.    
                r_LE : float
                    Leading edge radius.
                leading_edge_direction : float
                    Angle of the leading edge direction.
                trailing_wedge_angle : float
                    Trailing edge wedge angle.
                trailing_camberline_angle : float
                    Trailing edge camber line angle.
        """

        # Find the indices for the points of maximum thickness and maximum camber
        self.idx_maxT = np.where(self.thickness_distribution == np.max(self.thickness_distribution))[0][0]
        self.idx_maxC = np.where(self.camber_distribution == np.max(self.camber_distribution))[0][0]
        
        x_t = self.x_points_thickness[self.idx_maxT]  # X-coordinate of maximum thickness
        y_t = self.thickness_distribution[self.idx_maxT]  # Maximum thickness
        x_c = self.x_points_camber[self.idx_maxC]  # X-coordinate of maximum camber
        y_c = self.camber_distribution[self.idx_maxC]  # Maximum camber
        z_TE = self.camber_distribution[-1]  # Trailing edge vertical displacement
        dz_TE = self.thickness_distribution[-1]  # Half thickness of the trailing edge

        # To find the radius of curvature of the leading edge, we fit a circle to the points within the first 2 % of the upper surface      
        def GetLeadingEdgeRadius(params: tuple[float, float, float], 
                                 x: np.ndarray, 
                                 y: np.ndarray) -> np.ndarray:
            xc, yc, r = params
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r
        
        LE_points_idx = np.where(self.x_points_thickness < 0.02)[0]
        LE_x = self.x_points_thickness[:LE_points_idx[-1] + 1]
        LE_y = self.thickness_distribution[:LE_points_idx[-1] + 1]
        guess = [0, 0, 0]

        # Perform least squares fitting of circle to data points
        result = optimize.least_squares(GetLeadingEdgeRadius, guess, args=(LE_x, LE_y))
        r_LE = -1 * result.x[-1]  # LE radius multiplied by -1 to comply with sign convention of method

        # Calculate the leading edge direction gamma_LE, trailing wedge angle beta_TE, and trailing camber line angle alpha_TE
        leading_edge_direction = np.atan(self.camber_gradient_distribution[0])
        trailing_wedge_angle = np.atan(-1 * self.thickness_gradient_distribution[-1])  # Multiply gradient by -1 to comply with sign convention
        trailing_camberline_angle = np.atan(-1 * self.camber_gradient_distribution[-1])  # Multiply gradient by -1 to comply with sign convention

        # Return output dictionary
        return {
            "x_t": x_t,
            "y_t": y_t,
            "x_c": x_c,
            "y_c": y_c,
            "z_TE": z_TE,
            "dz_TE": dz_TE,
            "r_LE": r_LE,
            "leading_edge_direction": leading_edge_direction,
            "trailing_wedge_angle": trailing_wedge_angle,
            "trailing_camberline_angle": trailing_camberline_angle,
        }


    def GetThicknessControlPoints(self,
                                  b_8: float,
                                  b_15: float,
                                  airfoil_params: dict,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Calculate the control points for the thickness distribution Bezier curves.

        Parameters
        ----------
        b_8 : float
            Control point for the thickness curve.
        b_15 : float
            Control point for the thickness curve.
        airfoil_params : dict
            Dictionary containing the following parameters:
                x_t : float
                    X-coordinate of maximum thickness.
                y_t : float
                    Maximum thickness.
                r_LE : float
                    Leading edge radius.
                dz_TE : float
                    Half thickness of the trailing edge.
                trailing_wedge_angle : float
                    The trailing edge wedge angle.

        Returns
        -------
        x_leading_edge_thickness_coeff : np.ndarray
            X-coordinates of the control points for the leading edge thickness Bezier curve.
        y_leading_edge_thickness_coeff : np.ndarray
            Y-coordinates of the control points for the leading edge thickness Bezier curve.
        x_trailing_edge_thickness_coeff : np.ndarray
            X-coordinates of the control points for the trailing edge thickness Bezier curve.
        y_trailing_edge_thickness_coeff : np.ndarray
            Y-coordinates of the control points for the trailing edge thickness Bezier curve.
        """
        
        # Construct leading edge x coefficients
        x_leading_edge_thickness_coeff = np.array([0,
                                                   0,
                                                   (-3 * b_8 ** 2) / (2 * airfoil_params["r_LE"]),
                                                   airfoil_params["x_t"]])
        
        # Construct leading edge y coefficients
        y_leading_edge_thickness_coeff = np.array([0,
                                                   b_8,
                                                   airfoil_params["y_t"],
                                                   airfoil_params["y_t"]])
        
        # Contruct trailing edge x coefficients
        x_trailing_edge_thickness_coeff = np.array([airfoil_params["x_t"],
                                                    (7 * airfoil_params["x_t"] + (9 * b_8 ** 2) / (2 * airfoil_params["r_LE"])) / 4,
                                                    3 * airfoil_params["x_t"] + (15 * b_8 ** 2) / (4 * airfoil_params["r_LE"]),
                                                    b_15,
                                                    1])
        
        # Construct trailing edge y coefficients
        y_trailing_edge_thickness_coeff = np.array([airfoil_params["y_t"],
                                                    airfoil_params["y_t"],
                                                    (airfoil_params["y_t"] + b_8) / 2,
                                                    airfoil_params["dz_TE"] + (1 - b_15) * np.tan(airfoil_params["trailing_wedge_angle"]),
                                                    airfoil_params["dz_TE"]])
    
        return x_leading_edge_thickness_coeff, y_leading_edge_thickness_coeff, x_trailing_edge_thickness_coeff, y_trailing_edge_thickness_coeff


    def GetCamberControlPoints(self,
                               b_0: float,
                               b_2: float,
                               b_17: float,
                               airfoil_params: dict,
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the control points for the camber distribution Bezier curves.

        Parameters
        ----------
        b_0 : float
            Control point for the camber.
        b_2 : float
            Control point for the camber.
        b_17 : float
            Control point for the camber.
        airfoil_params : dict
            Dictionary containing the following parameters:
                x_c : float
                    X-coordinate of maximum camber.
                y_c : float
                    Maximum camber.
                z_TE : float
                    Trailing edge vertical displacement.
                leading_edge_direction : float
                    Angle of the leading edge direction.
                trailing_camberline_angle : float
                    Angle of the trailing edge camber line.

        Returns
        -------
        x_leading_edge_camber_coeff : np.ndarray
            X-coordinates of the control points for the leading edge camber Bezier curve.
        y_leading_edge_camber_coeff : np.ndarray
            Y-coordinates of the control points for the leading edge camber Bezier curve.
        x_trailing_edge_camber_coeff : np.ndarray
            X-coordinates of the control points for the trailing edge camber Bezier curve.
        y_trailing_edge_camber_coeff : np.ndarray
            Y-coordinates of the control points for the trailing edge camber Bezier curve.
        """

        # Construct leading edge x coefficients
        x_leading_edge_camber_coeff = np.array([0,
                                                b_0,
                                                b_2,
                                                airfoil_params["x_c"]])
        
        # Construct leading edge y coefficients
        y_leading_edge_camber_coeff = np.array([0,
                                                b_0 * np.tan(airfoil_params["leading_edge_direction"]),
                                                airfoil_params["y_c"],
                                                airfoil_params["y_c"]])
        
        # Construct trailing edge x coefficients
        x_trailing_edge_camber_coeff = np.array([airfoil_params["x_c"],
                                                 (3 * airfoil_params["x_c"] - airfoil_params["y_c"] / np.tan(airfoil_params["leading_edge_direction"])) / 2,
                                                 (-8 * airfoil_params["y_c"] / np.tan(airfoil_params["leading_edge_direction"]) + 13 * airfoil_params["x_c"]) / 6,
                                                 b_17,
                                                 1])
        
        # Construct trailing edge y coefficients
        y_trailing_edge_camber_coeff = np.array([airfoil_params["y_c"],
                                                 airfoil_params["y_c"],
                                                 5 * airfoil_params["y_c"] / 6,
                                                 airfoil_params["z_TE"] + (1 - b_17) * np.tan(airfoil_params["trailing_camberline_angle"]),
                                                 airfoil_params["z_TE"]])
        
        return x_leading_edge_camber_coeff, y_leading_edge_camber_coeff, x_trailing_edge_camber_coeff, y_trailing_edge_camber_coeff


    def ConvertBezier2AirfoilCoordinates(self,
                                         thickness_x: np.ndarray[float],
                                         thickness: np.ndarray[float],
                                         camber_x: np.ndarray[float],
                                         camber: np.ndarray[float],
                                         ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        Convert Bezier curves to airfoil coordinates.

        Parameters
        ----------
        thickness_x : np.ndarray[float]
            Array of x-coordinates for the thickness distribution.
        thickness : np.ndarray[float]
            Array of thickness values corresponding to the x-coordinates.
        camber_x : np.ndarray[float]
            Array of x-coordinates for the camber distribution.
        camber : np.ndarray[float]
            Array of camber values corresponding to the x-coordinates.

        Returns
        -------
        upper_x : np.ndarray[float]
            Array of x-coordinates for the upper surface of the airfoil.
        upper_y : np.ndarray[float]
            Array of y-coordinates for the upper surface of the airfoil.
        lower_x : np.ndarray[float]
            Array of x-coordinates for the lower surface of the airfoil.
        lower_y : np.ndarray[float]
            Array of y-coordinates for the lower surface of the airfoil.
        """

        # Handle symmetric airfoil case by differentiating between cambered and non-cambered cases
        if np.any(camber >= self.symmetric_limit): 
            thickness_distribution = np.zeros(len(thickness_x))
            thickness_interpolation = interpolate.CubicSpline(thickness_x, thickness)  # Interpolation of bezier thickness distribution
            for i in range(len(thickness_x)):
                thickness_distribution[i] = thickness_interpolation(camber_x[i])  # Use interpolation to get value of thickness at camber point

            # Calculate gradient of camber angle line
            theta = self.GetCamberAngleDistribution(camber_x, camber)

            # Coordinate transformation to create data for upper and lower surface
            upper_x = camber_x - thickness_distribution * np.sin(theta)
            upper_y = camber + thickness_distribution * np.cos(theta)
            lower_x = camber_x + thickness_distribution * np.sin(theta)
            lower_y = camber - thickness_distribution * np.cos(theta)

        else:
            upper_x = thickness_x
            upper_y = thickness
            lower_x = thickness_x
            lower_y = -thickness

        return upper_x, upper_y, lower_x, lower_y
    

    def GenerateBezierUVectors(self,
                               ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Create u-vectors for Bezier curve generation. Uses 100 points for the leading and trailing edges.

        Returns
        -------
        u_leading_edge : np.ndarray[float]
            Array of u-values for the leading edge Bezier curve.
        u_trailing_edge : np.ndarray[float]
            Array of u-values for the trailing edge Bezier curve.
        """

        # Create u-vectors for Bezier curve generation
        # Use 100 points
        n_points = 100
        u_leading_edge = np.zeros(n_points)
        u_trailing_edge = np.zeros(n_points)

        for i in range(n_points):
            u_leading_edge[i] = np.abs(1 - np.cos((i * np.pi) / (2 * (n_points - 1))))  # Space points using a cosine spacing for increased resolution at LE
            u_trailing_edge[i] = np.abs(np.sin((i * np.pi) / (2 * (n_points - 1))))  # Space points using a sine spacing for increased resolution at TE

        return u_leading_edge, u_trailing_edge


    def ComputeBezierCurves(self,
                            b_coeff: np.ndarray[float],
                            airfoil_params: dict,
                            ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """ 
        Calculate the thickness and camber Bezier distributions.

        Parameters
        ----------
        b_coeff : np.ndarray[float]
            Array of Bezier control points.
        airfoil_params : dict
            Dictionary containing airfoil parameters.

        Returns
        -------
        bezier_thickness : np.ndarray[float]
            Array of thickness values along the airfoil.
        bezier_thickness_x : np.ndarray[float]
            Array of x-coordinates corresponding to the thickness values.
        bezier_camber : np.ndarray[float]
            Array of camber values along the airfoil.
        bezier_camber_x : np.ndarray[float]
            Array of x-coordinates corresponding to the camber values.
        """

        # Extract the bezier coefficients from the input array
        b_0 = b_coeff[0]
        b_2 = b_coeff[1]
        b_8 = b_coeff[2]
        b_15 = b_coeff[3]
        b_17 = b_coeff[4]
        
        # Create Bezier U-vectors
        u_leading_edge, u_trailing_edge = self.GenerateBezierUVectors()

        # Calculate the Bezier curve coefficients for the thickness curves
        x_LE_thickness_coeff, y_LE_thickness_coeff, x_TE_thickness_coeff, y_TE_thickness_coeff = self.GetThicknessControlPoints(b_8, 
                                                                                                                                b_15,
                                                                                                                                airfoil_params)
        
        # Calculate the leading edge thickness distribution
        y_LE_thickness = self.BezierCurve3(y_LE_thickness_coeff, 
                                           u_leading_edge)  # Leading edge thickness represented by 3rd order Bezier curve
        x_LE_thickness = self.BezierCurve3(x_LE_thickness_coeff,
                                           u_leading_edge)  # Leading edge thickness bezier x-coordinates, represented by a 3rd order curve
        
        # Calculate the trailing edge thickness distribution
        y_TE_thickness = self.BezierCurve4(y_TE_thickness_coeff,
                                           u_trailing_edge[1:])  # Trailing edge thickness represented by 4th order Bezier curve      
        x_TE_thickness = self.BezierCurve4(x_TE_thickness_coeff,
                                           u_trailing_edge[1:])  # Trailing edge bezier x-coordinates, represented by 4th order curve    

        # Construct full curves by combining LE and TE data
        bezier_thickness = np.concatenate((y_LE_thickness, y_TE_thickness), 
                                          axis = 0)  # Construct complete thickness curve over length of profile
        bezier_thickness_x = np.concatenate((x_LE_thickness, x_TE_thickness), 
                                            axis = 0)  # Construct complete array of x-coordinates over length of profile
        
        # Calculate the camber distributions only if the camber is nonzero
        if airfoil_params["y_c"] >= self.symmetric_limit:
            # Calculate the Bezier curve coefficients for the camber curves
            x_LE_camber_coeff, y_LE_camber_coeff, x_TE_camber_coeff, y_TE_camber_coeff = self.GetCamberControlPoints(b_0,
                                                                                                                    b_2,
                                                                                                                    b_17,
                                                                                                                     airfoil_params)

            # Calculate the leading edge camber distribution            
            y_LE_camber = self.BezierCurve3(y_LE_camber_coeff, 
                                            u_leading_edge)  # Leading edge camber represented by 3rd order Bezier curve
            x_LE_camber = self.BezierCurve3(x_LE_camber_coeff,
                                            u_leading_edge)  # Leading edge camber bezier x-coordinates, represented by a 3rd order curve

            # Calculate the trailing edge camber distribution using the parameter b_17
            y_TE_camber = self.BezierCurve4(y_TE_camber_coeff, 
                                            u_trailing_edge[1:])  # Trailing edge camber represented by 4th order Bezier curve
            x_TE_camber = self.BezierCurve4(x_TE_camber_coeff,
                                        u_trailing_edge[1:])  # Trailing edge camber bezier x-coordinates, represented by a 4th order curve
            
            bezier_camber = np.concatenate((y_LE_camber, y_TE_camber),
                                           axis = 0)  # Construct complete camber curve over length of profile
            bezier_camber_x = np.concatenate((x_LE_camber, x_TE_camber),
                                             axis = 0)  # Construct complete array of x-coordinates over length of profile
        
        else:
            # If camber is zero, handle appropriately
            bezier_camber = np.zeros_like(bezier_thickness)
            bezier_camber_x = bezier_thickness_x
        
        return bezier_thickness, bezier_thickness_x, bezier_camber, bezier_camber_x
    

    def ComputeProfileCoordinates(self,
                                  b_coeff: np.ndarray[float],
                                  airfoil_params: dict,
                                  ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        Calculate the airfoil coordinates from the Bezier control points.

        Parameters
        ----------
        b_coeff : np.ndarray[float]
            Array of Bezier control points.
        airfoil_params : dict
            Dictionary containing airfoil parameters.

        Returns
        -------
        upper_x : np.ndarray[float]
            Array of x-coordinates for the upper surface of the airfoil.
        upper_y : np.ndarray[float]
            Array of y-coordinates for the upper surface of the airfoil.
        lower_x : np.ndarray[float]
            Array of x-coordinates for the lower surface of the airfoil.
        lower_y : np.ndarray[float]
            Array of y-coordinates for the lower surface of the airfoil.
        """

        # Obtain the Bezier data 
        bezier_thickness, bezier_thickness_x, bezier_camber, bezier_camber_x = self.ComputeBezierCurves(b_coeff,
                                                                                                        airfoil_params,
                                                                                                        )
            
        # Calculate the upper and lower surface coordinates from the bezier coordinates
        upper_x, upper_y, lower_x, lower_y = self.ConvertBezier2AirfoilCoordinates(bezier_thickness_x,
                                                                                   bezier_thickness,
                                                                                   bezier_camber_x,
                                                                                   bezier_camber)

        return upper_x, upper_y, lower_x, lower_y


    def CheckOptimizedResult(self,
                             airfoil_params: dict,
                             ) -> None:
        """
        Check the optimized result by plotting the thickness and camber distributions, and the airfoil shape.

        Parameters
        ----------
        airfoil_params : dict
            A dictionary containing the airfoil parameterization parameters. 

        Returns
        -------
        None
        """

        # Construct b coeff array for input into self.ComputeBezierCurves
        b_coeff = [airfoil_params["b_0"],
                   airfoil_params["b_2"],
                   airfoil_params["b_8"],
                   airfoil_params["b_15"],
                   airfoil_params["b_17"],
                   ]
        
        # Obtain bezier curves
        bezier_thickness, bezier_thickness_x, bezier_camber, bezier_camber_x = self.ComputeBezierCurves(b_coeff,
                                                                                                        airfoil_params)                                                               

        # Compute upper and lower surface coordinates
        upper_x, upper_y, lower_x, lower_y = self.ConvertBezier2AirfoilCoordinates(bezier_thickness_x, 
                                                                                   bezier_thickness,
                                                                                   bezier_camber_x,
                                                                                   bezier_camber)
        
        # Calculate bezier coefficients for the thickness curves for plotting
        x_LE_thickness_coeff, y_LE_thickness_coeff, x_TE_thickness_coeff, y_TE_thickness_coeff = self.GetThicknessControlPoints(airfoil_params["b_8"],
                                                                                                                                airfoil_params["b_15"],
                                                                                                                                airfoil_params)
        try:
            #Create plots of the thickness distribution compared to the input data
            plt.figure("Thickness Distributions")
            plt.plot(bezier_thickness_x, bezier_thickness, label="BezierThickness")
            plt.plot(x_LE_thickness_coeff, y_LE_thickness_coeff, '*', color='k', label="Bezier Coefficients")
            plt.plot(x_TE_thickness_coeff, y_TE_thickness_coeff, '*', color='k')  # Do not label this line to avoid duplicate legend entry            
            plt.plot(self.x_points_thickness, self.thickness_distribution, "-.", label="ThicknessInputData")
            plt.xlabel("x/c [-]")
            plt.ylabel("y_t/c [-]")
            plt.legend()

            if airfoil_params["y_c"] >= self.symmetric_limit:
                # Calculate bezier coefficients for the camber curves for plotting
                x_LE_camber_coeff, y_LE_camber_coeff, x_TE_camber_coeff, y_TE_camber_coeff = self.GetCamberControlPoints(airfoil_params["b_0"],
                                                                                                                         airfoil_params["b_2"],
                                                                                                                         airfoil_params["b_17"],
                                                                                                                         airfoil_params)

                plt.figure("Camber Distributions")
                plt.plot(bezier_camber_x, bezier_camber, label="BezierCamber")
                plt.plot(x_LE_camber_coeff, y_LE_camber_coeff, '*', color='k', label="Bezier Coefficients")
                plt.plot(x_TE_camber_coeff, y_TE_camber_coeff, '*', color='b')  # Do not label this line to avoid duplicate legend entry
                plt.plot(self.x_points_camber, self.camber_distribution, "-.", label="CamberInputData")
                plt.xlabel("x/c [-]")
                plt.ylabel("y_c/c [-]")
                plt.legend()

            plt.figure("Airfoil Shape")
            plt.plot(upper_x, upper_y, label="Reconstructed Upper Surface")
            plt.plot(lower_x, lower_y, label="Reconstructed Lower Surface")
            plt.plot(self.reference_data[:self.idx_LE + 1, 0], self.reference_data[:self.idx_LE + 1, 1], "-.", color="g")
            plt.plot(self.reference_data[self.idx_LE:, 0], self.reference_data[self.idx_LE:, 1], "-.", color="g")
            plt.xlabel("x/c [-]")
            plt.ylabel("y/c [-]")
            plt.show()

        except Exception as e:
            print(f"Warning: Failed to generate plots: {str(e)}")

    
    def Objective(self,
                  x: list[float],
                      ) -> float:
        """
        Objective function for least-squares minimization.

        Parameters
        ----------
        x : list[float]
            List of normalised design variables [b_0, b_2, b_8, b_15, b_17, x_t, y_t, x_c, y_c, r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction].
            These are denormalised using the guess design vector given by self.guess_design_vector. 

        Returns
        -------
        squared_fit_error : float
            Sum of squared fit errors of the upper and lower surfaces.
        """
        
        # Denormalise design vector
        x = np.multiply(self.guess_design_vector, x)

        # Split out the design vector into a bezier coefficient list and airfoil parameters dictionary.
        b_coeff = x[0:5]
        airfoil_params = {"b_0": x[0],
                          "b_2": x[1],
                          "b_8": x[2],
                          "b_15": x[3],
                          "b_17": x[4],
                          "x_t": x[5],
                          "y_t": x[6],
                          "x_c": x[7],
                          "y_c": x[8],
                          "z_TE": x[9],
                          "dz_TE": x[10],
                          "r_LE": x[11],
                          "trailing_wedge_angle": x[12],
                          "trailing_camberline_angle": x[13],
                          "leading_edge_direction": x[14]}       

        # Compute the upper and lower surface coordinates from the parameterization
        upper_x, upper_y, lower_x, lower_y = self.ComputeProfileCoordinates(b_coeff,
                                                                            airfoil_params)

        # Create interpolation of upper and lower surfaces to ensure we take data from same x-coordinates
        interpolated_upper_surface_data = interpolate.CubicSpline(np.flip(self.reference_data[:self.idx_LE + 1, 0]), 
                                                                  np.flip(self.reference_data[:self.idx_LE + 1, 1])
                                                                  )(upper_x)
            
        interpolated_lower_surface_data = interpolate.CubicSpline(self.reference_data[self.idx_LE:, 0], 
                                                                  self.reference_data[self.idx_LE:, 1])(lower_x)
            
        # Calculate the squared fit errors of the upper and lower surfaces and sum them to obtain the objective function
        squared_fit_error_upper_surface = np.sum((upper_y - interpolated_upper_surface_data) ** 2)
        squared_fit_error_lower_surface = np.sum((lower_y - interpolated_lower_surface_data) ** 2)
        
        # Objective is multiplied by 10 to ensure more weight is put on the error.
        return (squared_fit_error_upper_surface + squared_fit_error_lower_surface) * 10
        

    def FindInitialParameterization(self, 
                                    reference_file: str,
                                    *,
                                    plot: bool = False) -> dict:
        """
        Find the initial parameterization for the profile.
        Uses the SLSQP algorithm to minimize the squared fit error between the input reference file and the reconstructed profile shape. 

        Parameters
        ----------
        reference_file : str
            Path to the file containing the reference profile coordinates.
        plot : bool
            Boolean controlling if the initial parameterization is plotted. Default is False.

        Returns
        -------
        dict
            Dictionary containing the optimized airfoil parameters.
        """    

        def GetBounds() -> optimize.Bounds:
            """
            Get the bounds for the optimization problem. Note that the bounds are given in normalised form. 


            Returns
            -------
            - optimize.Bounds()
                A scipy.optimize.bounds instance of the bounds for the optimization problem. 
            """

            return optimize.Bounds(0.9, 1.1, keep_feasible=True)

        # Load in the reference profile shape and obtain the relevant parameters
        self.GetReferenceThicknessCamber(reference_file)
        self.airfoil_params = self.GetReferenceParameters()

        # Define a guess of the initial design vector
        self.guess_design_vector = np.array([0.1 * self.airfoil_params["x_c"],
                                            0.5 * self.airfoil_params["x_c"],
                                            0.7 * min(self.airfoil_params["y_t"], np.sqrt(-2 * self.airfoil_params["r_LE"] * self.airfoil_params["x_t"] / 3)),
                                            0.75,
                                            0.8,
                                            self.airfoil_params["x_t"],
                                            self.airfoil_params["y_t"],
                                            self.airfoil_params["x_c"],
                                            self.airfoil_params["y_c"],
                                            self.airfoil_params["z_TE"],
                                            self.airfoil_params["dz_TE"],
                                            self.airfoil_params["r_LE"],
                                            self.airfoil_params["trailing_wedge_angle"],
                                            self.airfoil_params["trailing_camberline_angle"],
                                            self.airfoil_params["leading_edge_direction"],
                                            ])
        
        # Define nonlinear constraint for the leading edge direction:
        def leading_edge_direction_constraint(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            if x[14] != 0:
                return x[14] - np.arctan(8 * x[8] / (7 * x[7]))
            else:
                return x[14]

        # Define nonlinear constraint for b8
        def b8_constraint(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return np.min([x[6], np.sqrt(-2 * x[11] * x[5] / 3)]) - x[2]
        
        # Define constraint for bezier control point 1 x-coordinate of TE thickness curve
        def x1_constraint_lower_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return (7 * x[5] + 9 * x[2] / (2 * x[11])) / 4 - x[5]
        def x1_constraint_upper_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - (7 * x[5] + 9 * x[2] / (2 * x[11])) / 4
        
        # Define constraint for bezier control point 2 x-coordinate of TE thickness curve
        def x2_constraint_lower_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 2 * x[5] + 15 * x[2] ** 2 / (4 * x[11])
        def x2_constraint_upper_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - 2 * x[5] + 15 * x[2] ** 2 / (4 * x[11])
        
        # Define constraint for bezier control point 2 x-coordinate of TE camber curve
        def constraint_6_lower(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return (3 * x[7] - x[8] / np.tan(x[14])) / 2 - x[7]

        def constraint_6_upper(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - (3 * x[7] - x[8] / np.tan(x[14])) / 2
                
        cons = [{'type': 'ineq', 'fun': leading_edge_direction_constraint},
                {'type': 'ineq', 'fun': b8_constraint},
                {'type': 'ineq', 'fun': x1_constraint_lower_thickness},
                {'type': 'ineq', 'fun': x1_constraint_upper_thickness},
                {'type': 'ineq', 'fun': x2_constraint_lower_thickness},
                {'type': 'ineq', 'fun': x2_constraint_upper_thickness},
                {'type': 'ineq', 'fun': constraint_6_lower},
                {'type': 'ineq', 'fun': constraint_6_upper}]
        
        optimized_coefficients = optimize.minimize(self.Objective,
                                 np.ones(15),
                                 method="SLSQP",
                                 bounds=GetBounds(),
                                 constraints=cons,
                                 options={'maxiter': 100,
                                          'disp': True},
                                          jac='3-point')
        
        # Denormalise the found coefficients and write them to the output dictionary
        optimized_coefficients.x = optimized_coefficients.x.astype(float)
        optimized_coefficients.x = np.multiply(optimized_coefficients.x, self.guess_design_vector)
        
        airfoil_params_optimized = {"b_0": optimized_coefficients.x[0],
                                    "b_2": optimized_coefficients.x[1],
                                    "b_8": optimized_coefficients.x[2],
                                    "b_15": optimized_coefficients.x[3],
                                    "b_17": optimized_coefficients.x[4],
                                    "x_t": optimized_coefficients.x[5],
                                    "y_t": optimized_coefficients.x[6],
                                    "x_c": optimized_coefficients.x[7],
                                    "y_c": optimized_coefficients.x[8],
                                    "z_TE": optimized_coefficients.x[9],
                                    "dz_TE": optimized_coefficients.x[10],
                                    "r_LE": optimized_coefficients.x[11],
                                    "trailing_wedge_angle": optimized_coefficients.x[12],
                                    "trailing_camberline_angle": optimized_coefficients.x[13],
                                    "leading_edge_direction": optimized_coefficients.x[14]}

        # Generate plots if requested
        if plot:
            self.CheckOptimizedResult(airfoil_params_optimized)

        return airfoil_params_optimized


if __name__ == "__main__":
    import time
    call_class = AirfoilParameterization()
    
    start_time = time.time()
    inputfile = Path('Test Airfoils') / 'X22_root.dat'
    airf_params = call_class.FindInitialParameterization(inputfile,
                                                         plot=False)
    end_time = time.time()
    print(f"Execution of FindInitialParameterization({inputfile}) took {end_time-start_time} seconds")
    print("-----")
    print(airf_params)

    test = call_class.CheckOptimizedResult(airf_params)
    
    




