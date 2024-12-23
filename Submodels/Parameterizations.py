"""

This module provides a class for airfoil parameterization using Bezier curves.
Contains full parameterization, including least-squares estimation of the leading edge radius, and airfoil angles. 

@author: T.S. Vermeulen
@email: thomas0708.vermeulen@gmail.com / T.S.Vermeulen@student.tudelft.nl
@version: 1.1

Changelog:
- V1.1: Updated with comments from coderabbitAI. 
- V1.0: Adapted version of a parameterization using only the bezier coefficients to give an improved fit to the reference data. 

"""



import numpy as np
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

    def __init__(self) -> None:
        """
        Initialize the AirfoilParameterization class.
        
        This method sets up the initial state of the class.

        Returns
        -------
        None
        """
        

    def BezierCurve3(self,
                     coeff: list[float], 
                     u: float,
                     ) -> np.ndarray|float:
        """
        Calculate a 3rd degree Bezier curve.

        Parameters
         ----------
        coeff : list[float]
            List of 4 control points for the Bezier curve.
        u : float
            Parameter ranging from 0 to 1.

        Returns
        -------
        y : float
            Value of the Bezier curve at parameter u.
        """

        #Input checking
        if len(coeff) != 4:
            raise ValueError(f"Coefficient list must contain exactly 4 elements. Coefficient list contains {len(coeff)} elements")
    
        # Calculate the value of y at u using a 3rd degree Bezier curve
        return coeff[0] * (1 - u) ** 3 + 3 * coeff[1] * u * (1 - u) ** 2 + 3 * coeff[2] * u ** 2 * (1 - u) + coeff[3] * u ** 3
        

    def BezierCurve4(self,
                     coeff: list[float], 
                     u: float,
                     ) -> np.ndarray|float:
        """
        Calculate a 4th degree Bezier curve.

        Parameters
        ----------
        coeff : list[float]
            List of 5 control points for the Bezier curve.
        u : float
            Parameter ranging from 0 to 1.

        Returns
        -------
        y : float
            Value of the Bezier curve at parameter u.
        """

        # Input checking
        if len(coeff) != 5:
            raise ValueError(f"Coefficient list must contain exactly 5 elements. Coefficient list contains {len(coeff)} elements.")     

        return coeff[0] * (1 - u) ** 4 + 4 * coeff[1] * u * (1 - u) ** 3 + 6 * coeff[2] * u ** 2 * (1 - u) ** 2 + 4 * coeff[3] * u ** 3 * (1 - u) + coeff[4] * u ** 4


    def GetCamberAngleDistribution(self,
                       X: np.ndarray|float,
                       Y: np.ndarray|float,
                       ) -> np.ndarray|float:
        """
        Calculate the camber angle distribution over the length of the airfoil.

        Parameters
        ----------
        X : np.ndarray
            Array of x-coordinates along the airfoil.
        Y : np.ndarray
            Array of camber values corresponding to the x-coordinates.

        Returns
        -------
        theta : np.ndarray
            Array of camber gradient angles at each x-coordinate.
        """

        camber_gradient = np.gradient(Y, X)

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
        # Final assumption: profile shape must be provided in input file with unit chord length
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
    

    def GetReferenceParameters(self) -> tuple[float, float, float, float]:
        """
        Extract key parameters of the reference profile from the thickness and camber distributions.

        Calculates the leading edge radius, leading edge direction, 
        trailing edge wedge angle, and trailing edge camber line angle.
        
        Parameters
        ----------
        None

        Returns
        -------
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
        
        self.x_t = self.x_points_thickness[self.idx_maxT]  # X-coordinate of maximum thickness
        self.y_t = self.thickness_distribution[self.idx_maxT]  # Maximum thickness
        self.x_c = self.x_points_camber[self.idx_maxC]  # X-coordinate of maximum camber
        self.y_c = self.camber_distribution[self.idx_maxC]  # Maximum camber
        self.z_TE = self.camber_distribution[-1]  # Trailing edge vertical displacement
        self.dz_TE = self.thickness_distribution[-1]  # Half thickness of the trailing edge

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

        return r_LE, leading_edge_direction, trailing_wedge_angle, trailing_camberline_angle


    def GetThicknessControlPoints(self,
                                  b_8: float,
                                  b_15: float,
                                  r_LE: float,
                                  trailing_wedge_angle: float,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Calculate the control points for the thickness distribution Bezier curves.

        Parameters
        ----------
        b_8 : float
            Control point for the thickness curve.
        b_15 : float
            Control point for the thickness curve.
        r_LE : float
            Leading edge radius.
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
                                                   (-3 * b_8 ** 2) / (2 * r_LE),
                                                   self.x_t])
        
        # Construct leading edge y coefficients
        y_leading_edge_thickness_coeff = np.array([0,
                                                   b_8,
                                                   self.y_t,
                                                   self.y_t])
        
        # Contruct trailing edge x coefficients
        x_trailing_edge_thickness_coeff = np.array([self.x_t,
                                                    (7 * self.x_t + (9 * b_8 ** 2) / (2 * r_LE)) / 4,
                                                    3 * self.x_t + (15 * b_8 ** 2) / (4 * r_LE),
                                                    b_15,
                                                    1])
        
        # Construct trailing edge y coefficients
        y_trailing_edge_thickness_coeff = np.array([self.y_t,
                                                    self.y_t,
                                                    (self.y_t + b_8) / 2,
                                                    self.dz_TE + (1 - b_15) * np.tan(trailing_wedge_angle),
                                                    self.dz_TE])
    
        return x_leading_edge_thickness_coeff, y_leading_edge_thickness_coeff, x_trailing_edge_thickness_coeff, y_trailing_edge_thickness_coeff


    def GetCamberControlPoints(self,
                               b_0: float,
                               b_2: float,
                               b_17: float,
                               leading_edge_direction: float,
                               trailing_camberline_angle: float,
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
                                                self.x_c])
        
        # Construct leading edge y coefficients
        y_leading_edge_camber_coeff = np.array([0,
                                                b_0 * np.tan(leading_edge_direction),
                                                self.y_c,
                                                self.y_c])
        
        # Construct trailing edge x coefficients
        x_trailing_edge_camber_coeff = np.array([self.x_c,
                                                 (3 * self.x_c - self.y_c / np.tan(leading_edge_direction)) / 2,
                                                 (-8 * self.y_c / np.tan(leading_edge_direction) + 13 * self.x_c) / 6,
                                                 b_17,
                                                 1])
        
        # Construct trailing edge y coefficients
        y_trailing_edge_camber_coeff = np.array([self.y_c,
                                                 self.y_c,
                                                 5 * self.y_c / 6,
                                                 self.z_TE + (1 - b_17) * np.tan(trailing_camberline_angle),
                                                 self.z_TE])
        
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

        return upper_x, upper_y, lower_x, lower_y


    def FindInitialParameterization(self, 
                                    reference_file: str,
                                    plot: bool = False) -> tuple[float, float, float, float, float, float, float, float, float]:
        """
        Find the initial parameterization for the profile.

        Uses least-squares minimization of the squared fit error between the reconstructed profile shape 
        and the reference profile shape to find the optimal Bezier control points.

        Parameters
        ----------
        reference_file : str
            Path to the file containing the reference profile coordinates.
        plot : bool
            Boolean controlling if the initial parameterization is plotted. Default is False.

        Returns
        -------
        Tuple of optimized Bezier control points for the airfoil parameterization.
        """

        def Objective(x: list[float],
                      ) -> float:
            """
            Objective function for least-squares minimization.

            Parameters
            ----------
            x : list[float]
                List of design variables [b_8, b_15, b_0, b_2, b_17, r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction].

            Returns
            -------
            squared_fit_error : float
                Sum of squared fit errors of the upper and lower surfaces.
            """

            # Extract the design variables from the design vector x
            # [b_8, b_15, b_0, b_2, b_17, r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction]
            b_8 = x[0]
            b_15 = x[1]
            b_0 = x[2]
            b_2 = x[3]
            b_17 = x[4]
            r_LE = x[5]
            trailing_wedge_angle = x[6]
            trailing_camberline_angle = x[7]
            leading_edge_direction = x[8]

            # Create u-vectors for Bezier curve generation
            # Use 100 points
            n_points = 100
            u_leading_edge = np.zeros(n_points)
            u_trailing_edge = np.zeros(n_points)

            for i in range(n_points):
                u_leading_edge[i] = np.abs(1-np.cos((i*np.pi)/(2*(n_points-1))))  # Space points using a cosine spacing for increased resolution at LE
                u_trailing_edge[i] = np.abs(np.sin((i*np.pi)/(2*(n_points-1))))  # Space points using a sine spacing for increased resolution at TE

            # Calculate the Bezier curve coefficients for the thickness curves
            x_LE_thickness_coeff, y_LE_thickness_coeff, x_TE_thickness_coeff, y_TE_thickness_coeff = self.GetThicknessControlPoints(b_8, 
                                                                                                                                    b_15,
                                                                                                                                    r_LE,
                                                                                                                                    trailing_wedge_angle)
            # Calculate the Bezier curve coefficients for the camber curves
            x_LE_camber_coeff, y_LE_camber_coeff, x_TE_camber_coeff, y_TE_camber_coeff = self.GetCamberControlPoints(b_0,
                                                                                                                     b_2,
                                                                                                                     b_17,
                                                                                                                     leading_edge_direction,
                                                                                                                     trailing_camberline_angle)

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
            
            # Construct full curves by combining LE and TE data
            bezier_thickness = np.concatenate((y_LE_thickness, y_TE_thickness), 
                                              axis = 0)  # Construct complete thickness curve over length of profile
            bezier_thickness_x = np.concatenate((x_LE_thickness, x_TE_thickness), 
                                                axis = 0)  # Construct complete array of x-coordinates over length of profile
            
            bezier_camber = np.concatenate((y_LE_camber, y_TE_camber),
                                           axis = 0)  # Construct complete camber curve over length of profile
            bezier_camber_x = np.concatenate((x_LE_camber, x_TE_camber),
                                             axis = 0)  # Construct complete array of x-coordinates over length of profile
            
            # Calculate the upper and lower surface coordinates from the bezier coordinates
            upper_x, upper_y, lower_x, lower_y = self.ConvertBezier2AirfoilCoordinates(bezier_thickness_x,
                                                                                       bezier_thickness,
                                                                                       bezier_camber_x,
                                                                                       bezier_camber)

            #Need to create interpolation of upper and lower surfaces to ensure we take data from same x-coordinates
            interpolated_upper_surface_data = interpolate.CubicSpline(np.flip(self.reference_data[:self.idx_LE + 1, 0]), 
                                                                      np.flip(self.reference_data[:self.idx_LE + 1, 1])
                                                                      )(upper_x)
            
            interpolated_lower_surface_data = interpolate.CubicSpline(self.reference_data[self.idx_LE:, 0], 
                                                                      self.reference_data[self.idx_LE:, 1])(lower_x)
            
            # Calculate the squared fit errors of the upper and lower surfaces and sum them to obtain the objective function
            squared_fit_error_upper_surface = np.sum((upper_y - interpolated_upper_surface_data) ** 2)
            squared_fit_error_lower_surface = np.sum((lower_y - interpolated_lower_surface_data) ** 2)

            return squared_fit_error_upper_surface + squared_fit_error_lower_surface
            
        def CheckOptimizedResult(b_0: float,
                                 b_2: float, 
                                 b_8: float, 
                                 b_15: float, 
                                 b_17: float,
                                 r_LE: float,
                                 trailing_wedge_angle: float,
                                 trailing_camberline_angle: float,
                                 leading_edge_direction: float,
                                 ) -> None:
            """
            Check the optimized result by plotting the thickness and camber distributions, and the airfoil shape.

            Parameters
            ----------
            b_0 : float
                Control point for the camber.
            b_2 : float
                Control point for the camber.
            b_8 : float
                Control point for the thickness curve.
            b_15 : float
                Control point for the thickness curve.
            b_17 : float
                Control point for the camber.
            r_LE : float
                Leading edge radius.
            trailing_wedge_angle : float
                Trailing edge wedge angle.
            trailing_camberline_angle : float
                Trailing edge camber line angle.
            leading_edge_direction : float
                Leading edge direction angle.

            Returns
            -------
            None
            """
            
            # Create u-vectors for Bezier curve generation
            # Use 100 points
            n_points = 100
            u_leading_edge = np.zeros(n_points)
            u_trailing_edge = np.zeros(n_points)

            for i in range(n_points):
                u_leading_edge[i] = np.abs(1-np.cos((i*np.pi)/(2*(n_points-1))))  # Space points using a cosine spacing for increased resolution at LE
                u_trailing_edge[i] = np.abs(np.sin((i*np.pi)/(2*(n_points-1))))  # Space points using a sine spacing for increased resolution at TE

            # Calculate the Bezier curve coefficients for the thickness curves
            x_LE_thickness_coeff, y_LE_thickness_coeff, x_TE_thickness_coeff, y_TE_thickness_coeff = self.GetThicknessControlPoints(b_8, 
                                                                                                                                    b_15,
                                                                                                                                    r_LE,
                                                                                                                                    trailing_wedge_angle)
            # Calculate the Bezier curve coefficients for the camber curves
            x_LE_camber_coeff, y_LE_camber_coeff, x_TE_camber_coeff, y_TE_camber_coeff = self.GetCamberControlPoints(b_0,
                                                                                                                     b_2,
                                                                                                                     b_17,
                                                                                                                     leading_edge_direction,
                                                                                                                     trailing_camberline_angle)

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
            
            # Construct full curves by combining LE and TE data
            bezier_thickness = np.concatenate((y_LE_thickness, y_TE_thickness), 
                                              axis = 0)  # Construct complete thickness curve over length of profile
            bezier_thickness_x = np.concatenate((x_LE_thickness, x_TE_thickness), 
                                                axis = 0)  # Construct complete array of x-coordinates over length of profile
            
            bezier_camber = np.concatenate((y_LE_camber, y_TE_camber),
                                           axis = 0)  # Construct complete camber curve over length of profile
            bezier_camber_x = np.concatenate((x_LE_camber, x_TE_camber),
                                             axis = 0)  # Construct complete array of x-coordinates over length of profile
            
            # Calculate the upper and lower surface coordinates from the bezier coordinates
            upper_x, upper_y, lower_x, lower_y = self.ConvertBezier2AirfoilCoordinates(bezier_thickness_x,
                                                                                       bezier_thickness,
                                                                                       bezier_camber_x,
                                                                                       bezier_camber)      

            # Create plots of the thickness distribution compared to the input data
            plt.figure("Thickness Distributions")
            plt.plot(x_LE_thickness, y_LE_thickness, label="LeadingEdgeThickness")
            plt.plot(x_TE_thickness, y_TE_thickness, label="TrailingEdgeThickness")
            plt.plot(x_LE_thickness_coeff, y_LE_thickness_coeff, '*', color='k', label="Bezier Coefficients")
            plt.plot(x_TE_thickness_coeff, y_TE_thickness_coeff, '*', color='k')  # Do not label this line to avoid duplicate legend entry
            plt.plot(self.x_points_thickness, self.thickness_distribution, label="ThicknessInputData")
            plt.legend()

            plt.figure("Camber Distributions")
            plt.plot(x_LE_camber, y_LE_camber, label="LeadingEdgeCamber")
            plt.plot(x_TE_camber, y_TE_camber, label="TrailingEdgeCamber")
            plt.plot(x_LE_camber_coeff, y_LE_camber_coeff, '*', color='k', label="Bezier Coefficients")
            plt.plot(x_TE_camber_coeff, y_TE_camber_coeff, '*', color='k')  # Do not label this line to avoid duplicate legend entry
            plt.plot(self.x_points_camber, self.camber_distribution, label="CamberInputData")
            plt.legend()

            plt.figure("Airfoil Shape")
            plt.plot(upper_x, upper_y, label="Reconstructed Upper Surface")
            plt.plot(lower_x, lower_y, label="Reconstructed Lower Surface")
            plt.plot(self.reference_data[:self.idx_LE + 1, 0], self.reference_data[:self.idx_LE + 1, 1], "-.", color="g")
            plt.plot(self.reference_data[self.idx_LE:, 0], self.reference_data[self.idx_LE:, 1], "-.", color="g")
            plt.show()

            return

        # Load in the reference profile shape and obtain the relevant parameters
        self.GetReferenceThicknessCamber(reference_file)
        r_LE, leading_edge_direction, trailing_wedge_angle, trailing_camberline_angle = self.GetReferenceParameters()
        
        # First define the upper and lower bounds on the bezier parameters 
        # b_8 MUST be within the bounds, exclusive of the bounds, so hence 1E-5 is subtracted from both bounds
        # [b_8, b_15, b_0, b_2, b_17, r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction]
        upper_bounds = [min(self.y_t, np.sqrt(-2 * r_LE * self.x_t / 3)) - 1E-5, np.inf, np.inf, np.inf, np.inf, 0, np.pi, np.pi, np.pi]
        lower_bounds = [1E-5, -np.inf, -np.inf, -np.inf, -np.inf, -0.5, -np.pi, -np.pi, -np.pi]
        bounds = (lower_bounds, upper_bounds)

        # Perform non-linear least-squares regression
        guess_design_vector = [0.7 * upper_bounds[0], 0.75, self.y_c, self.y_c, 0.7, 
                               r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction]  

        optimized_coefficients = optimize.least_squares(Objective, 
                                                        x0=guess_design_vector, 
                                                        bounds=bounds,
                                                        verbose=1,
                                                        gtol=1e-12,
                                                        ftol=1e-12,
                                                        xtol=1e-12,
                                                        max_nfev = 1000,
                                                        )
        
        # Write found coefficients to variables
        optimized_coefficients.x = optimized_coefficients.x.astype(float)
        b_8 = optimized_coefficients.x[0]
        b_15 = optimized_coefficients.x[1]
        b_0 = optimized_coefficients.x[2]
        b_2 = optimized_coefficients.x[3]
        b_17 = optimized_coefficients.x[4]
        r_LE = optimized_coefficients.x[5] 
        trailing_wedge_angle = optimized_coefficients.x[6] 
        trailing_camberline_angle = optimized_coefficients.x[7] 
        leading_edge_direction = optimized_coefficients.x[8]
        
        # Generate
        if plot:
            CheckOptimizedResult(b_0, b_2, b_8, b_15, b_17, r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction)

        return b_0, b_2, b_8, b_15, b_17, r_LE, trailing_wedge_angle, trailing_camberline_angle, leading_edge_direction


if __name__ == "__main__":
    call_class = AirfoilParameterization()
    
    call_class.FindInitialParameterization(r'Test Airfoils\N2415.test',
                                           plot=True)
    
    
    




