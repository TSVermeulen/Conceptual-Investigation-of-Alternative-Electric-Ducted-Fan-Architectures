"""
repair
===========

Description
-----------
This module provides repair operators for evolutionary optimization, specifically for airfoil and blade parameterizations.
It ensures that individuals in the population satisfy geometric and physical constraints. 

Classes
-------
RepairIndividuals
    A class derived from pymoo's Repair, implementing custom repair logic for airfoil and blade design variables.

Examples
--------
>>> repair = RepairIndividuals()
>>> repaired_population = repair._do(problem, population)

Notes
-----
This module is intended for use with the PyMoo optimization framework. 

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Date [dd-mm-yyyy]: [22-05-2025]
Version: 1.4

Changelog:
- V1.0: Initial implementation of repair operators for profile parameterizations.
- V1.1: Added enforcement of positive sweepback for blade leading edge.
- V1.2: Improved one-to-one enforcement for Bezier curves.
- V1.3: Refactored repair logic and updated documentation. Improved robustness of one-to-one enforcing by including additonal equation for gamma_LE.
- V1.4: Made bounds on repair enforce_one2one a reference to the design vector initialisation to ensure single source of truth. Added explicit repair out of bounds operator. 
"""

# Import 3rd party libraries
import numpy as np
from pymoo.core.repair import Repair
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside

# Import local libraries
from utils import ensure_repo_paths #type: ignore
ensure_repo_paths()

import config #type: ignore
from Submodels.Parameterizations import AirfoilParameterization #type: ignore
from design_vector_interface import DesignVectorInterface #type: ignore
from init_designvector import DesignVector #type: ignore


class RepairIndividuals(Repair):
    """
    A repair operator class. This class repairs individuals in the 
    population by checking if the profile parameterizations are one-to-one and 
    if the sweep angle distribution is physical. 
    """


    def __init__(self) -> None:
        """
        Initialization of the RepairIndividuals class.
        """

        # Call the parent class constructor
        super().__init__()
        
        # Define feasibility offset between consecutive control points
        self.feasibility_offset = config.PROFILE_FEASIBILITY_OFFSET

        # Initialize the airfoil parameterization class
        self.airfoil_parameterization = AirfoilParameterization()

        # Create Bezier U-vectors
        self._u_vectors: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]] = self.airfoil_parameterization.GenerateBezierUVectors()

        # Extract BP3434 bounds from the design vector class
        self.BP_bounds = DesignVector().BP_3434_bounds

        # Initialize upper and lower bound lists for the complete design vector array
        self.xu = None
        self.xl = None

    def _computebezier(self, 
                       profile_params: dict[str, float]) -> tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
        """
        Compute the Bezier curves for the x-coordinates and y-coordinates of the leading and trailing edge thickness and camber distributions
        of the airfoil profile.

        Parameters
        ----------
        - profile_params : dict[str, float]
            Dictionary containing the profile parameters
        
        Returns
        -------
        - tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            - Tuple containing the Bezier curves for the leading and trailing edge thickness and camber distributions x-coordinates
                - x_LE_thickness : np.ndarray
                    Bezier curve for the leading edge thickness x-coordinates
                - x_TE_thickness : np.ndarray
                    Bezier curve for the trailing edge thickness x-coordinates
                - x_LE_camber : np.ndarray
                    Bezier curve for the leading edge camber x-coordinates
                - x_TE_camber : np.ndarray
                    Bezier curve for the trailing edge camber x-coordinates
            - Tuple containing the Bezier curves for the leading and trailing edge thickness and camber distributions y-coordinates
                - y_LE_thickness : np.ndarray
                    Bezier curve for the leading edge thickness y-coordinates
                - y_TE_thickness : np.ndarray
                    Bezier curve for the trailing edge thickness y-coordinates
                - y_LE_camber : np.ndarray
                    Bezier curve for the leading edge camber y-coordinates
                - y_TE_camber : np.ndarray
                    Bezier curve for the trailing edge camber y-coordinates
        """

        # Calculate the Bezier curve coefficients for the thickness curves
        (x_LE_thickness_coeff, 
         y_LE_thickness_coeff, 
         x_TE_thickness_coeff, 
         y_TE_thickness_coeff) = self.airfoil_parameterization.GetThicknessControlPoints(profile_params)
            
        # Compute the bezier curves for thickness
        u_leading_edge, u_trailing_edge = self._u_vectors
        x_LE_thickness = self.airfoil_parameterization.BezierCurve3(x_LE_thickness_coeff,
                                                                    u_leading_edge)
        x_TE_thickness = self.airfoil_parameterization.BezierCurve4(x_TE_thickness_coeff,
                                                                    u_trailing_edge[1:])
        y_LE_thickness = self.airfoil_parameterization.BezierCurve3(y_LE_thickness_coeff, 
                                                                    u_leading_edge)
        y_TE_thickness = self.airfoil_parameterization.BezierCurve4(y_TE_thickness_coeff,
                                                                    u_trailing_edge[1:])
        
        if profile_params["y_c"] >= 1e-3:
            # Only calculate the bezier curves for camber if the camber is non-zero
            (x_LE_camber_coeff, 
             y_LE_camber_coeff, 
             x_TE_camber_coeff, 
             y_TE_camber_coeff) = self.airfoil_parameterization.GetCamberControlPoints(profile_params)

            # Compute the bezier curves for camber
            x_LE_camber = self.airfoil_parameterization.BezierCurve3(x_LE_camber_coeff,
                                            u_leading_edge) 
            x_TE_camber = self.airfoil_parameterization.BezierCurve4(x_TE_camber_coeff,
                                            u_trailing_edge[1:])
            y_LE_camber = self.airfoil_parameterization.BezierCurve3(y_LE_camber_coeff, 
                                            u_leading_edge)
            y_TE_camber = self.airfoil_parameterization.BezierCurve4(y_TE_camber_coeff, 
                                            u_trailing_edge[1:])
            
        else:
            x_LE_camber = np.zeros_like(x_LE_thickness)
            x_TE_camber = np.zeros_like(x_TE_thickness)
            y_LE_camber = np.zeros_like(y_LE_thickness)
            y_TE_camber = np.zeros_like(y_TE_thickness)
        
        return (x_LE_thickness, x_TE_thickness, x_LE_camber, x_TE_camber), \
               (y_LE_thickness, y_TE_thickness, y_LE_camber, y_TE_camber)


    def _enforce_one2one(self, 
                         profile_params: dict[str, float]) -> dict[str, float]:
        """
        Enforce that the x-coordinates x bezier curves and y-coordinates y bezier curves for thickness and camber are one to one.
        If the enforcing fails, the function returns the best attempt. 

        Parameters
        ----------
        - profile_params : dict[str, float]
            Dictionary containing the profile parameters
        
        Returns
        -------
        - profile_params : dict[str, float]
            Dictionary containing the profile parameters with adjusted values to ensure one to one mapping
        """

        # Attempt to enforce one to one mapping of the bezier x-curves for 100 attempts. 
        # If we fail to find a one to one mapping, we will return the profile parameters as is and accept the infeasibility
        original_params = profile_params.copy()
        for _ in range(200):
            profile_params = profile_params.copy()  # Isolate each attempted fix
            # Compute the bezier curves for the x-coordinates. x_LE_thickness is always one to one, so we can ignore it.
            ((_, 
              x_TE_thickness, 
              x_LE_camber, 
              x_TE_camber),
             (_,
              y_TE_thickness,
              y_LE_camber,
              y_TE_camber)) = self._computebezier(profile_params)

            # Check one to one of all x points
            one_to_one_TE_thickness_x = np.all(np.diff(x_TE_thickness) >= 0)
            one_to_one_LE_camber_x = np.all(np.diff(x_LE_camber) >= 0)
            one_to_one_TE_camber_x = np.all(np.diff(x_TE_camber) >= 0)

            # Check one to one of all y points
            one_to_one_TE_thickness_y = np.all(np.diff(y_TE_thickness) <= 0)  # <=0 since TE thickness should be decreasing
            one_to_one_LE_camber_y = np.all(np.diff(y_LE_camber) >= 0)  # >=0 since LE camber should be increasing
            one_to_one_TE_camber_y = np.all(np.diff(y_TE_camber) <= 0)  # <=0 since TE camber should be decreasing

            # Check if all x points are one to one. If so, we return the updated profile parameters
            if one_to_one_TE_thickness_x and one_to_one_LE_camber_x and one_to_one_TE_camber_x and one_to_one_TE_thickness_y and one_to_one_LE_camber_y and one_to_one_TE_camber_y:
                return profile_params

            # Handle TE thickness x points
            if not one_to_one_TE_thickness_x:
                # Adjust the third x control point to enforce x3 = x_2 + feasibility_offset
                if (profile_params["b_15"] - profile_params["x_t"]) / (1 - profile_params["x_t"]) < 3 * profile_params["x_t"] + 15 * profile_params["b_8"] ** 2 / (4 * profile_params["r_LE"]):
                    b_15_adjusted_coor = 3 * profile_params["x_t"] + 15 * profile_params["b_8"] ** 2 / (4 * profile_params["r_LE"]) + self.feasibility_offset
                    b_15_adjusted = profile_params["x_t"] + (1 - profile_params["x_t"]) * b_15_adjusted_coor
                    b_15_adjusted = np.clip(b_15_adjusted, self.BP_bounds["b_15"][0], self.BP_bounds["b_15"][1])  # Enfoce b_15 to bounds
                    profile_params["b_15"] = b_15_adjusted

                if (3 * profile_params["x_t"] + 15 * profile_params["b_8"] ** 2 / (4 * profile_params["r_LE"])) > profile_params["b_15"]:
                    # Adjust the second control point to enforce x2 < x3
                    b_8_adjusted = np.sqrt(-10 * profile_params["x_t"] * profile_params["r_LE"] / 21) - 1e-2
                    b_8_map = b_8_adjusted / min(profile_params["y_t"], np.sqrt(-2 * profile_params["r_LE"] * profile_params["x_t"] / 3))
                    b_8_clipped_map = np.clip(b_8_map, self.BP_bounds["b_8"][0], self.BP_bounds["b_8"][1])
                    b_8_adjusted_clipped = b_8_clipped_map * min(profile_params["y_t"], np.sqrt(-2 * profile_params["r_LE"] * profile_params["x_t"] / 3))
                    profile_params["b_8"] = b_8_adjusted_clipped
            
            # Handle LE camber x points
            if not one_to_one_LE_camber_x:
                if profile_params["b_2"] < profile_params["b_0"]: 
                    b_2 = profile_params["b_0"] + self.feasibility_offset
                    profile_params["b_2"] = np.clip(b_2, self.BP_bounds["b_2"][0], self.BP_bounds["b_2"][1])  # Enforce b_2 to bounds

                if profile_params["b_2"] > profile_params["x_c"]:
                    b_2 = profile_params["x_c"] - self.feasibility_offset
                    profile_params["b_2"] = np.clip(b_2, self.BP_bounds["b_2"][0], self.BP_bounds["b_2"][1])  # Enforce b_2 to bounds

            # Handle TE camber x points
            if not one_to_one_TE_camber_x:
                if (profile_params["b_17"] - profile_params["x_c"]) / (1 - profile_params["x_c"]) < (-8 * profile_params["y_c"] / np.tan(profile_params["leading_edge_direction"]) + 13 * profile_params["x_c"]) / 6:
                    b_17_adjusted_coor = (-8 * profile_params["y_c"] / np.tan(profile_params["leading_edge_direction"]) + 13 * profile_params["x_c"]) / 6 + self.feasibility_offset
                    b_17_adjusted = (1 - profile_params["x_c"]) * b_17_adjusted_coor + profile_params["x_c"]
                    b_17_adjusted = np.clip(b_17_adjusted, self.BP_bounds["b_17"][0], self.BP_bounds["b_17"][1])  # Enforce b_17 to bounds
                    profile_params["b_17"] = b_17_adjusted  

                elif profile_params["x_c"] > (3 * profile_params["x_c"] - profile_params["y_c"] / np.tan(profile_params["leading_edge_direction"]))  / 2:
                    gamma_LE_adjusted_x_based = np.atan(profile_params["y_c"] / (profile_params["x_c"] - 2 * self.feasibility_offset)) + 1e-3
                    gamma_LE_adjusted_b0_based = np.atan(profile_params["y_c"] / (profile_params["b_0"] * profile_params["x_c"])) - 1e-3  # based on the LE camber y coordinates

                    # gamma_LE must lie somewhere between the two computed values for it to be feasible, so we simply take the middle value.
                    gamma_LE_adjusted = (gamma_LE_adjusted_x_based + gamma_LE_adjusted_b0_based) / 2
                                                   
                    profile_params["leading_edge_direction"] = np.clip(gamma_LE_adjusted, self.BP_bounds["leading_edge_direction"][0], self.BP_bounds["leading_edge_direction"][1])  # Enforce gamma_LE to bounds

                elif profile_params["x_c"] > (-8 * profile_params["y_c"] / np.tan(profile_params["leading_edge_direction"]) + 13 * profile_params["x_c"]) / 6:
                    y_c_adjusted = 7 / 8 * profile_params["x_c"] * np.tan(profile_params["leading_edge_direction"]) - 1e-3
                    y_c_adjusted = np.clip(y_c_adjusted, self.BP_bounds["y_c"][0], self.BP_bounds["y_c"][1])  # Enforce y_c to bounds
                    profile_params["y_c"] = y_c_adjusted
            
            # Handle TE thickness y points
            if not one_to_one_TE_thickness_y:
                # Set the TE thickness to the minimum value
                profile_params["dz_TE"] = 0
                
                # Compute the new trailing edge wedge angle
                beta_TE = np.atan((profile_params["y_t"] + profile_params["b_8"]) / (2 * (1 - profile_params["b_15"]))) - 1e-3
                beta_TE = np.clip(beta_TE, self.BP_bounds["trailing_wedge_angle"][0], self.BP_bounds["trailing_wedge_angle"][1])  # Enforce beta_TE to bounds
                profile_params["trailing_wedge_angle"] = beta_TE

            # Handle LE camber y points
            if not one_to_one_LE_camber_y:
                # Adjust the b_0 control point
                b_0_coor = profile_params["y_c"] / np.tan(profile_params["leading_edge_direction"]) - 1e-3
                b_0 = b_0_coor / profile_params["x_c"]
                b_0 = np.clip(b_0, self.BP_bounds["b_0"][0], self.BP_bounds["b_0"][1])  # Enforce b_0 to bounds
                profile_params["b_0"] = b_0
            
            # # Handle TE camber y points
            if not one_to_one_TE_camber_y:
                # Adjust z_TE to 0 
                profile_params["z_TE"] = 0

                # Compute the new trailing camberline angle
                alpha_TE = np.atan((5/6 * profile_params["y_c"]) / (1 - profile_params["b_17"])) + 1e-3
                alpha_TE = np.clip(alpha_TE, self.BP_bounds["trailing_camberline_angle"][0], self.BP_bounds["trailing_camberline_angle"][1])  # Enforce alpha_TE to bounds
                profile_params["trailing_camberline_angle"] = alpha_TE
                           
        return original_params


    def _enforce_blade_LE_positive_sweepback(self, blading_params: dict[str, any]) -> dict[str, any]:
        """
        Enforce that the leading edge of the blade has a positive sweepback angle along the span. 
        We require that the leading edge of the blade row has a positive sweepangle distribution along the span, 
        although its slope may vary. In other works, from root to tip, the sweep angle may increase, 
        but it may not decrease. 

        Parameters
        ----------
        - blading_params : dict[str, any]
            Dictionary containing the blading parameters
        
        Returns
        -------
        - blading_params : dict[str, any]
            Dictionary containing the blading parameters with adjusted values to ensure positive sweepback angle
        """

        # Extract the corresponding sweep angles and make them positive increasing,
        # and write them back to the blading parameters. 
        blading_params["sweep_angle"] = np.maximum.accumulate(blading_params["sweep_angle"])  

        return blading_params
    

    def _do(self, 
            problem: object, 
            X: list[dict[str, float | int]], **kwargs) -> list[dict[str, float | int]]:
        """
        Perform a simple repair on all individuals in the population. 
        This is not guaranteed to fix the parameterization, but testing shows it fixes 90% of the infeasible design vectors. 

        Parameters
        ----------
        - problem : object
            The optimisation problem.
        - X : list[dict[str, float | int]]
            The list of design vector dictionaries for all individuals in the population. 
        
        Returns
        -------
        - X : list[dict[str, float | int]]
            The repaired list of design vector dictionaries for all individuals in the population. 
        """

        dvi = DesignVectorInterface()

        # Loop over all individuals in the population and repair them if needed
        for i, individual in enumerate(X):
            # First deconstruct the design vector in to the different design dictionaries
            # We do not need to compute the duct LE y coordinate here, so we can skip this step 
            # and speed up the computation by setting compute_duct = False.
            (centerbody_variables,
            duct_variables,
            blade_design_parameters,
            blade_blading_parameters,
            _) = dvi.DeconstructDesignVector(individual, compute_duct = False)
            
            if config.OPTIMIZE_CENTERBODY:
                # Repair the centerbody parameters
                centerbody_variables = self._enforce_one2one(centerbody_variables)
            
            if config.OPTIMIZE_DUCT:
                # Repair the duct parameters
                duct_variables = self._enforce_one2one(duct_variables)
            
            for j, optimise_stage in enumerate(config.OPTIMIZE_STAGE):
                if optimise_stage:
                    # Repair the blading parameters
                    blade_blading_parameters[j] = self._enforce_blade_LE_positive_sweepback(blade_blading_parameters[j])

                    # Loop over all the radial sections and repair the profile parameters
                    for k in range(config.NUM_RADIALSECTIONS[j]):
                        # Repair the profile parameters
                        blade_design_parameters[j][k] = self._enforce_one2one(blade_design_parameters[j][k])
            
            # Reconstruct the design vector into a singular dictionary
            x = dvi.ReconstructDesignVector(centerbody_variables,
                                               duct_variables,
                                               blade_design_parameters,
                                               blade_blading_parameters)
            
            # Convert design vector into array together with bounds to enforce design variable bounds
            x_array = np.array(list(x.values()))

            # Only extract the bounds of they are not already written in self. 
            if self.xu is None or self.xl is None:
                if problem is None:
                    raise ValueError("'problem' must expose xl/xu when calling RepairIndividuals._do()")
                
                self.xl = np.array(list(problem.xl.values()))
                self.xu = np.array(list(problem.xu.values()))

            # Repair the design vector if it is out of bounds
            x_array = set_to_bounds_if_outside(x_array, self.xl, self.xu)

            # Convert the array back to a dictionary and write the result to X
            X[i] = dict(zip(x.keys(), x_array))
            
        return X


if __name__ == "__main__":
    from init_population import InitPopulation #type: ignore

    pop = InitPopulation("biased").GeneratePopulation()

    pop_dict = [pop.get("X")[i] for i in range(len(pop))]

    # Create an instance of the RepairIndividuals class
    repair = RepairIndividuals()
    repaired_pop = repair._do(None, pop_dict)

    test_repaired_pop = repair._do(None, repaired_pop)