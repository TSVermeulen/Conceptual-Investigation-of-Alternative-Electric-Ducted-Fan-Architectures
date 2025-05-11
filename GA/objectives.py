"""
objectives
==========

Description
-----------
This module provides an interface to define and compute objectives for optimization problems. 
It includes methods to calculate various sub-objectives such as efficiency, weight, frontal area, 
pressure ratio, and multi-point objectives.

Classes
-------
Objectives
    A class to define and compute the objectives for the optimization problem.

Examples
--------
>>> objectives_class = Objectives()
>>> outputs = {"data": {"EtaP": 0.85, "Pressure Ratio": 1.2}}
>>> objective_IDs = [0, 3]
>>> computed_objectives = objectives_class.ComputeObjective(outputs, objective_IDs)
>>> print(computed_objectives)

Notes
-----
This module is designed to work with optimization frameworks such as PyMoo. 
The objectives are structured to be compatible with PyMoo's minimization-based approach.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 2.0

Changelog:
- V1.0: Initial implementation with basic sub-objectives and placeholders for unimplemented methods.
- V1.1: Added ComputeObjective method to handle multiple objectives dynamically.
- V1.2: Improved documentation and added type hints for better clarity.
- V2.0: Refactored code for better modularity and maintainability. Updated examples and notes.
"""

# Import 3rd party libraries
import numpy as np

# Import config module
import config


class Objectives:
    """
    A class to define the objectives for the optimization problem.
    """

    def __init__(self,
                 duct_variables : dict[str, any],
                 **kwargs) -> None:
        """
        Initialisation of the Objectives class.

        Parameters
        ----------
        - duct_variables: dict[str, any]
            The duct design varaible dictionary.
        
        Returns
        -------
        None
        """

        # Write the inputs to self
        self.duct_variables = duct_variables


    def Efficiency(self,
                   outputs: dict) -> float:
        """
        Define the efficiency (sub-)objective.
        This sub-objective has identifier 0.

        Parameters
        ----------
        - outputs : dict
            A dictionary containing the outputs from the forces.xxx file. 
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - Propulsive Efficiency: float
            A float of the propulsive efficiency, defined as CT/CP.
        """

        return 1 - outputs['data']['EtaP']


    def FrontalArea(self,
                    outputs: dict) -> None:
        """
        Define the frontal area (sub-)objective.
        This sub-objective has identifier 2.

        Returns
        -------
        None
        """

        # To comput the frontal area, we need the maximum radius of the ducted propeller/fan.
        # This can be computed based on the radial LE coordinate of the duct, 
        # together with the maximum y-coordinate of the duct profile.

        # Lazy import the airfoil parameterization class to construct the x,y coordinates of the duct geometry
        from Submodels.Parameterizations import AirfoilParameterization

        # Compute the airfoil coordinates
        # We only care about the upper y coordinates so they are the only ones we store
        _, upper_y, _, _ = AirfoilParameterization().ComputeProfileCoordinates([self.duct_variables["b_0"],
                                                                                self.duct_variables["b_2"],
                                                                                self.duct_variables["b_8"],
                                                                                self.duct_variables["b_15"],
                                                                                self.duct_variables["b_17"]],
                                                                                self.duct_variables)

        # Dimensionalise the y coordinates using the chord length
        upper_y *= self.duct_variables["Chord Length"]

        # Compute the maximum radius
        max_radius = self.duct_variables["Leading Edge Coordinates"][1] + np.max(upper_y)

        # Since we deal with axisymmetric geometry, the frontal area is then simply the area of a circle
        frontal_area = np.pi * max_radius ** 2

        # Return the frontal area normalised by the reference frontal area in config
        # This is needed to ensure all objectives are of the same order of magnitude and thus have equal weight to the GA optimiser. 
        return frontal_area / config.REF_FRONTAL_AREA


    def PressureRatio(self,
                      outputs: dict) -> float:
        """
        Define the pressure ratio (sub-)objective.
        This sub-objective has identifier 3.

        Parameters
        ----------
        - outputs : dict
            A dictionary containing the outputs from the forces.xxx file. 
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - Pressure Ratio : float
            A float of the exit pressure ratio.
        """

        return 1 - outputs["data"]["Pressure Ratio"]


    def MultiPointTOCruise(self,
                           outputs: dict) -> None:
        """
        Define the multi-point take-off to cruise (sub-)objective.
        This sub-objective has identifier 4.

        Returns
        -------
        None
        """
        #TODO: Implement multi-point objective function calculation.,
        raise NotImplementedError("Multi-point objective function is not implemented yet.")


    def ComputeObjective(self,
                         analysis_outputs: dict,
                         objective_IDs: list[int],
                         out: dict) -> None:
        """
        Computes the objectives for optimization based on the provided analysis outputs.
        This method evaluates a list of objective functions, specified by their IDs in the 
        configuration, and returns their computed values. The objectives are negated to 
        convert maximization objectives (e.g., maximize efficiency) into minimization 
        objectives, as required by the PyMoo optimization framework.
                        
        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs of the analysis 
            required for computing the objectives.
        - out : dict
            A dictionary to store the computed objectives. 

        Returns
        -------
        None, the out dictionary is updated in place with the computed objectives.                
        """

        objectives_list = [self.Efficiency, self.FrontalArea, self.PressureRatio, self.MultiPointTOCruise]

        objectives = [objectives_list[i] for i in objective_IDs]

        computed_objectives = []

        for i in range(len(objectives)):
            # Rounds the objective values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
            computed_objectives.append(round(objectives[i](analysis_outputs), 5))

        out["F"] = np.column_stack(computed_objectives)
        
        
if __name__ == "__main__":
    # Run a test of the objectives class

    # Add the parent folder path to the system path
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).resolve().parent.parent
    submodels_path = parent_dir / "Submodels"
    sys.path.extend([str(parent_dir), str(submodels_path)])

    from Submodels.output_handling import output_processing
    import config

    objectives_class = Objectives()
    output = {}
    objectives_class.ComputeObjective(output_processing('test_case').GetAllVariables(3),
                                      config.objective_IDs,
                                      output) 
    print(output)       