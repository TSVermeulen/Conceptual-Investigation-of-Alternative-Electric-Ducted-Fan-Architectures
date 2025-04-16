"""
objectives
==========


"""

import numpy as np

class Objectives:
    """
    A class to define the objectives for the optimization problem.
    """

    def _init__(self,
                x: np.ndarray = None) -> None:
        """
        Initialisation of the Objectives class.

        Parameters
        ----------
        - x : np.ndarray
            The PyMoo design vector. 
        
        Returns
        -------
        None
        """


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

        return outputs['data']['EtaP']


    def Weight(self) -> None:
        """
        Define the weight (sub-)objective.
        This sub-objective has identifier 1.

        Returns
        -------
        None
        """
        #TODO: Implement weight calculation based on design variables
        pass

    
    def FrontalArea(self) -> None:
        """
        Define the frontal area (sub-)objective.
        This sub-objective has identifier 2.

        Returns
        -------
        None
        """
        #TODO: Implement frontal area calculation/extraction based on forces.xxx file or design variables
        pass


    def PressureRatio(self,
                      output: dict) -> float:
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

        return output["data"]["Pressure Ratio"]


    def MultiPointTOCruise(self) -> None:
        """
        Define the multi-point take-off to cruise (sub-)objective.
        This sub-objective has identifier 4.

        Returns
        -------
        None
        """
        #TODO: Implement multi-point objective function calculation.,
        pass


    def ComputeObjective(self,
                         outputs: dict,
                         objective_IDs: list[int]) -> list[float]:
        """
        Compute the 
        """

        objectives_list = [self.Efficiency, self.FrontalArea, self.Weight, self.PressureRatio, self.MultiPointTOCruise]

        objectives = [objectives_list[i] for i in objective_IDs]

        computed_objectives = []

        for i in range(len(objectives)):
            # We multiply the objectives by -1 to turn the maximisation objectives (i.e. maximise efficiency) 
            # into the PyMoo expected minimisation objectives
            computed_objectives.append(- objectives[i](outputs))

        return computed_objectives
        
if __name__ == "__main__":
    # Run a test of the objectives class
    import os
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    submodels_path = os.path.join(parent_dir, "Submodels")
    # Add the submodels path to the system path
    sys.path.append(submodels_path)
    # Add the parent folder path to the system path
    sys.path.append(parent_dir)

    import config
    from Submodels.output_handling import output_processing

    objectives_class = Objectives()
    output = objectives_class.ComputeObjective(output_processing('test_case').GetAllVariables(3),
                                               config.objective_IDs) 
    print(output)       