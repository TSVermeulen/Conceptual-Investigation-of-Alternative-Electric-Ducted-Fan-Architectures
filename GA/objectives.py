"""
objectives
==========


"""

import sys
import os
import numpy as np


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
submodels_path = os.path.join(parent_dir, "Submodels")

# Add the submodels path to the system path
# sys.path.append(submodels_path)

# Add the parent folder path to the system path
# sys.path.append(parent_dir)

# Import the required submodels
from ..MTFLOW_caller import MTFLOW_caller


class Objectives:
    """
    A class to define the objectives for the optimization problem.
    """

    def _init__(self,
                x: np.ndarray) -> None:
        """
        
        """

        pass


    def Efficiency(self) -> None:
        """
        Define the efficiency (sub-)objective.
        This sub-objective has identifier 0.

        Returns
        -------
        None
        """

        pass


    def Weight(self) -> None:
        """
        Define the weight (sub-)objective.
        This sub-objective has identifier 1.

        Returns
        -------
        None
        """

        pass

    
    def FrontalArea(self) -> None:
        """
        Define the frontal area (sub-)objective.
        This sub-objective has identifier 2.

        Returns
        -------
        None
        """

        pass


    def PressureRatio(self) -> None:
        """
        Define the pressure ratio (sub-)objective.
        This sub-objective has identifier 3.

        Returns
        -------
        None
        """

        pass


    def MultiPointTOCruise(self) -> None:
        """
        Define the multi-point take-off to cruise (sub-)objective.
        This sub-objective has identifier 4.

        Returns
        -------
        None
        """

        pass


    def ComputeObjective(self,
                         objective_IDs: list[int]) -> float:
        """
        
        """

        objectives_list = [self.Efficiency, self.FrontalArea, self.Weight, self.PressureRatio, self.MultiPointTOCruise]

        