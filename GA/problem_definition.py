"""
problem_definition
==================


"""

import os
import sys
import hashlib

import numpy as np

from pymoo.core.problem import ElementwiseProblem

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
submodels_path = os.path.join(parent_dir, "Submodels")

# Add the submodels path to the system path
sys.path.append(submodels_path)

# Add the parent folder path to the system path
sys.path.append(parent_dir)

# import MTFLOW interface submodels
from MTFLOW_caller import MTFLOW_caller
from Submodels.output_handling import output_processing




class OptimizationProblem(ElementwiseProblem):
    """
    Class definition of the optimization problem to be solved using the genetic algorithm. 
    Inherits from the ElementwiseProblem class from pymoo.core.problem.
    """

    def __init__(self,
                 design_var_count: int,
                 obj_count : int,
                 constraint_count : int,
                 lower_bounds : np.ndarray[float],
                 upper_bounds : np.ndarray[float]) -> None:
        """
        Initialization of the OptimizationProblem class. 

        Returns
        -------
        None
        """

        # Class input validation
        if np.any([design_var_count, obj_count, constraint_count] < 0):
            raise ValueError(f"Either the design variable count, objective count, or constraint count is negative: {design_var_count, obj_count, constraint_count}")

        if np.any(lower_bounds > upper_bounds):
            raise ValueError(f"One of the lower bounds is larger than the corresponding upper bound")

        # Initialize the parent class
        super.__init__(n_vars=design_var_count,
                       n_obj=obj_count,
                       n_constr=constraint_count,
                       xl=lower_bounds,
                       xu=upper_bounds)
        

    def GenerateAnalysisName(pop_idx: int, 
                             gen_idx: int) -> str:
        """
        Generate a unique analysis name with a maximum length of 32 characters.
        This is required to enable multi-threading of the optimization problem, since each evaluation of MTFLOW requires a unique set of files. 

        Parameters
        ----------
        - pop_idx : int
            Population index of the current evaluation
        - gen_idx : int
            Generation index of the current evaluation

        Returns
        -------
        - analysis_name : str
            A unique hashed analysis name for the population and generation indices provided. 
        """

        # Construct the base name based on the population and generation indices
        base_name = f"pop{pop_idx}_gen{gen_idx}"

        # Construct a hash suffix
        hash_suffix = hashlib.md5(base_name.encode()).hexdigest()

        # Construct the full analysis name and trim it to be no longer than 32 characters
        analysis_name = base_name + "_" + hash_suffix
        analysis_name = analysis_name[:32]

        return analysis_name

        

    def _evaluate(self, x, out, *args, **kwargs):
        # Initialize the appropriate classes

        
        # Generate a unique analysis name
        pop_idx = kwargs.get("pop_idx", 0)
        gen_idx = kwargs.get("gen_idx", 0)
        analysis_name = self.GenerateAnalysisName(pop_idx,
                                                  gen_idx)


        # Run MTFLOW


        # Extract outputs


        # Obtain objective(s)


        # Compute constraints

        return
