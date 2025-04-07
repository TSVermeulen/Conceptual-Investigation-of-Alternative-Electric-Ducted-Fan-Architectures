"""

This file is a simple example of the pymoo library implementation for UNSGA-3.
This is NOT a functional code, but a GitHub copilot generated code.

"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize
from pymoo.factory import get_reference_directions
from Submodels.output_handling import output_processing  # Import the class containing the getCTCPEta method

# Define your custom optimization problem
class MyOptimizationProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=3,  # Number of decision variables
                         n_obj=3,  # Number of objectives
                         n_constr=0,  # Number of constraints
                         xl=np.array([0.0, 0.0, 0.0]),  # Lower bounds of decision variables
                         xu=np.array([1.0, 1.0, 1.0]))  # Upper bounds of decision variables

    def _evaluate(self, x, out, *args, **kwargs):
        # Initialize the output_processing class
        op = output_processing(analysis_name='x22a_validation')

        # Call the getCTCPEta method to get the objective values
        CT, CP, EtaP = op.GetCTCPEtaP()

        # Set the objectives
        out["F"] = [CT, CP, EtaP]

# Example usage with pymoo
if __name__ == "__main__":
    # Define the reference directions for the UNSGA3 algorithm
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    # Define the optimization problem
    problem = MyOptimizationProblem()

    # Define the algorithm
    algorithm = UNSGA3(
        ref_dirs=ref_dirs,
        pop_size=92
    )

    # Perform the optimization
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=True)

    # Print the results
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))