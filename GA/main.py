"""
main
====

Description
-----------
This module defines the main entry point for running a single-threaded optimization problem using the pymoo framework. 

Functionality
-------------
- Initializes an optimization problem with mixed-variable support.
- Configures and runs a genetic algorithm (GA) for optimization.
- Saves the results to a shelve database for future reference.

Examples
--------
>>> python main.py

Notes
-----
This module integrates with the pymoo framework and requires the problem definition and population initialization modules. 
Ensure that all dependencies are installed and properly configured.

References
----------
For more details on pymoo and its capabilities, refer to the official documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.1

Changelog:
- V1.0: Initial implementation. 
- V1.1: Updated documentation to reflect changes in the main module structure and added examples for usage.
"""

from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
import dill
import config
import datetime
import os

from problem_definition import OptimizationProblem
from init_population import InitPopulation

# Initialize the optimization problem
problem = OptimizationProblem()

# Initialize the algorithm
algorithm = MixedVariableGA(pop_size=config.POPULATION_SIZE,
                            sampling=InitPopulation(population_type="biased").GeneratePopulation())

# Run the optimization
res = minimize(problem,
               algorithm,
               termination=('n_gen', config.MAX_GENERATIONS),
               seed=1,
               verbose=True,
               save_history=True,
               return_least_infeasible=True,)

# This avoids needing to re-run the optimization if the results are needed later.
# The filename is generated using the process ID and current timestamp to ensure uniqueness.
process_ID = f"{os.getpid() % 10000:04d}" 
now = datetime.datetime.now()
timestamp = f"{now:%y%m%d%H%M%S%f}"	
output_name = f"res_{process_ID}_{timestamp}.dill"
try:
    with open(output_name, 'wb') as f:
        dill.dump(res, f)
    print(f"Results saved to {output_name}")
except Exception as e:
    print(f"Error saving results: {e}")