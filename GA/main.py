"""
main
====

Description
-----------
This module defines the main entry point for running an optimization problem using the pymoo framework. 

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
import shelve

import config

from problem_definition import OptimizationProblem
from init_population import InitPopulation

# Initialize the optimization problem
problem = OptimizationProblem()

# Initialize the algorithm
algorithm = MixedVariableGA(pop_size=config.POPULATION_SIZE,
                            sampling=InitPopulation(type="biased",
                                                    cfg=config).GeneratePopulation(),
                            )


# Run the optimization
res = minimize(problem,
               algorithm,
               termination=('n_evals', 100),
               seed=1,
               verbose=True,
               save_history=True)

print(res)

# Save the results to a shelve file to enble easy access to the results
# and to avoid the need to re-run the optimization in the future.
myshelf = shelve.open('results.db', 'n')

for key in dir():
    try:
        myshelf[key] = globals()[key]
    except Exception as e:
        print(f"Error saving {key}: {e}")

    finally:
        myshelf.close()