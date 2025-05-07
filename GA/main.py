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
- Saves the results to a dill database for future reference.

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
- V1.2: Updated to match structure of main-parallelized, but with n_processes = 1.
"""

from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
import multiprocessing
import dill
import config
import datetime
import os

from problem_definition import OptimizationProblem
from init_population import InitPopulation
from utils import ensure_repo_paths


if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows compatibility when using multiprocessing
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    
    """ Initialize the thread pool and create the runner """
    n_processes = 1  # Use 1 worker
    with multiprocessing.Manager() as manager:
        shared_cache = manager.dict()  # Initialize shared cache

        with multiprocessing.Pool(processes=n_processes,
                                  initializer=ensure_repo_paths,
                                  initargs=()) as pool:

            # Create runner
            runner = StarmapParallelization(pool.starmap)

            """ Initialize the optimization problem and algorithm """
            # Initialize the optimization problem by passing the configuration and the starmap interface of the thread_pool
            problem = OptimizationProblem(elementwise_runner=runner,
                                          seed=42,
                                          cache=shared_cache)

            # Initialize the algorithm
            algorithm = MixedVariableGA(pop_size=config.POPULATION_SIZE,
                                        sampling=InitPopulation(population_type="biased",
                                                                seed=42).GeneratePopulation())

            # Run the optimization
            res = minimize(problem,
                        algorithm,
                        termination=('n_gen', config.MAX_GENERATIONS),
                        seed=42,
                        verbose=True,
                        save_history=True,
                        return_least_infeasible=True)

    # Print some performance metrics
    print(f"Optimization completed in {res.exec_time:.2f} seconds")

    """ Save the results to a dill file for future reference """
    # This avoids needing to re-run the optimization if the results are needed later.
    # The filename is generated using the process ID and current timestamp to ensure uniqueness.

    now = datetime.datetime.now()
    timestamp = f"{now:%y%m%d%H%M%S%f}"	
    output_name = f"res_pop{config.POPULATION_SIZE}_gen{config.MAX_GENERATIONS}_{timestamp}.dill"
    try:
        with open(output_name, 'wb') as f:
            dill.dump(res, f)
        print(f"Results saved to {output_name}")
    except Exception as e:
        print(f"Error saving results: {e}")