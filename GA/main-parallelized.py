"""
main-parallelized
====

Description
-----------
This module defines the main entry point for running a multi-threaded optimization problem using the pymoo framework.
Uses the starmap parallelization method for flexible parallelization opportunities.

Functionality
-------------
- Initializes an optimization problem with mixed-variable support.
- Configures and runs a genetic algorithm (GA) for optimization.
- Saves the results to a dill database for future reference.

Examples
--------
>>> python main-parallelized.py

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
Version: 1.3

Changelog:
- V1.0: Initial implementation. 
- V1.1: Updated documentation to reflect changes in the main module structure and added examples for usage.
- V1.2: Updated to include reserved thread for MTSOL output reader.
- V1.3: Updated to use the utils.ensure_repo_paths function.
"""

# Import standard libraries
import dill
import datetime
import os
import multiprocessing
from pathlib import Path

# Import 3rd party libraries
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize

# Ensure custom package paths are discoverable *before* importing
from utils import ensure_repo_paths #type: ignore
ensure_repo_paths()

# Import interface submodels and other dependencies
import config #type: ignore
from problem_definition import OptimizationProblem #type: ignore
from init_population import InitPopulation #type: ignore
from termination_conditions import GetTerminationConditions #type: ignore
from repair import RepairIndividuals #type: ignore

if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows compatibility when using multiprocessing
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    
    """ Initialize the thread pool and create the runner """
    total_threads = multiprocessing.cpu_count()
    threads_per_eval = max(1, getattr(config, "THREADS_PER_EVALUATION", 2))
    total_threads_avail = max(0, total_threads - config.RESERVED_THREADS)
    
    if total_threads_avail < threads_per_eval:
        # No point spawning processes that will immediately contend for the same cores
        n_processes = 0
    else:
        n_processes = total_threads_avail // threads_per_eval
    
    # Always fall back to at least one serial worker to ensure the script still runs. 
    n_processes = max(1, n_processes)

    # Do not spawn more processes than the GA can effectively use
    n_processes = min(n_processes, config.POPULATION_SIZE)
    
    print(f"Spawning {n_processes} worker processes (total threads: {total_threads}, available: {total_threads_avail}, threads per eval: {threads_per_eval})")
    with multiprocessing.Pool(processes=n_processes,
                              initializer=ensure_repo_paths,
                              maxtasksperchild=100,
                              ) as pool:

        # Create runner
        runner = StarmapParallelization(pool.starmap)

        """ Initialize the optimization problem and algorithm """
        # Initialize the optimization problem by passing the configuration and the starmap interface of the thread_pool
        problem = OptimizationProblem(elementwise_runner=runner,
                                      seed=config.GLOBAL_SEED)

        # Initialize the algorithm
        algorithm = MixedVariableGA(pop_size=config.POPULATION_SIZE,
                                    sampling=InitPopulation(population_type="biased",
                                                            seed=config.GLOBAL_SEED).GeneratePopulation(),
                                    repair=RepairIndividuals())
        
        # Run the optimization
        res = minimize(problem,
                       algorithm,
                       termination=GetTerminationConditions(),
                       seed=config.GLOBAL_SEED,
                       verbose=True,
                       save_history=True,
                       return_least_infeasible=True)

        # Print some performance metrics
        print(f"Optimization completed in {res.exec_time:.2f} seconds")
        print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

        """ Save the results to a dill file for future reference """
        # This avoids needing to re-run the optimization if the results are needed later.
        # The filename is generated using the process ID and current timestamp to ensure uniqueness.

        # First generate the results folder if it does not exist already
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)

        now = datetime.datetime.now()
        timestamp = f"{now:%y%m%d%H%M%S%f}"	
        output_name = results_dir / f"res_pop{config.POPULATION_SIZE}_gen{config.MAX_GENERATIONS}_{timestamp}.dill"
        try:
            with open(output_name, 'wb') as f:
                dill.dump(res, f)
            print(f"Results saved to {output_name}")
        except Exception as e:
            print(f"Error saving results: {e}")