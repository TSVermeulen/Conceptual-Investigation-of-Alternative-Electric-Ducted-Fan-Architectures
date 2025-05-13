"""
main-parallelized-UNSGAIII
====

Description
-----------
This module defines the main entry point for running a multi-threaded optimization problem using the pymoo framework.
Uses the starmap parallelization method for flexible parallelization opportunities.
Uses the Unified-Nondominated Sorting Genetic Algorithm 3 with the mixed-variable implementation. 

Functionality
-------------
- Initializes an optimization problem using UNSGA-III with mixed-variable support.
- Configures and runs a genetic algorithm (GA) for optimization.
- Saves the results to a dill database for future reference.

Examples
--------
>>> python main-parallelized-UNSGAIII.py

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
Version: 1.0

Changelog:
- V1.0: Initial implementation. 
"""

# Import standard libraries
import dill
import datetime
import os
import multiprocessing
from pathlib import Path

# Import 3rd party libraries
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.unsga3 import UNSGA3, comp_by_rank_and_ref_line_dist
from pymoo.operators.selection.tournament import TournamentSelection

# Import interface submodels and other dependencies
import config
from problem_definition import OptimizationProblem
from init_population import InitPopulation
from termination_conditions import GetTerminationConditions
from utils import ensure_repo_paths

if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows compatibility when using multiprocessing
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    
    """ Initialize the thread pool and create the runner """
    total_threads = multiprocessing.cpu_count()
    total_threads_avail = (total_threads - config.RESERVED_THREADS) // config.THREADS_PER_EVALUATION
    n_processes = max(1, total_threads_avail)  # Ensure at least one worker is used

    with multiprocessing.Pool(processes=n_processes,
                              initializer=ensure_repo_paths,
                              initargs=()) as pool:

        # Create runner
        runner = StarmapParallelization(pool.starmap)

        """ Initialize the optimization problem and algorithm """
        # Initialize the optimization problem by passing the configuration and the starmap interface of the thread_pool
        if getattr(config, "PROBLEM_TYPE", "single_point") == "multi_point":
            from multi_point_problem_definition import MultiPointOptimizationProblem
            problem = MultiPointOptimizationProblem(elementwise_runner=runner,
                                                    seed=config.GLOBAL_SEED)
        else:        
            problem = OptimizationProblem(elementwise_runner=runner,
                                          seed=config.GLOBAL_SEED)
                
        # Create the reference directions to be used for the optimisation
        ref_dirs = get_reference_directions("energy",
                                            n_dim=len(config.objective_IDs),
                                            n_points=config.POPULATION_SIZE)

        # Initialize the algorithm
        algorithm = UNSGA3(ref_dirs=ref_dirs,
                           pop_size=config.POPULATION_SIZE,
                           mating=MixedVariableMating(selection=TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist),
                                                      eliminate_duplicates=MixedVariableDuplicateElimination()),
                           sampling=InitPopulation(population_type="biased",
                                                   seed=config.GLOBAL_SEED).GeneratePopulation(),
                           eliminate_duplicates=MixedVariableDuplicateElimination(),
                           selection=TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist)
                           )
        
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
    output_name = results_dir / f"res_pop{config.POPULATION_SIZE}_eval{config.MAX_EVALUATIONS}_{timestamp}.dill"
    try:
        with open(output_name, 'wb') as f:
            dill.dump(res, f)
        print(f"Results saved to {output_name}")
    except Exception as e:
        print(f"Error saving results: {e}")