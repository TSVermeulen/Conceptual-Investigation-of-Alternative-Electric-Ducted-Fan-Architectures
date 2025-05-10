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
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination, MultiObjectiveSpaceTermination
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

# Import interface submodels and other dependencies
import config
from problem_definition import OptimizationProblem
from init_population import InitPopulation
from utils import ensure_repo_paths

if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows compatibility when using multiprocessing
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)
    
    """ Initialize the thread pool and create the runner """
    total_threads = multiprocessing.cpu_count()
    RESERVED_THREADS = 2 # Number of threads reserved for the main process and any other non-python processes (OS, programs, etc.)
    total_threads_avail = (total_threads - RESERVED_THREADS) // 2  # Divide by 2 as each MTFLOW evaluation uses 2 threads: one for running MTSET/MTSOL/MTFLO and one for polling outputs

    n_processes = max(1, total_threads_avail)  # Ensure at least one worker is used
    with multiprocessing.Pool(processes=n_processes,
                            initializer=ensure_repo_paths,
                            initargs=()) as pool:

        # Create runner
        runner = StarmapParallelization(pool.starmap)

        """ Initialize the optimization problem and algorithm """
        # Initialize the optimization problem by passing the configuration and the starmap interface of the thread_pool
        problem = OptimizationProblem(elementwise_runner=runner,
                                      seed=config.GLOBAL_SEED)

        # Initialize the algorithm
        algorithm = MixedVariableGA(pop_size=config.POPULATION_SIZE,
                                    sampling=InitPopulation(population_type="biased",
                                                            seed=config.GLOBAL_SEED).GeneratePopulation())

        # Set the termination conditions
        if len(config.objective_IDs) == 1:
            # Set termination conditions for a single objective optimisation
            term_conditions = TerminationCollection(RobustTermination(SingleObjectiveSpaceTermination(tol=1E-6, 
                                                                                                    only_feas=True), 
                                                                                                    period=10),  # Chance in objective value termination condition
                                                    get_termination("n_gen", config.MAX_GENERATIONS),  # Maximum generation count termination condition
                                                    get_termination("n_evals", config.MAX_EVALUATIONS),  # Maximum evaluation count termination condition
                                                    RobustTermination(DesignSpaceTermination(tol=1E-8), 
                                                                    period=10),  # Maximum change in design vector termination condition
                                                    RobustTermination(ConstraintViolationTermination(tol=1E-8, terminate_when_feasible=False), 
                                                                    period=10)  # Maximum change in constriant violation termination condition
                                                    )
        else:
            # Set termination conditions for a multiobjective optimisation
            term_conditions = TerminationCollection(RobustTermination(MultiObjectiveSpaceTermination(tol=1E-6, 
                                                                                                    only_feas=True), 
                                                                                                    period=10),  # Chance in objective value termination condition
                                                    get_termination("n_gen", config.MAX_GENERATIONS),  # Maximum generation count termination condition
                                                    get_termination("n_evals", config.MAX_EVALUATIONS),  # Maximum evaluation count termination condition
                                                    RobustTermination(DesignSpaceTermination(tol=1E-8), 
                                                                    period=10),  # Maximum change in design vector termination condition
                                                    RobustTermination(ConstraintViolationTermination(tol=1E-8, terminate_when_feasible=False), 
                                                                    period=10)  # Maximum change in constraint violation termination condition
                                                    )
        # Run the optimization
        res = minimize(problem,
                       algorithm,
                       termination=term_conditions,
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