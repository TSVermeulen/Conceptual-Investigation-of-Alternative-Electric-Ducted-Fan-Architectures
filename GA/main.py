"""
main
====




"""
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize

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