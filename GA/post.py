from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
import dill

import os

# Add the parent and submodels paths to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

result_path = parent_dir / 'res.dill'
with open('res.dill', 'rb') as f:
    res = dill.load(f)

print(res.F)
