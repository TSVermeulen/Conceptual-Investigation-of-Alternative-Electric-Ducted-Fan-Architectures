from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
import dill
import config

from problem_definition import OptimizationProblem
from init_population import InitPopulation
import os
import sys
from pathlib import Path

# Add the parent and submodels paths to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
submodels_path = os.path.join(parent_dir, "Submodels")
sys.path.extend([parent_dir, submodels_path])
submodels_path = Path(submodels_path)  # Convert to Path object for easier manipulation
os.chdir(parent_dir)

with open('res.dill', 'rb') as f:
    res = dill.load(f)

print(res.F)
