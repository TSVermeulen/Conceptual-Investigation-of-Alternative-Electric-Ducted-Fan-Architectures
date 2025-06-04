"""
utils
=====

Simple utility script to ensure the paths are correctly set up and
compute the required number of points for the appropriate reference directions count

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.1

Changelog:
- V1.0: Initial implementation.
- V1.1: Implemented reference direction point counter to ensure correct number of reference directions are used in the optimisation.
        Updated documentation and switched from sys.path.append to sys.path.insert(0, path) to ensure local code wins over any global packages.
"""

# Import standard libraries
import sys
import math
from pathlib import Path


def ensure_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    submodels = repo_root / "Submodels"
    validation = repo_root / "Validation"
    ga = repo_root / "GA"
    for p in (repo_root, submodels, validation, ga):
        path_str = str(p)
        if path_str not in sys.path:
            # Pre-pend so local code wins over any globally installed packages
            sys.path.insert(0, path_str)


def calculate_n_reference_points(cfg: object) -> int:
    """
    Calculate the number of points needed to construct
    the right number of reference directions using a
    binomial coefficient

    Parameters
    ----------
    - cfg: object
        Configuration object.
        We cannot import config directly in this file, since config already uses ensure_repo_paths, which would result in a circular import error.

    Returns
    -------
    - p : int
        The number of points needed
    """

    m = cfg.n_objectives
    p = 0

    max_iter = 10000 # hard-stop for safety; tweak if needed
    while p < max_iter:
        p += 1
        count = math.comb(p + m - 1, m - 1)
        if count >= max(1, cfg.POPULATION_SIZE):
            return p

    if m == 1:
        print(f"Unable to find suitable p within {max_iter} iterations for {m} objectives and population size: {cfg.POPULATION_SIZE}. Setting p equal to the number of objectives...")
        return m
    print(f"Unable to find suitable p within {max_iter} iterations for {m} objectives and population size: {cfg.POPULATION_SIZE}. Setting p equal to the population size...")
    return cfg.POPULATION_SIZE
