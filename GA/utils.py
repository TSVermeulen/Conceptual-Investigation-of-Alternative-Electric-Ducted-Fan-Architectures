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
            sys.path.append(path_str)


def calculate_n_reference_points(cfg: object) -> int:
    """
    Calculate the number of points needed to construct 
    the right number of reference directions. 

    Parameters
    ----------
    - cfg: object
        Configuration object

    Returns
    -------
    - p : int
        The number of points needed
    """

    m = cfg.n_objectives
    p = 0
    while True:
        p += 1
        count = math.comb(p + m - 1, m -1)
        if count >= cfg.POPULATION_SIZE:
            return p
