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
    binomial coefficient.

    Parameters
    ----------
    - cfg: object
        Configuration object.
        We cannot import config directly in this file, since config already uses 
        ensure_repo_paths, which would result in a circular import error.
        Must have the n_objectives and POPULATION_size attributes.

    Returns
    -------
    - p : int
        The number of points needed
    """

    try:
        m = cfg.n_objectives
    except AttributeError as e:
        raise AttributeError(f"Configuration object is missing required attribute: {e}")

    if m == 1:
        # Single objective uses only 1 reference point
        return m

    max_iter = 10000 # hard-stop for safety; tweak if needed
    p = 0
    best_diff = float('inf')
    best_p = 1
    m = 5

    for p in range(1, max_iter):
        count = math.comb(p + m - 1, m - 1)
        diff = abs(count - cfg.POPULATION_SIZE)
        if diff < best_diff:
            best_diff = diff
            best_p = p

            if diff == 0:
                # If an exact match is found, break from the for loop
                break

    return best_p


if __name__ == "__main__":
    import config
    print(calculate_n_reference_points(config))