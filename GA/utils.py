"""
utils
=====

Simple utility script to ensure the paths are correctly set up.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial implementation. 
"""

from pathlib import Path
import sys

def ensure_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    submodels = repo_root / "Submodels"
    validation = repo_root / "Validation"
    ga = repo_root / "GA"
    for p in (repo_root, submodels, validation, ga):
        sys.path.append(str(p)) if str(p) not in sys.path else None