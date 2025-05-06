import dill
from pathlib import Path
import sys

# Add the parent and submodels paths to the system path if they are not already in the path
parent_path = str(Path(__file__).resolve().parent.parent)
submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

if parent_path not in sys.path:
    sys.path.append(parent_path)

if submodels_path not in sys.path:
    sys.path.append(submodels_path)
 

def main(fname: str) -> object:
    """
    Load and return the optimization results from the specified .dill file.

    Parameters
    ----------
    - fname : str
        The filename of the .dill file to be loaded. The file must be in the same directory as this file. 
    
    Returns
    - res : object
        The reconstructed pymoo optimisation results object
    """

    path = Path(__file__).resolve().parent
    results_path = path / fname
    try:
        # Open and load the results file
        with results_path.open('rb') as f:
            res = dill.load(f, ignore=False)
            
        return res
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Results file not found. Ensure {fname} exists") from e

    except Exception as e:
        raise Exception(f"Error loading results") from e


if __name__ == "__main__":
    output = 'res_pop2_gen2_250506154138455301.dill'
    main(output)