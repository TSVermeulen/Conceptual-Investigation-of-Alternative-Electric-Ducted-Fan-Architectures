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
        The filename or path of the .dill file to be loaded. 
        If not an absolute path, it will be relative to base_dir. 
    - base_dir : Path, optional
        The base directory to use if fname is not an absolute path.
        Defaults to the directory containing this script. 
    
    Returns
    - res : object
        The reconstructed pymoo optimisation results object
    """

    # If base_dir is not provided, use the script's directory
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    # Convert fname to Path and resolve it if it's not alredy absolute
    fname_path = Path(fname)
    if not fname_path.is_absolute():
        results_path = base_dir / fname_path
    else:
        results_path = fname_path
    
    # Validate file extension
    if results_path.suffix.lower() != '.dill':
        raise ValueError(f"File must have .dill extension. Got: {results_path.suffix}")

    try:
        # Open and load the results file
        with results_path.open('rb') as f:
            # ignore=False ensures we get an error if the object cannot be reconstructed. 
            res = dill.load(f, ignore=False)
            
        return res
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Results file not found. Ensure {fname} exists") from e

    except Exception as e:
        raise Exception(f"Error loading results: {e}") from e


if __name__ == "__main__":
    output = 'res_pop2_gen2_250506154138455301.dill'
    main(output)