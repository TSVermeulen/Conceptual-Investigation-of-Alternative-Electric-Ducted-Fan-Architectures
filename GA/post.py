import dill
from pathlib import Path

# Ensure all paths are correctly setup
from utils import ensure_repo_paths
ensure_repo_paths()
 

def main(fname: Path,
         base_dir: Path = None) -> object:
    """
    Load and return the optimization results from the specified .dill file.

    Parameters
    ----------
    - fname : Path
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
    if not fname.is_absolute():
        results_path = base_dir / fname
    else:
        results_path = fname
    
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

    except dill.UnpicklingError as e:
        raise RuntimeError(f"Error loading results: {e}") from e


if __name__ == "__main__":
    output = Path('res_pop2_gen2_250506154138455301.dill')
    main(output)