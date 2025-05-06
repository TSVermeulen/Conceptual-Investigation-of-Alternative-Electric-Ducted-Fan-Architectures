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
 

def main(fname: str):
    """Load and display optimization results from the res.dill file."""
    path = Path(__file__).resolve().parent
    results_path = path / fname
    print(results_path)
    try:
        # Open and load the results file
        with results_path.open('rb') as f:
            res = dill.load(f,
                            ignore=False)

        # Print the objective function values
        print("Objective function values:")
        print(res.F)
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Results file not found. Ensure .dill exists: {e}")

    except Exception as e:
        print(f"Error loading results: {e}")


if __name__ == "__main__":
    output = 'res_pop2_gen2_250506154138455301.dill'
    main(output)