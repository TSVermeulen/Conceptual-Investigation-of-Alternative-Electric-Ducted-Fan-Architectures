import dill
from pathlib import Path
 

def main():
    """Load and display optimization results from the res.dill file."""
    try:
        # Get the parent directory path
        parent_dir = Path(__file__).resolve().parent.parent
        result_path = parent_dir / 'res.dill'

        # Open and load the results file
        with result_path.open('rb') as f:
            res = dill.load(f)

        # Print the objective function values
        print("Objective function values:")
        print(res.F)
        
    except FileNotFoundError:
        print("Error: Results file not found. Ensure .dill exists.")
    
    except Exception as e:
        print(f"Error loading results: {e}")


if __name__ == "__main__":
    main()