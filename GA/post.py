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

        return 0
        
    except FileNotFoundError:
        print(f"Error: Results file not found. Ensure res.dill exists.")
        return 1
    
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1

if __name__ == "__main__":
    main()