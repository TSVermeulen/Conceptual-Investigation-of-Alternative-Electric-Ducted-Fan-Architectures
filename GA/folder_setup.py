"""
folder_setup
=====

Simple script to ensure all folders exist/are created

"""


from pathlib import Path

def setup() -> None:
    # Define main paths
    parent_dir = Path(__file__).resolve().parent.parent
    submodels_path = parent_dir / "Submodels"

    # Define non-convergence dump folder and create it if it doesn't exist already
    dump_folder_nonconv = submodels_path / "MTSOL_output_files"
    dump_folder_nonconv.mkdir(exist_ok=True)

    # Define convergence dump folder and create it if it doesn't exist already
    dump_folder_tdat = submodels_path / "Evaluated_tdat_state_files"
    dump_folder_tdat.mkdir(exist_ok=True)
