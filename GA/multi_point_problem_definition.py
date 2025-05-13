"""
problem_definition-multistage
==================

Description
-----------
This module defines an optimization problem for a multi-point optimisation at n operating points for the pymoo framework, based on the ElementwiseProblem parent class. 
The model is based on the single-point optimisation routine in the problem_definition.py file

Classes
-------
MultiPointOptimizationProblem(ElementwiseProblem)
    Class defining the optimization problem with mixed-variable support.

Examples
--------
>>> problem = MultiPointOptimizationProblem()
>>> out = {}
>>> problem._evaluate(1, out)

Notes
-----
This module integrates with the MTFLOW executable for aerodynamic analysis. Ensure that the executable and required 
input files are present in the appropriate directories. The module is designed to handle mixed-variable optimization 
problems, including real and integer variables.

References
----------
For more details on the MTFLOW solver and its input/output requirements, refer to the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial implementation. 
"""

# Import standard libraries
import os
import shutil
import uuid
from pathlib import Path
import datetime

# Import 3rd party libraries
import numpy as np
from pymoo.core.problem import ElementwiseProblem

# Ensure all paths are correctly setup
from utils import ensure_repo_paths
ensure_repo_paths()

# Import interface submodels and other dependencies
from Submodels.MTSOL_call import OutputType
from objectives import Objectives
from constraints import Constraints
from init_designvector import DesignVector
from design_vector_interface import DesignVectorInterface
import config


class MultiPointOptimizationProblem(ElementwiseProblem):
    """
    Class definition of the optimization problem to be solved using the genetic algorithm. 
    Inherits from the ElementwiseProblem class from pymoo.core.problem.
    """

    # Define the file names relevant for MTFLOW
    FILE_TEMPLATES = {"walls": "walls.{}",
                      "tflow": "tflow.{}",
                      "forces": "forces.{}",
                      "flowfield": "flowfield.{}",
                      "boundary_layer": "boundary_layer.{}",
                      "tdat": "tdat.{}"}
    
    # Define the path for the tflow file
    _tflow_file_path = Path(__file__).parents[1] / "Submodels"

    def __init__(self,
                 **kwargs) -> None:
        """
        Initialization of the OptimizationProblem class. 

        Returns
        -------
        None
        """

        # Import control variables
        self.num_radial = config.NUM_RADIALSECTIONS
        self.num_stages = config.NUM_STAGES
        self.optimize_stages = config.OPTIMIZE_STAGE

        # Initialize variable list with variable types.
        design_vars = DesignVector()._construct_vector(config)

        # Initialize the parent class
        super().__init__(vars=design_vars,
                         n_obj=len(config.objective_IDs),
                         n_ieq_constr=len(config.constraint_IDs[0]),
                         n_eq_constr=len(config.constraint_IDs[1]),
                         **kwargs)
        
        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"
        
        # Create folder path to store statefiles
        self.dump_folder = self.submodels_path / "Evaluated_tdat_state_files"
        # Check existance of dump folder
        self.dump_folder.mkdir(exist_ok=True,
                               parents=True)

        # Define analysisname template
        self.timestamp_format = "%m%d%H%M%S"
        self.analysis_name_template = "{}_{:04d}_{}"

        # Initialize design vector interface
        self.design_vector_interface = DesignVectorInterface()

        # Initialize output dictionary to use in case of an infeasible design. 
        # This equals the outputs of the output_handling.output_processing.GetAllVariables(3) method, 
        # but is quicker as it does not involve reading a file.
        self.crash_outputs = {'data':
                              {'Total power CP': 0.00000, 
                               'EtaP': 0.00000, 
                               'Total force CT': 0.00000, 
                               'Element 2 top CTV': 0.00000, 
                               'Element 2 bot CTV': 0.00000, 
                               'Axis body CTV': 0.00000, 
                               'Viscous CTv': 0.00000, 
                               'Inviscid CTi': 0.00000, 
                               'Friction CTf': 0.00000, 
                               'Pressure CTp': 0.00000, 
                               'Pressure Ratio': 0.00000}, 
                              'grouped_data': 
                              {'Element 2': 
                               {'CTf': 0.00000, 
                                'CTp': 0.00000, 
                                'top Xtr': 0.00000, 
                                'bot Xtr': 0.00000}, 
                               'Axis Body': 
                               {'CTf': 0.00000, 
                                'CTp': 0.00000, 
                                'Xtr': 0.00000}}}
                

    def GenerateAnalysisName(self) -> None:
        """
        Generate a unique analysis name.
        This is required to enable multi-threading of the optimization problem, and log each state file,
        since each evaluation of MTFLOW requires a unique set of files. 

        Returns
        -------
        - None
            The analysis_name is written to self.
        """

        # Generate a timestamp string in the format MMDDHHMMSS
        now = datetime.datetime.now()
        timestamp = now.strftime(self.timestamp_format)

        # Generate a unique identifier using UUID
        unique_id = uuid.uuid4().hex[:8]  # 8 chars max

        # Add a process ID to the analysis name to ensure uniqueness in multi-threaded environments.
        process_id = os.getpid() % 10000  # 4 chars max

        # The analysis name is formatted as: <MMDDHHMMSS>_<process_ID>_<unique_id>.
        # Analysis name has a length of 24 characters, satisfying the maximum length of 32 characters accepted by MTFLOW. 
        self.analysis_name = self.analysis_name_template.format(timestamp, process_id, unique_id)

        # Additionally update the tflow file path
        self._tflow_file_path = self._tflow_file_path / f"tflow.{self.analysis_name}"


    def ComputeReynolds(self) -> None:
        """
        A simple function to compute the inlet Reynolds number,
        and write it to the oper dictionary.

        Returns
        -------
        None
        """

        # Compute the inlet Reynolds number and write it to self.oper
        # Uses Vinl [m/s], Lref [m], and kinematic_viscosity [m^2/s]
        self.oper["Inlet_Reynolds"] = round(float((self.oper["Vinl"] * self.Lref) / config.atmosphere.kinematic_viscosity[0]), 3)


    def ComputeOmega(self,
                     idx : int) -> None:
        """
        A simple function to compute the non-dimensional MTFLOW rotational rate Omega,
        and write it to the oper dictionary.

        Parameters
        ----------
        - idx : int
            The index of the operating condition in the multi_oper dictionary. 
            This is used to extract the correct RPS from the blading dictionary.

        Returns
        -------
        None
        """

        # Compute the non-dimensional rotational rate Omega for MTFLOW and write it to the blading parameters
        # Multiplied by -1 to comply with sign convention in MTFLOW. 
        for blading_params in self.blade_blading_parameters:
            blading_params["RPS"] = blading_params["RPS_lst"][idx]
            blading_params["rotational_rate"] = float((-blading_params["RPS"] * np.pi * 2 * self.Lref) / (self.oper["Vinl"]))


    def SetOmega(self,
                 oper_idx) -> None:
        """
        A simple function to correctly set the rotational rate Omega in the tflow.analysis_name file.
        This is used in a multi-point analysis to update the tflow file for each analysis rather than regenerating the full tflow file.

        Parameters
        ----------
        - oper_idx : int
            The index of the operating condition in the multi_oper dictionary. 
            This is used to extract the correct RPS from the blading dictionary.

        Returns
        -------
        None
        """

        # Compute / update the rotational rates in the blading dictionaries
        self.ComputeOmega(idx=oper_idx)

        # Open the tflow.analysis_name file
        with open(self._tflow_file_path, "r") as file:
            lines = file.readlines()

        match_counter = 0
        for idx, line in enumerate(lines):
            if line.startswith("OMEGA"):
                rate = self.blade_blading_parameters[match_counter]["rotational_rate"]
                lines[idx + 1] = f"{rate} \n"
                match_counter += 1
        
        # Write the updated tflow data back to the file
        with open(self._tflow_file_path, "w") as file:
            file.writelines(lines)


    def CleanUpFiles(self) -> None:
        """
        Archive the MTFLOW statefile to a separate folder and clean up temporary files.

        This method:
        1. Copies the tdat statefile to a persistent archive folder.
        2. Removes all temporary MTFLOW input/output files, including the original statefile.
        
        Note that the output files can always be regenerated from the statefile.
    
        Returns
        -------
        None
        """

        # Files to be deleted directly
        file_types = ["walls", "tflow", "forces", "flowfield", "boundary_layer", "tdat"]

        for file_type in file_types:
            # Construct filepath
            file_path = self.submodels_path / self.FILE_TEMPLATES[file_type].format(self.analysis_name)

            # Archive the state file if it exists
            if file_type == "tdat": 
                if file_path.exists():
                    copied_file = self.dump_folder / self.FILE_TEMPLATES[file_type].format(self.analysis_name)
                    try:
                        os.replace(file_path, copied_file)
                    except OSError:
                        shutil.move(file_path, copied_file)
            else:
                if file_path.exists():
                    # Cleanup all temporary files
                    file_path.unlink(missing_ok=True)


    def GenerateMTFLOWInputs(self) -> bool:
        """
        Generates the input files required for the MTFLOW simulation.
        This method creates the necessary input files for the MTFLOW simulation by utilizing the 
        `fileHandling` class from the `Submodels.file_handling` module. It generates two input files:
        - walls.analysis_name: The MTSET input file, which contains the axisymmetric geometries.
        - tflow.analysis_name: The MTFLO blading input file, which contains the blading and design parameters.

        By generating the input files, validation of the design vector is performed, since an infeasible design vector 
        will raise a ValueError (somewhere) in the input generation method.

        Returns
        -------
        - output_generated: bool
            - True if the input files were successfully generated, False if a ValueError occurred 
              during the process (indicating potential interpolation issues or infeasible axisymmetric bodies).
        """   

        # Lazy import the file_handling class
        from Submodels.file_handling import fileHandling

        # Create a file_handling parent-class instance
        file_handler = fileHandling()

        # Generate the MTSET input file containing the axisymmetric geometries and the MTFLO blading input file
        try:
            file_handler.fileHandlingMTSET(params_CB=self.centerbody_variables,
                                           params_duct=self.duct_variables,
                                           case_name=self.analysis_name,
                                           ref_length=self.Lref).GenerateMTSETInput()  # Generate the MTSET input file
            
            file_handler.fileHandlingMTFLO(case_name=self.analysis_name,
                                           ref_length=self.Lref).GenerateMTFLOInput(blading_params=self.blade_blading_parameters,
                                                                                    design_params=self.blade_design_parameters,
                                                                                    plot=False)  # Generate the MTFLO input file
            
            output_generated =  True  # If both input generation routines succeeded, set output_generated to True

            

        except ValueError as e:
            # Any value error that might occur while generating the MTSET input file will be caused by interpolation issues arising from the input values, so 
            # this is an efficient and simple method to check if the axisymmetric bodies are feasible. 
            output_generated = False  # If any of the input generation routines raised an error, set output_generated to False
            error_code = "INVALID_DESIGN"
            print(f"[{error_code}] Invalid design vector encountered: {e}")
        except Exception as e:
            # If any unexpected errors occur, log them as well
            output_generated = False
            error_code = f"UNEXPECTED_{type(e).__name__}"
            print(f"[{error_code}] Unexpected error in input generation: {e}")
        
        return output_generated
                  

    def _evaluate(self, 
                  x:dict, 
                  out:dict, 
                  *args, 
                  **kwargs) -> None:
        """
        Element-wise evaluation function.
        """

        # Lazy import the MTFLOW interface and output_handling interface
        # This helps to improve startup time and memory usage in parallel workers
        from MTFLOW_caller import MTFLOW_caller
        from Submodels.output_handling import output_processing
        
        # Generate a unique analysis name
        self.GenerateAnalysisName()
        
        # Deconstruct the design vector
        try:
            (self.centerbody_variables, 
            self.duct_variables, 
            self.blade_design_parameters, 
            self.blade_blading_parameters, 
            self.Lref) = self.design_vector_interface.DeconstructDesignVector(x_dict=x)
            deconstruction_okay = True
        except ValueError as e:
            deconstruction_okay = False
            error_code = "INVALID_DESIGN"
            print(f"[{error_code}] Invalid design vector encountered: {e}")

        # Generate the MTFLOW input files.
        # If design_okay is false, this indicates an error in the input file generation caused by an infeasible design vector. 
        if deconstruction_okay:
            design_okay = self.GenerateMTFLOWInputs()
        else:
            design_okay = deconstruction_okay

        # Only perform the MTFLOW analyses if the input generation has succeeded.
        # Initialise the MTFLOW output list of dictionaries. Use the crash outputs in 
        # initialisation to pre-populate them in case of a crash or infeasible design vector
        self.multi_oper = config.multi_oper.copy()
        MTFLOW_outputs = [self.crash_outputs] * len(self.multi_oper) 
        if design_okay:
            for idx, operating_point in enumerate(self.multi_oper):
                # Compute the necessary inputs
                self.oper = operating_point.copy()  # Copy the appropriate operating condition dictionary
                self.ComputeReynolds()
                self.SetOmega(oper_idx=idx)

                MTFLOW_interface = MTFLOW_caller(operating_conditions=self.oper,
                                             ref_length=self.Lref,
                                             analysis_name=self.analysis_name,
                                             **kwargs)

                # Run MTFLOW
                MTFLOW_interface.caller(external_inputs=True,
                                        output_type=OutputType.FORCES_ONLY)

                # Extract outputs
                output_handler = output_processing(analysis_name=self.analysis_name)
                MTFLOW_outputs[idx] = output_handler.GetAllVariables(output_type=3)

        # Obtain objective(s)
        # The out dictionary is updated in-place
        Objectives(self.duct_variables).ComputeMultiPointObjectives(analysis_outputs=MTFLOW_outputs,
                                                                    objective_IDs=config.objective_IDs,
                                                                    out=out)

        # Compute constraints
        # The out dictionary is updated in-place
        Constraints().ComputeMultiPointConstraints(analysis_outputs=MTFLOW_outputs,
                                                   Lref=self.Lref,
                                                   oper=self.multi_oper,
                                                   out=out)

        # Cleanup the generated files
        try:
            self.CleanUpFiles()
        except Exception as e:
            print(f"Warning: Failed to clean up files for {self.analysis_name}: {e}")
    

if __name__ == "__main__":
    # Disable parameterizations to allow for testing with empty design vector
    config.OPTIMIZE_CENTERBODY = False
    config.OPTIMIZE_DUCT = False
    config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)

    test = MultiPointOptimizationProblem()

    output = {}
    test._evaluate({}, output)

    print(output)