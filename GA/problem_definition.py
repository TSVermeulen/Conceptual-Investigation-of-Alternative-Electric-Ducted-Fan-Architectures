"""
problem_definition
==================

Description
-----------
This module defines an optimization problem for the pymoo framework, based on the ElementwiseProblem parent class. 

Classes
-------
OptimizationProblem(ElementwiseProblem)
    Class defining the optimization problem with mixed-variable support.

Examples
--------
>>> problem = OptimizationProblem()
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
Version: 1.2

Changelog:
- V1.0: Initial implementation. 
- V1.1: Improved documentation. Fixed issues with deconstruction of design vector. Fixed analysisname generator and switched to using datetime & evaluation counter for name generation. 
- V1.1.5: Changed analysis name generation to only use datetime to simplify naming generation. 
- V1.1.6: Updated to remove iter_count from MTFLOW_caller outputs
- V1.2: Extracted design vector handling to separate file/class
"""

# Import standard libraries
import os
import sys
import numpy as np
import shutil
import uuid
from pathlib import Path
import datetime

# Import 3rd party libraries
from pymoo.core.problem import ElementwiseProblem

# Add the parent and submodels paths to the system path if they are not already in the path
parent_path = str(Path(__file__).resolve().parent.parent)
submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

if parent_path not in sys.path:
    sys.path.append(parent_path)

if submodels_path not in sys.path:
    sys.path.append(submodels_path)

# Import interface submodels and other dependencies
from Submodels.MTSOL_call import OutputType
from objectives import Objectives
from constraints import Constraints
from init_designvector import DesignVector
from design_vector_interface import DesignVectorInterface
import config

class OptimizationProblem(ElementwiseProblem):
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
    

    def __init__(self,
                 **kwargs) -> None:
        """
        Initialization of the OptimizationProblem class. 

        Parameters
        ----------
        - **kwargs : dict[str, Any]
            Additional keyword arguments

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

        # Initialize cache
        self.cache = kwargs.pop('cache', None)

        # Initialize design vector interface
        self.design_vector_interface = DesignVectorInterface()
                

    def GenerateAnalysisName(self) -> None:
        """
        Generate a unique analysis name with a length of 32 characters.
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
        unique_id = str(uuid.uuid4().hex)[:16]  # 16 chars max

        # Add a process ID to the analysis name to ensure uniqueness in multi-threaded environments.
        process_id = os.getpid() % 10000  # 4 chars max

        # The analysis name is formatted as: <MMDDHHMMSS>_<process_ID>_<unique_id>. with a maximum total length of 32 characters
        self.analysis_name = self.analysis_name_template.format(timestamp, process_id, unique_id)[:32]


    def ComputeReynolds(self) -> None:
        """
        A simple function to compute the inlet Reynolds number,
        and write it to the oper dictionary.

        Returns
        -------
        None
        """

        # Compute the inlet Reynolds number and write it to self.oper
        self.oper["Inlet_Reynolds"] = round(float((self.oper["Vinl"] * self.Lref) / config.atmosphere.kinematic_viscosity[0]), 3)


    def ComputeOmega(self) -> None:
        """
        A simple function to compute the non-dimensional MTFLOW rotational rate Omega,
        and write it to the oper dictionary and the.

        Returns
        -------
        None
        """

        # Compute the non-dimensional rotational rate Omega for MTFLOW and write it to the blading parameters
        # Multiplied by -1 to comply with sign convention in MTFLOW. 
        for i, blading_params in enumerate(self.blade_blading_parameters):
            blading_params["rotational_rate"] = float((-blading_params["RPS"] * np.pi * 2 * self.Lref) / (self.oper["Vinl"]))


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

            if not file_path.exists():
                continue
            
            # Archive the state file
            if file_type == "tdat": 
                copied_file = self.dump_folder / self.FILE_TEMPLATES[file_type].format(self.analysis_name)
                try:
                    os.replace(file_path, copied_file)
                except OSError:
                    shutil.move(file_path, copied_file)
            else:
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
            print(f"Invalid design vector encountered: {e}")
        
        return output_generated
        

    def _evaluate(self, 
                  x:dict, 
                  out:dict, 
                  *args, 
                  **kwargs) -> None:
        """
        Element-wise evaluation function.

        Parameters
        ----------
        - x : dict
            The pymoo design vector dictionary.
        - out : dict
            The pymoo elementwise evaluation output dictionary.
        - *args : tuple
            Additional arguments
        - **kwargs : dict[str, Any]
            Additional keyword arguments

        Returns
        -------
        - None
            The output dictionary is modified in-place. 
        """

        # Construct key for design vector in cache
        key = tuple(sorted(x.items()))

        # Check if key in cache
        if self.cache is not None and key in self.cache:
            out.update(self.cache[key])
            return

        # Lazy import the MTFLOW interface and output_handling interface
        # This helps to improve startup time and memory usage in parallel workers
        from MTFLOW_caller import MTFLOW_caller
        from Submodels.output_handling import output_processing
        
        # Generate a unique analysis name
        self.GenerateAnalysisName()
        
        # Deconstruct the design vector
        (self.centerbody_variables, 
         self.duct_variables, 
         self.blade_design_parameters, 
         self.blade_blading_parameters, 
         self.Lref) = self.design_vector_interface.DeconstructDesignVector(x_dict=x)

        # Compute the necessary inputs (Reynolds, Omega)
        self.oper = config.oper.copy()
        self.ComputeReynolds()
        self.ComputeOmega()
        
        # Generate the MTFLOW input files.
        # If design_okay is false, this indicates an error in the input file generation caused by an infeasible design vector. 
        design_okay = self.GenerateMTFLOWInputs()

        # Initialize the MTFLOW caller class
        if design_okay:
            MTFLOW_interface = MTFLOW_caller(operating_conditions=self.oper,
                                             ref_length=self.Lref,
                                             analysis_name=self.analysis_name,
                                             **kwargs)

            # Run MTFLOW
            MTFLOW_interface.caller(external_inputs=True,
                                    output_type=OutputType.FORCES_ONLY)

            # Extract outputs
            output_handler = output_processing(analysis_name=self.analysis_name)
        else:
            # If the design is infeasible, we load the crash outputs
            # This is a predefined file with all outputs set to 0.
            output_handler = output_processing(analysis_name="crash_outputs")
        
        MTFLOW_outputs = output_handler.GetAllVariables(output_type=3)

        # Obtain objective(s)
        # The out dictionary is updated in-place
        Objectives().ComputeObjective(analysis_outputs=MTFLOW_outputs,
                                      objective_IDs = config.objective_IDs,
                                      out=out)

        # Compute constraints
        # The out dictionary is updated in-place
        Constraints().ComputeConstraints(analysis_outputs=MTFLOW_outputs,
                                         Lref=self.Lref,
                                         out=out)
        
        # Cleanup the generated files
        self.CleanUpFiles()

        # Add result to cache
        if self.cache is not None:
            self.cache[key] = out.copy()
    

if __name__ == "__main__":
    # Disable parameterizations to allow for testing with empty design vector
    config.OPTIMIZE_CENTERBODY = False
    config.OPTIMIZE_DUCT = False
    config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)

    test = OptimizationProblem()

    output = {}
    test._evaluate({}, output)

    print(output)