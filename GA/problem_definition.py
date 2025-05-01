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

import os
import sys
import numpy as np
import shutil
import uuid
from pathlib import Path
import datetime
from pymoo.core.problem import ElementwiseProblem
from scipy import interpolate

# Add the parent and submodels paths to the system path if they are not already in the path
parent_path = str(Path(__file__).resolve().parent.parent)
submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

if parent_path not in sys.path:
    sys.path.append(parent_path)

if submodels_path not in sys.path:
    sys.path.append(submodels_path)

# Import interface submodels and other dependencies
from Submodels.MTSOL_call import OutputType
from Submodels.Parameterizations import AirfoilParameterization
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

        # Initialize the AirfoilParameterization class for slightly better memory usage
        self.Parameterization = AirfoilParameterization()
                

    def GenerateAnalysisName(self) -> str:
        """
        Generate a unique analysis name with a length of 32 characters.
        This is required to enable multi-threading of the optimization problem, and log each state file,
        since each evaluation of MTFLOW requires a unique set of files. 

        Returns
        -------
        - analysis_name : str
            A unique analysis name based on the current date and time and a unique identifier.
        """

        # Generate a timestamp string in the format MMDDHHMMSS
        now = datetime.datetime.now()
        timestamp = now.strftime(self.timestamp_format)

        # Generate a unique identifier using UUID
        unique_id = str(uuid.uuid4().hex)[:16]  # 16 chars max

        # Add a process ID to the analysis name to ensure uniqueness in multi-threaded environments.
        process_id = os.getpid() % 10000  # 4 chars max

        # The analysis name is formatted as: <MMDDHHMMSS>_<process_ID>_<unique_id>. with a maximum total length of 32 characters
        analysis_name = self.analysis_name_template.format(timestamp, process_id, unique_id)[:32]
        
        return analysis_name


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
        and write it to the oper dictionary.

        Returns
        -------
        None
        """

        # Compute the non-dimensional rotational rate Omega for MTFLOW and write it to self.oper
        # Multiplied by -1 to comply with sign convention in MTFLOW. 
        self.oper["Omega"] = float((-self.oper["RPS"] * np.pi * 2 * self.Lref) / (self.oper["Vinl"]))


    def SetOmega(self) -> None:
        """
        A simple function to correctly set the rotational rate Omega in the blading list(s).

        Returns
        -------
        None
        """

        rotating = config.ROTATING
        for i, blading_params in enumerate(self.blade_blading_parameters):
            if rotating[i]:
                blading_params["rotational_rate"] = self.oper["Omega"]
            else:
                blading_params["rotational_rate"] = 0


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


    def ComputeDuctRadialLocation(self) -> None:
        """
        Compute the y-coordinate of the LE of the duct based on the design variables. 

        Returns
        -------
        None
        """

        # Initialize data array for the radial duct coordinates
        radial_duct_coordinates = np.asarray(self.blade_diameters, dtype=float) / 2

        # Compute the duct x,y coordinates. Note that we are only interested in the lower surface.
        _, _, lower_x, lower_y = self.Parameterization.ComputeProfileCoordinates([self.duct_variables["b_0"],
                                                                                  self.duct_variables["b_2"],
                                                                                  self.duct_variables["b_8"],
                                                                                  self.duct_variables["b_15"],
                                                                                  self.duct_variables["b_17"]],
                                                                                  self.duct_variables)
        lower_x = lower_x * self.duct_variables["Chord Length"]
        lower_y = lower_y * self.duct_variables["Chord Length"]

        # Shift the duct x coordinate to the correct location in space
        lower_x += self.duct_variables["Leading Edge Coordinates"][0]

        # Construct cubic spline interpolant of the duct surface
        duct_interpolant = interpolate.CubicSpline(lower_x,
                                                   np.abs(lower_y),  # Take absolute value of y-coordinates since we need the distance, not the actual coordinate
                                                   extrapolate=False) 

        rot_flags = config.ROTATING
        x_min, x_max = lower_x[0], lower_x[-1]
        tip_gap = config.tipGap
        for i in range(self.num_stages):
            blading = self.blade_blading_parameters[i]
            y_tip = self.blade_diameters[i] / 2
            if not rot_flags[i]:
                continue
        
            sweep = np.tan(blading["sweep_angle"][-1])
            x_tip_LE = blading["root_LE_coordinate"] + sweep * y_tip
            projected_chord = blading["chord_length"][-1] * np.cos(np.pi/2 - 
                                                                   (blading["blade_angle"][-1] + blading["ref_blade_angle"] - blading["reference_section_blade_angle"]))
            
            x_tip_TE = x_tip_LE + projected_chord

            # Compute the offsets for the LE and TE of the blade tip
            LE_offset = float(duct_interpolant(x_tip_LE)) if x_min <= x_tip_LE <= x_max else 0  # Set to 0 if duct does not lie above LE
            TE_offset = float(duct_interpolant(x_tip_TE)) if x_min <= x_tip_TE <= x_max else 0  # Set to 0 if duct does not lie above TE

            # Compute the radial location of the duct
            radial_duct_coordinates[i] = y_tip + tip_gap + max(LE_offset, TE_offset)

        # Update the duct variables in self
        self.duct_variables["Leading Edge Coordinates"] = (self.duct_variables["Leading Edge Coordinates"][0],
                                                           np.max(radial_duct_coordinates))


    def CheckBlades(self) -> None:
        """
        Function to check the validity of the blade geometry by evaluating the blade parameters at each radial section.
        Simply constructs the blade profile x,y for each section of each stage. If there is any invalid section, it will throw an error, which will be catched by the try-except block in _evaluate.

        Returns
        -------
        None
        """

        for i in range(self.num_stages):
            if config.OPTIMIZE_STAGE[i]:
                for j in range(self.num_radial[i]):
                    blade_section = self.blade_design_parameters[i][j]
                    self.Parameterization.ComputeProfileCoordinates([blade_section["b_0"],
                                                                     blade_section["b_2"],
                                                                     blade_section["b_8"],
                                                                     blade_section["b_15"],
                                                                     blade_section["b_17"]],
                                                                     blade_section)


    def CheckCenterbody(self) -> None:
        """
        Function to check the validity of the centerbody geometry by evaluating the profile x,y. If the design is invalid, it will throw a ValueError, 
        which will be catched by the try-except block in _evaluate
        """

        if config.OPTIMIZE_CENTERBODY:
           self.Parameterization.ComputeProfileCoordinates([self.centerbody_variables["b_0"],
                                                            self.centerbody_variables["b_2"],
                                                            self.centerbody_variables["b_8"],
                                                            self.centerbody_variables["b_15"],
                                                            self.centerbody_variables["b_17"]],
                                                            self.centerbody_variables)
                  

    def _evaluate(self, 
                  x:dict, 
                  out:dict, 
                  *args, 
                  **kwargs) -> None:
        """
        Element-wise evaluation function.
        """

        # Construct key for design vector in cache
        key = tuple(sorted(x.items()))

        # Obtain current cache
        self.cache = kwargs.pop('cache', None)

        # Check if key in cache
        if self.cache is not None and key in self.cache:
            out.update(self.cache[key])
            return

        # Lazy import the MTFLOW interface and output_handling interface
        # This helps to improve startup time and memory usage in parallel workers
        from MTFLOW_caller import MTFLOW_caller
        from Submodels.output_handling import output_processing
        
        # Generate a unique analysis name
        self.analysis_name = self.GenerateAnalysisName()
        
        # Deconstruct the design vector
        self.centerbody_variables, self.duct_variables, self.blade_design_parameters, self.blade_blading_parameters, self.blade_diameters, self.Lref = DesignVectorInterface(x).DeconstructDesignVector()

        # Compute the necessary inputs (Reynolds, Omega)
        self.oper = config.oper.copy()
        self.ComputeReynolds()
        self.ComputeOmega()
        self.SetOmega()
        
        # Check validity of the duct, centerbody and blades
        design_okay = True
        try:
            self.ComputeDuctRadialLocation()
            self.CheckBlades()
            self.CheckCenterbody()
        except ValueError:
            # If a value error occurs with interpolation of the duct surface, this is an indication that the duct geometry is invalid. so we can set a crash flag and skip the MTFLOW analysis
            design_okay = False

        # Initialize the MTFLOW caller class
        if design_okay:
            MTFLOW_interface = MTFLOW_caller(operating_conditions=self.oper,
                                            centrebody_params=self.centerbody_variables,
                                            duct_params=self.duct_variables,
                                            blading_parameters=self.blade_blading_parameters,
                                            design_parameters=self.blade_design_parameters,
                                            ref_length=self.Lref,
                                            analysis_name=self.analysis_name,
                                            **kwargs)

            # Run MTFLOW
            MTFLOW_interface.caller(external_inputs=False,
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
        # self.CleanUpFiles()

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