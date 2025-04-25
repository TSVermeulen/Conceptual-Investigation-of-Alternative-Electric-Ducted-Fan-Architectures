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
>>> problem = OptimizationProblem(obj_count=1)
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
Version: 1.1.5

Changelog:
- V1.0: Initial implementation. 
- V1.1: Improved documentation. Fixed issues with deconstruction of design vector. Fixed analysisname generator and switched to using datetime & evaluation counter for name generation. 
- V1.1.5: Changed analysis name generation to only use datetime to simplify naming generation. 
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

# Add the parent and submodels paths to the system path
sys.path.extend([str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent.parent / "Submodels")])

# Import MTFLOW interface submodels and other dependencies
from MTFLOW_caller import MTFLOW_caller
from Submodels.MTSOL_call import OutputType
from Submodels.output_handling import output_processing
from Submodels.Parameterizations import AirfoilParameterization
from objectives import Objectives
from constraints import Constraints
from init_designvector import DesignVector
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
        vars = DesignVector()._construct_vector(config)

        # Preconstruct design variable x-keys for more efficient indexing
        self.x_keys = list(vars.keys())

        # Initialize the parent class
        super().__init__(vars=vars,
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
        self.dump_folder.mkdir(exist_ok=True)

        # Define analysisname template
        self.timestamp_format = "%m%d%H%M%S"
        self.analysis_name_template = "{}_{:04d}_{}"
                

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
        analysis_name = self.analysis_name_template.format(timestamp, process_id, unique_id)
        
        return analysis_name

    
    def DeconstructDesignVector(self,
                                x: dict[str, float|int]) -> None:
        """
        Decompose the design vector x into dictionaries of all the design variables to match the expected input formats for 
        the MTFLOW code interface. 
        The design vector has the format: [centerbody, duct, blades]
        
        Parameters
        ----------
        - x : np.ndarray
            A 1D array of the design vector being analysed. 

        Returns
        -------
        None
        """

        # Define a helper function to access the design vector values
        def GetX(x_dict: dict,
                 base_idx: int,
                 offset: int = 0) -> float|int:
            """ 
            Helper function to access the design vector without repeated f-string formatting.
            If the key does not exist, raises a KeyError.
            """
            try:
                return x_dict[self.x_keys[base_idx + offset]]
            except KeyError as err:
                raise KeyError(f"Design vector key 'x{base_idx + offset} missing. Check design vector initialisation.") from err
        
        # Define a helper function to compute parameter b_8 using the mapping design variable
        def Getb8(b_8_map: float, 
                  r_le: float, 
                  x_t: float, 
                  y_t: float) -> float:
            """
            Helper function to compute the bezier parameter b_8 using the mapping parameter 0 <= b_8_map <= 1
            """
            
            term = -2 * r_le * x_t / 3
            if term <= 0:
                sqrt_term = 0
            else:
                sqrt_term = np.sqrt(term)

            return b_8_map * min(y_t, sqrt_term)

        # Define a pointer to count the number of variable parameters
        idx = 0
        centerbody_designvar_count = 8
        duct_designvar_count = 17

        # Deconstruct the centerbody values if it's variable.
        # If the centerbody is constant, read in the centerbody values from config.
        # Note that if the centerbody is variable, we keep the LE coordinate fixed, as the LE coordinate of the duct would already be free to move. 
        if config.OPTIMIZE_CENTERBODY:
            self.centerbody_variables = {"b_0": 0,
                                         "b_2": 0, 
                                         "b_8": Getb8(GetX(x, idx), GetX(x, idx, 5), GetX(x, idx, 2), GetX(x, idx, 3)),
                                         "b_15": GetX(x, idx, 1),
                                         "b_17": 0,
                                         "x_t": GetX(x, idx, 2),
                                         "y_t": GetX(x, idx, 3),
                                         "x_c": 0,
                                         "y_c": 0,
                                         "z_TE": 0,
                                         "dz_TE": GetX(x, idx, 4),
                                         "r_LE": GetX(x, idx, 5),
                                         "trailing_wedge_angle": GetX(x, idx, 6),
                                         "trailing_camberline_angle": 0,
                                         "leading_edge_direction": 0, 
                                         "Chord Length": GetX(x, idx, 7),
                                         "Leading Edge Coordinates": (0, 0)}
            
            # Update the index to point to the blade design variables, since we need the blade variables deconstructed first in order to correctly set the duct variables. 
            idx += (centerbody_designvar_count + duct_designvar_count) if config.OPTIMIZE_DUCT else centerbody_designvar_count
        else:
            self.centerbody_variables = config.CENTERBODY_VALUES
                
        # Deconstruct the rotorblade parametersPrecompute indices for rotorblade parameters if they are variable.
        # If the rotorblade parameters are constant, read in the parameters from config.
        self.blade_design_parameters = []
        for i in range(self.num_stages):
            # Initiate empty list for each stage
            stage_design_parameters = []
            if self.optimize_stages[i]:
                # If the stage is to be optimized, read in the design vector for the blade profiles
                for _ in range(self.num_radial):
                    # Loop over the number of radial sections and append each section to stage_design_parameters
                    section_parameters = {"b_0": GetX(x, idx),
                                        "b_2": GetX(x, idx, 1), 
                                        "b_8": Getb8(GetX(x, idx, 2), GetX(x, idx, 11), GetX(x, idx, 5), GetX(x, idx, 6)), 
                                        "b_15": GetX(x, idx, 3),
                                        "b_17": GetX(x, idx, 4),
                                        "x_t": GetX(x, idx, 5),
                                        "y_t": GetX(x, idx, 6),
                                        "x_c": GetX(x, idx, 7),
                                        "y_c": GetX(x, idx, 8),
                                        "z_TE": GetX(x, idx, 9),
                                        "dz_TE": GetX(x, idx, 10),
                                        "r_LE": GetX(x, idx, 11),
                                        "trailing_wedge_angle": GetX(x, idx, 12),
                                        "trailing_camberline_angle": GetX(x, idx, 13),
                                        "leading_edge_direction": GetX(x, idx, 14)}
                    idx += 15
                    stage_design_parameters.append(section_parameters)
            else:
                # If the stage is meant to be constant, read it in from config. 
                stage_design_parameters = config.STAGE_DESIGN_VARIABLES[i]
            # Write the stage nested list to blade_design_parameters
            self.blade_design_parameters.append(stage_design_parameters)

        self.blade_blading_parameters = []
        self.blade_diameters = []
        radial_linspace = np.linspace(0, 1, self.num_radial)
        for i in range(self.num_stages):
            # Initiate empty list for each stage
            stage_blading_parameters = {}
            if self.optimize_stages[i]:
                # If the stage is to be optimized, read in the design vector for the blading parameters
                stage_blading_parameters["root_LE_coordinate"] = GetX(x, idx)
                stage_blading_parameters["blade_count"] = int(round(GetX(x, idx, 1)))
                stage_blading_parameters["ref_blade_angle"] = GetX(x, idx, 2)
                stage_blading_parameters["reference_section_blade_angle"] = config.REFERENCE_SECTION_ANGLES[i]
                stage_blading_parameters["radial_stations"] = radial_linspace * GetX(x, idx, 3)  # Radial stations are defined as fraction of blade radius * local radius
                self.blade_diameters.append(GetX(x, idx, 3) * 2)

                # Initialize sectional blading parameter lists
                stage_blading_parameters["chord_length"] = [None] * self.num_radial
                stage_blading_parameters["sweep_angle"] = [None] * self.num_radial
                stage_blading_parameters["blade_angle"] = [None] * self.num_radial

                base_idx = idx + 4
                for j in range(self.num_radial):
                    # Loop over the number of radial sections and write their data to the corresponding lists
                    stage_blading_parameters["chord_length"][j]= GetX(x, base_idx, j)
                    stage_blading_parameters["sweep_angle"][j] = GetX(x, base_idx, self.num_radial + j)
                    stage_blading_parameters["blade_angle"][j] = GetX(x, base_idx, self.num_radial * 2 + j)
                idx = base_idx + 3 * self.num_radial                
            else:
                stage_blading_parameters = config.STAGE_BLADING_PARAMETERS[i]
                self.blade_diameters.append(config.BLADE_DIAMETERS[i])
            
            # Append the stage blading parameters to the main list
            self.blade_blading_parameters.append(stage_blading_parameters)
        
        # Write the reference length for MTFLOW
        self.Lref = self.blade_diameters[0]

        # Deconstruct the duct values if it's variable.
        # If the duct is constant, read in the duct values from config.
        # The duct parameters must be read in last, because the LE y coordinate of the duct is dependent on the blade rows to maintain a minimum tip gap. 
        if config.OPTIMIZE_DUCT:
            idx = centerbody_designvar_count if config.OPTIMIZE_CENTERBODY else 0

            self.duct_variables = {"b_0": GetX(x, idx),
                                   "b_2": GetX(x, idx, 1), 
                                   "b_8": Getb8(GetX(x, idx, 2), GetX(x, idx, 11), GetX(x, idx, 5), GetX(x, idx, 6)),
                                   "b_15": GetX(x, idx, 3),
                                   "b_17": GetX(x, idx, 4),
                                   "x_t": GetX(x, idx, 5),
                                   "y_t": GetX(x, idx, 6),
                                   "x_c": GetX(x, idx, 7),
                                   "y_c": GetX(x, idx, 8),
                                   "z_TE": GetX(x, idx, 9),
                                   "dz_TE": GetX(x, idx, 10),
                                   "r_LE": GetX(x, idx, 11),
                                   "trailing_wedge_angle": GetX(x, idx, 12),
                                   "trailing_camberline_angle": GetX(x, idx, 13),
                                   "leading_edge_direction": GetX(x, idx, 14), 
                                   "Chord Length": GetX(x, idx, 15),
                                   "Leading Edge Coordinates": (GetX(x, idx, 16), 0)}
            idx += 17
        else:
            self.duct_variables = config.DUCT_VALUES


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

        for i in range(len(self.blade_blading_parameters)):
            if config.ROTATING[i]:
                self.blade_blading_parameters[i]["rotational_rate"] = self.oper["Omega"]
            else:
                self.blade_blading_parameters[i]["rotational_rate"] = 0


    def CleanUpFiles(self) -> None:
        """
        Move the MTFLOW statefile to a separate folder to maintain clarity, and delete the no-longer needed output files. 
        Note that the output files can always be regenerated from the statefile.

        Returns
        -------
        None
        """

        # Precompute all file paths: 
        file_paths = { file_type: self.submodels_path / self.FILE_TEMPLATES[file_type].format(self.analysis_name) for file_type in ["walls", "tflow", "forces", "flowfield", "boundary_layer", "tdat"]}

        # Delete the walls, tflow, forces, flowfield, and boundary layer files if they exist
        [path.unlink() for path in 
         [file_paths[file_type] for file_type in ["walls", "tflow", "forces", "flowfield", "boundary_layer"]] 
         if path.exists()]

        # Move the state file into the dump_folder
        tdat_path = file_paths["tdat"]
        if tdat_path.exists():
            copied_file = self.dump_folder / self.FILE_TEMPLATES["tdat"].format(self.analysis_name)
            shutil.copy(tdat_path, 
                        copied_file)
            tdat_path.unlink()


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
        _, _, lower_x, lower_y = AirfoilParameterization().ComputeProfileCoordinates([self.duct_variables["b_0"],
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

        # Loop over all stages
        for i in range(self.num_stages):
            blading_params = self.blade_blading_parameters[i]

            # Compute the blade tip coordinate
            y_tip = self.blade_diameters[i] / 2
            
            if config.ROTATING[i]:
                # Compute the y value of the duct inner surface at the blade rotor
                sweep = np.tan(blading_params["sweep_angle"][-1])
                x_tip = blading_params["root_LE_coordinate"] + sweep * y_tip    

                y_tip_clearance = y_tip + config.tipGap
                if (x_tip <= lower_x[-1] and x_tip >= lower_x[0]):
                    # Only add the duct offset if the duct is over the blade row
                    radial_duct_coordinates[i] = y_tip_clearance + float(duct_interpolant(x_tip))
                else:
                    radial_duct_coordinates[i] = y_tip_clearance
    
        # Update the duct variables in self
        self.duct_variables["Leading Edge Coordinates"] = (self.duct_variables["Leading Edge Coordinates"][0],
                                                           np.max(radial_duct_coordinates))


    def _evaluate(self, x, out, *args, **kwargs) -> None:
        # Generate a unique analysis name
        self.analysis_name = self.GenerateAnalysisName()
        
        # Deconstruct the design vector
        self.DeconstructDesignVector(x)

        # Compute the necessary inputs (Reynolds, Omega)
        self.oper = config.oper.copy()
        self.ComputeReynolds()
        self.ComputeOmega()
        self.SetOmega()
        self.ComputeDuctRadialLocation()

        # Initialize the MTFLOW caller class
        MTFLOW_interface = MTFLOW_caller(operating_conditions=self.oper,
                                         centrebody_params=self.centerbody_variables,
                                         duct_params=self.duct_variables,
                                         blading_parameters=self.blade_blading_parameters,
                                         design_parameters=self.blade_design_parameters,
                                         ref_length=self.Lref,
                                         analysis_name=self.analysis_name,
                                         **kwargs)

        # Run MTFLOW
        _, _ = MTFLOW_interface.caller(external_inputs=False,
                                       output_type=OutputType.FORCES_ONLY)

        # Extract outputs
        output_handler = output_processing(analysis_name=self.analysis_name)
        MTFLOW_outputs = output_handler.GetAllVariables(3)

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
    

if __name__ == "__main__":
    # Disable parameterizations to allow for testing with empty design vector
    config.OPTIMIZE_CENTERBODY = False
    config.OPTIMIZE_DUCT = False
    for i in range(len(config.OPTIMIZE_STAGE)):
        config.OPTIMIZE_STAGE[i] = False

    test = OptimizationProblem()

    output = {}
    test._evaluate({}, output)

    print(output)