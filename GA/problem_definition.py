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
from pathlib import Path
import datetime
from pymoo.core.problem import ElementwiseProblem
from scipy import interpolate

# Add the parent and submodels paths to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
submodels_path = os.path.join(parent_dir, "Submodels")
sys.path.extend([parent_dir, submodels_path])

# Import MTFLOW interface submodels and other dependencies
from MTFLOW_caller import MTFLOW_caller
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

        # Initialize the parent class
        super().__init__(vars=vars,
                         n_obj=len(config.objective_IDs),
                         n_ieq_constr=len(config.constraint_IDs[0]),
                         n_eq_constr=len(config.constraint_IDs[1]),
                         **kwargs)
        
        # Change working directory to the parent folder
        try:
            os.chdir(parent_dir)
        except OSError as e:
            raise OSError from e
        

    def GenerateAnalysisName(self) -> str:
        """
        Generate a unique analysis name with a maximum length of 32 characters.
        This is required to enable multi-threading of the optimization problem, and log each state file,
        since each evaluation of MTFLOW requires a unique set of files. 

        Returns
        -------
        - analysis_name : str
            A unique analysis name based on the current date and time.
        """

        # Construct the analysis_name based on the current date and time
        now = datetime.datetime.now()	
        analysis_name = now.strftime("%Y%m%d_%H%M%S_%f")
        analysis_name = analysis_name[:32]

        return analysis_name

    
    def DeconstructDesignVector(self,
                                x: np.ndarray) -> None:
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
                                         "b_8": x[f"x{idx}"] * min(x[f"x{3 + idx}"], np.sqrt(-2 * x[f"x{5 + idx}"] * x[f"x{2 + idx}"] / 3)),
                                         "b_15": x[f"x{1 + idx}"],
                                         "b_17": 0,
                                         "x_t": x[f"x{2 + idx}"],
                                         "y_t": x[f"x{3 + idx}"],
                                         "x_c": 0,
                                         "y_c": 0,
                                         "z_TE": 0,
                                         "dz_TE": x[f"x{4 + idx}"],
                                         "r_LE": x[f"x{5 + idx}"],
                                         "trailing_wedge_angle": x[f"x{6 + idx}"],
                                         "trailing_camberline_angle": 0,
                                         "leading_edge_direction": 0, 
                                         "Chord Length": x[f"x{7 + idx}"],
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
                    section_parameters = {"b_0": x[f"x{idx}"],
                                        "b_2": x[f"x{1 + idx}"], 
                                        "b_8": x[f"x{2 + idx}"] * min(x[f"x{6 + idx}"], np.sqrt(-2 * x[f"x{11 + idx}"] * x[f"x{5 + idx}"] / 3)),
                                        "b_15": x[f"x{3 + idx}"],
                                        "b_17": x[f"x{4 + idx}"],
                                        "x_t": x[f"x{5 + idx}"],
                                        "y_t": x[f"x{6 + idx}"],
                                        "x_c": x[f"x{7 + idx}"],
                                        "y_c": x[f"x{8 + idx}"],
                                        "z_TE": x[f"x{9 + idx}"],
                                        "dz_TE": x[f"x{10 + idx}"],
                                        "r_LE": x[f"x{11 + idx}"],
                                        "trailing_wedge_angle": x[f"x{12 + idx}"],
                                        "trailing_camberline_angle": x[f"x{13 + idx}"],
                                        "leading_edge_direction": x[f"x{14 + idx}"]}
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
                stage_blading_parameters["root_LE_coordinate"] = x[f"x{idx}"]
                stage_blading_parameters["blade_count"] = x[f"x{1 + idx}"]
                stage_blading_parameters["ref_blade_angle"] = x[f"x{2 + idx}"]
                stage_blading_parameters["reference_section_blade_angle"] = config.REFERENCE_SECTION_ANGLES[i]
                stage_blading_parameters["radial_stations"] = radial_linspace * x[f"x{3 + idx}"]  # Radial stations are defined as fraction of blade radius * local radius
                self.blade_diameters.append(x[f"x{3 + idx}"] * 2)

                # Initialize sectional blading parameter lists
                stage_blading_parameters["chord_length"] = [None] * self.num_radial
                stage_blading_parameters["sweep_angle"] = [None] * self.num_radial
                stage_blading_parameters["blade_angle"] = [None] * self.num_radial

                base_idx = idx + 4
                for j in range(self.num_radial):
                    # Loop over the number of radial sections and write their data to the corresponding lists
                    stage_blading_parameters["chord_length"][j]= x[f"x{base_idx + j}"]
                    stage_blading_parameters["sweep_angle"][j] = x[f"x{base_idx + self.num_radial + j}"]
                    stage_blading_parameters["blade_angle"][j] = x[f"x{base_idx + 2 * self.num_radial + j}"]
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

            if np.any(self.optimize_stages):
                LE_coords = (x[f"x{16 + idx}"], 0)
            else:
                LE_coords = (x[f"x{16 + idx}"], 0)
            self.duct_variables = {"b_0": x[f"x{idx}"],
                                   "b_2": x[f"x{1 + idx}"], 
                                   "b_8": x[f"x{2 + idx}"] * min(x[f"x{6 + idx}"], np.sqrt(-2 * x[f"x{11 + idx}"] * x[f"x{5 + idx}"] / 3)),
                                   "b_15": x[f"x{3 + idx}"],
                                   "b_17": x[f"x{4 + idx}"],
                                   "x_t": x[f"x{5 + idx}"],
                                   "y_t": x[f"x{6 + idx}"],
                                   "x_c": x[f"x{7 + idx}"],
                                   "y_c": x[f"x{8 + idx}"],
                                   "z_TE": x[f"x{9 + idx}"],
                                   "dz_TE": x[f"x{10 + idx}"],
                                   "r_LE": x[f"x{11 + idx}"],
                                   "trailing_wedge_angle": x[f"x{12 + idx}"],
                                   "trailing_camberline_angle": x[f"x{13 + idx}"],
                                   "leading_edge_direction": x[f"x{14 + idx}"], 
                                   "Chord Length": x[f"x{15 + idx}"],
                                   "Leading Edge Coordinates": LE_coords}
            idx += 17
        else:
            self.duct_variables = config.DUCT_VALUES


    def ComputeReynolds(self) -> None:
        """
        A simple function to compute the inlet Reynolds number,
        and write it to the oper dictionary in config.py.

        Returns
        -------
        None
        """

        # Compute the inlet Reynolds number and write it to config.oper
        config.oper["Inlet_Reynolds"] = round(float((config.oper["Vinl"] * self.Lref) / config.atmosphere.kinematic_viscosity[0]), 3)


    def ComputeOmega(self) -> None:
        """
        A simple function to compute the non-dimensional MTFLOW rotational rate Omega,
        and write it to the oper dictionary in config.py.

        Returns
        -------
        None
        """

        # Compute the non-dimensional rotational rate Omega for MTFLOW and write it to config.oper
        config.oper["Omega"] = float((-config.oper["RPS"] * np.pi * 2 * self.Lref) / (config.oper["Vinl"]))


    def SetOmega(self) -> None:
        """
        A simple function to correctly set the rotational rate Omega in the blading list(s).

        Returns
        -------
        None
        """

        for i in range(len(self.blade_blading_parameters)):
            if config.ROTATING[i]:
                self.blade_blading_parameters[i]["rotational_rate"] = config.oper["Omega"]
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

        # Change working directory to the submodels folder
        current_dir = os.getcwd()
        os.chdir(submodels_path)

        # Delete the walls, tflow, forces, flowfield, and boundary layer files if they exist
        os.remove(f"walls.{self.analysis_name}") if os.path.exists(f"walls.{self.analysis_name}") else None 
        os.remove(f"tflow.{self.analysis_name}") if os.path.exists(f"tflow.{self.analysis_name}") else None 
        os.remove(f"forces.{self.analysis_name}") if os.path.exists(f"forces.{self.analysis_name}") else None 
        os.remove(f"flowfield.{self.analysis_name}") if os.path.exists(f"flowfield.{self.analysis_name}") else None 
        os.remove(f"boundary_layer.{self.analysis_name}") if os.path.exists(f"boundary_layer.{self.analysis_name}") else None 

        # Create folder to store statefiles if it does not exist yet. 
        dump_folder = Path("Evaluated_tdat_state_files")
        os.makedirs(dump_folder, 
                    exist_ok=True)
        
        # Move the state file into the dump_folder
        shutil.copy(f"tdat.{self.analysis_name}", dump_folder / f"tdat.{self.analysis_name}")
        os.remove(f"tdat.{self.analysis_name}")

        # Revert back to the original working directory
        os.chdir(current_dir)


    def ComputeDuctRadialLocation(self) -> None:
        """
        Compute the y-coordinate of the LE of the duct based on the design variables. 

        Returns
        -------
        None
        """

        # Initialize empty data array for the radial duct coordinates
        radial_duct_coordinates = np.zeros(self.num_stages)

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
        lower_x -= self.duct_variables["Leading Edge Coordinates"][0]

        # Construct cubic spline interpolant of the duct surface
        duct_interpolant = interpolate.CubicSpline(lower_x,
                                                   lower_y,
                                                   extrapolate=False)

        # Loop over all stages
        for i in range(self.num_stages):
            blading_params = self.blade_blading_parameters[i]

            # Compute the blade tip coordinate
            y_tip = self.blade_diameters[i] / 2
            
            if blading_params["rotational_rate"] != 0:
                # Compute the y value of the duct inner surface at the blade rotor
                x_tip = blading_params["root_LE_coordinate"] + np.tan(blading_params["sweep_angle"][-1]) * self.blade_diameters[i] / 2
                duct_y = duct_interpolant(x_tip,
                                          extrapolate=False)
                
                if not np.isnan(duct_y):
                    # Filter out NaN values and compute the y-distance between the LE of the duct and the blade row tip LE. 
                    # NaN would correspond to there being no duct at the blade station, in which case the offset would be zero.
                    radial_duct_coordinates[i] = y_tip + config.tipGap + np.abs(duct_y)
                else:
                    radial_duct_coordinates[i] = y_tip + config.tipGap
            else:
                # For a stator
                radial_duct_coordinates[i] = y_tip
    
        # The radial location of the LE of the duct is equal to the max value in radial_duct_coordinates    
        radial_duct_coordinate = np.max(radial_duct_coordinates)

        # Update the duct variables in self
        self.duct_variables["Leading Edge Coordinates"] = (self.duct_variables["Leading Edge Coordinates"][0],
                                                           radial_duct_coordinate)


    def _evaluate(self, x, out, *args, **kwargs) -> None:
        # Generate a unique analysis name
        self.analysis_name = self.GenerateAnalysisName()
        
        # Deconstruct the design vector
        self.DeconstructDesignVector(x)

        # Compute the necessary inputs (Reynolds, Omega)
        self.ComputeReynolds()
        self.ComputeOmega()
        self.SetOmega()
        self.ComputeDuctRadialLocation()

        # Initialize the MTFLOW caller class
        MTFLOW_interface = MTFLOW_caller(operating_conditions=config.oper,
                                         centrebody_params=self.centerbody_variables,
                                         duct_params=self.duct_variables,
                                         blading_parameters=self.blade_blading_parameters,
                                         design_parameters=self.blade_design_parameters,
                                         ref_length=self.Lref,
                                         analysis_name=self.analysis_name)

        # Run MTFLOW
        _, _ = MTFLOW_interface.caller(debug=False,
                                       external_inputs=False,
                                       output_type=0)

        # Extract outputs
        output_handler = output_processing(analysis_name=self.analysis_name)
        MTFLOW_outputs = output_handler.GetAllVariables(3)

        # Obtain objective(s)
        Objectives().ComputeObjective(analysis_outputs=MTFLOW_outputs,
                                      objective_IDs = config.objective_IDs,
                                      out=out)

        # Compute constraints
        Constraints().ComputeConstraints(analysis_outputs=MTFLOW_outputs,
                                         Lref=self.Lref,
                                         out=out,
                                         cfg=config)

        # Cleanup the generated files
        self.CleanUpFiles()
    

if __name__ == "__main__":
    test = OptimizationProblem(1,
                               )
    
    out = {}
    output = test._evaluate(0, out)

    print(out)
