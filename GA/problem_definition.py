"""
problem_definition
==================


"""

import os
import sys
import hashlib
import numpy as np
import shutil
from pathlib import Path
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
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
import config


class OptimizationProblem(ElementwiseProblem):
    """
    Class definition of the optimization problem to be solved using the genetic algorithm. 
    Inherits from the ElementwiseProblem class from pymoo.core.problem.
    """

    def __init__(self,
                 obj_count : int,
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
        # This is required to handle the mixed-variable nature of the optimisation, where the blade count is an integer
        vars = []
        if config.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, initialise the variable types
            vars.append(Real(bounds=(0, 1)))  # mapping variable for b_8
            vars.append(Real(bounds=(0, 1)))  # b_15
            vars.append(Real(bounds=(0, 1)))  # x_t
            vars.append(Real(bounds=(0, 0.25)))  # y_t
            vars.append(Real(bounds=(0, 0.05)))  # dz_TE
            vars.append(Real(bounds=(-0.1, 0)))  # r_LE
            vars.append(Real(bounds=(0, 0.5)))  # trailing_wedge_angle
            vars.append(Real(bounds=(0.25, 4)))  # Chord Length
        if config.OPTIMIZE_DUCT:
            # If the duct is to be optimised, intialise the variable types
            vars.append(Real(bounds=(0, 1)))  # b_0
            vars.append(Real(bounds=(0, 0.5)))  # b_2
            vars.append(Real(bounds=(0, 1)))  # mapping variable for b_8
            vars.append(Real(bounds=(0, 1)))  # b_15
            vars.append(Real(bounds=(0, 1)))  # b_17
            vars.append(Real(bounds=(0, 1)))  # x_t
            vars.append(Real(bounds=(0, 0.25)))  # y_t
            vars.append(Real(bounds=(0, 1)))  # x_c
            vars.append(Real(bounds=(0, 0.1)))  # y_c
            vars.append(Real(bounds=(0, 0.2)))  # z_TE
            vars.append(Real(bounds=(0, 0.05)))  # dz_TE
            vars.append(Real(bounds=(-0.1, 0)))  # r_LE
            vars.append(Real(bounds=(0, 0.5)))  # trailing_wedge_angle
            vars.append(Real(bounds=(0, 0.5)))  # trailing_camberline_angle
            vars.append(Real(bounds=(0, 0.5)))  # leading_edge_direction
            vars.append(Real(bounds=(0.25, 2.5)))  # Chord Length
            vars.append(Real(bounds=(-0.5, 0.5)))  # Leading Edge X-Coordinate

        for i in range(self.num_stages):
            # If (any of) the rotor/stator stage(s) are to be optimised, initialise the variable types
            if self.optimize_stages[i]:
                for _ in range(self.num_radial):
                    vars.append(Real(bounds=(0, 1)))  # b_0
                    vars.append(Real(bounds=(0, 0.5)))  # b_2
                    vars.append(Real(bounds=(0, 1)))  # mapping variable for b_8
                    vars.append(Real(bounds=(0, 1)))  # b_15
                    vars.append(Real(bounds=(0, 1)))  # b_17
                    vars.append(Real(bounds=(0, 1)))  # x_t
                    vars.append(Real(bounds=(0, 0.25)))  # y_t
                    vars.append(Real(bounds=(0, 1)))  # x_c
                    vars.append(Real(bounds=(0, 0.1)))  # y_c
                    vars.append(Real(bounds=(0, 0.2)))  # z_TE
                    vars.append(Real(bounds=(0, 0.05)))  # dz_TE
                    vars.append(Real(bounds=(-0.1, 0)))  # r_LE
                    vars.append(Real(bounds=(0, 0.5)))  # trailing_wedge_angle
                    vars.append(Real(bounds=(0, 0.5)))  # trailing_camberline_angle
                    vars.append(Real(bounds=(0, 0.5)))  # leading_edge_direction

        for i in range(self.num_stages):
            if self.optimize_stages[i]:
                vars.append(Real(bounds=(0.1)))  # root_LE_coordinate
                vars.append(Integer(bounds=(3, 20)))  # blade_count
                vars.append(Real(bounds=(-np.pi/4, np.pi/4)))  # ref_blade_angle
                vars.append(Real(bounds=(0, 1.5)))  # blade radius

                for _ in range(self.num_radial): 
                    vars.append(Real(bounds=(0.05, 0.5)))  # chord length
                    vars.append(Real(bounds=(0, np.pi/3)))  # sweep_angle
                    vars.append(Real(bounds=(-np.pi/4, np.pi/4)))  # blade_angle

        # For a mixed-variable problem, PyMoo expects the vars to be a dictionary, so we convert vars to a dictionary.
        # Note that all variables are given a name xi.
        # TODO: update this algoritm to automatically give the appropriate name to each variable. 
        vars = {f"x{i}": var for i, var in enumerate(vars)}

        # Initialize the parent class
        super().__init__(vars=vars,
                       n_obj=obj_count,
                       **kwargs)
        
        # Change working directory to the parent folder
        try:
            os.chdir(parent_dir)
        except OSError as e:
            raise OSError from e
        

    def GenerateAnalysisName(self,
                             pop_idx: int, 
                             gen_idx: int) -> str:
        """
        Generate a unique analysis name with a maximum length of 30 characters.
        This is required to enable multi-threading of the optimization problem, since each evaluation of MTFLOW requires a unique set of files. 

        Parameters
        ----------
        - pop_idx : int
            Population index of the current evaluation
        - gen_idx : int
            Generation index of the current evaluation

        Returns
        -------
        - analysis_name : str
            A unique hashed analysis name for the population and generation indices provided. 
        """

        # Construct the base name based on the population and generation indices
        base_name = f"p{pop_idx}g{gen_idx}_"

        # Construct a hash suffix
        hash_suffix = hashlib.md5(base_name.encode()).hexdigest()

        # Construct the full analysis name and trim it to be no longer than 32 characters
        analysis_name = base_name + "_" + hash_suffix
        analysis_name = analysis_name[:30]

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
                                         "b_8": x[0],
                                         "b_15": x[1],
                                         "b_17": 0,
                                         "x_t": x[2],
                                         "y_t": x[3],
                                         "x_c": 0,
                                         "y_c": 0,
                                         "z_TE": 0,
                                         "dz_TE": x[4],
                                         "r_LE": x[5],
                                         "trailing_wedge_angle": x[6],
                                         "trailing_camberline_angle": 0,
                                         "leading_edge_direction": 0, 
                                         "Chord Length": x[7],
                                         "Leading Edge Coordinates": (0, 0)}
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
                    section_parameters = {"b_0": x[idx],
                                        "b_2": x[idx + 1], 
                                        "b_8": x[idx + 2],
                                        "b_15": x[idx + 3],
                                        "b_17": x[idx + 4],
                                        "x_t": x[idx + 5],
                                        "y_t": x[idx + 6],
                                        "x_c": x[idx + 7],
                                        "y_c": x[idx + 8],
                                        "z_TE": x[idx + 9],
                                        "dz_TE": x[idx + 10],
                                        "r_LE": x[idx + 11],
                                        "trailing_wedge_angle": x[idx + 12],
                                        "trailing_camberline_angle": x[idx + 13],
                                        "leading_edge_direction": x[idx + 14]}
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
                stage_blading_parameters["root_LE_coordinate"] = x[idx]
                stage_blading_parameters["blade_count"] = x[idx + 1]
                stage_blading_parameters["ref_blade_angle"] = x[idx + 2]
                stage_blading_parameters["reference_section_blade_angle"] = config.REFERENCE_SECTION_ANGLES[i]
                stage_blading_parameters["radial_stations"] = radial_linspace * x[idx + 3]  # Radial stations are defined as fraction of blade radius * local radius
                self.blade_diameters.append(x[idx + 3] * 2)

                # Initialize sectional blading parameter lists
                stage_blading_parameters["chord_length"] = [None] * self.num_radial
                stage_blading_parameters["sweep_angle"] = [None] * self.num_radial
                stage_blading_parameters["blade_angle"] = [None] * self.num_radial

                base_idx = idx + 4
                for j in range(self.num_radial):
                    # Loop over the number of radial sections and write their data to the corresponding lists
                    stage_blading_parameters["chord_length"][j]= x[base_idx + j]
                    stage_blading_parameters["sweep_angle"][j] = x[base_idx + self.num_radial + j]
                    stage_blading_parameters["blade_angle"][j] = x[base_idx + 2 * self.num_radial + j]
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
                LE_coords = (x[idx + 16], 0)
            else:
                LE_coords = (x[idx+16], 0)
            self.duct_variables = {"b_0": x[idx],
                                   "b_2": x[idx + 1], 
                                   "b_8": x[idx + 2],
                                   "b_15": x[idx + 3],
                                   "b_17": x[idx + 4],
                                   "x_t": x[idx + 5],
                                   "y_t": x[idx + 6],
                                   "x_c": x[idx + 7],
                                   "y_c": x[idx + 8],
                                   "z_TE": x[idx + 9],
                                   "dz_TE": x[idx + 10],
                                   "r_LE": x[idx + 11],
                                   "trailing_wedge_angle": x[idx + 12],
                                   "trailing_camberline_angle": x[idx + 13],
                                   "leading_edge_direction": x[idx + 14], 
                                   "Chord Length": x[idx + 15],
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

        # Compute the inlet speed
        V_inl = config.oper["Inlet_Mach"] * config.atmosphere.speed_of_sound[0]

        # Compute the inlet Reynolds number and write it to config.oper
        config.oper["Inlet_Reynolds"] = round(float((V_inl * self.Lref) / config.atmosphere.kinematic_viscosity[0]), 3)


    def ComputeOmega(self) -> None:
        """
        A simple function to compute the non-dimensional MTFLOW rotational rate Omega,
        and write it to the oper dictionary in config.py.

        Returns
        -------
        None
        """

        # Compute the inlet speed
        V_inl = config.oper["Inlet_Mach"] * config.atmosphere.speed_of_sound[0]

        # Compute the non-dimensional rotational rate Omega for MTFLOW and write it to config.oper
        config.oper["Omega"] = round(float((-config.oper["RPS"] * np.pi * 2 * self.Lref) / (V_inl)),3)


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

        # Initialize empty data arrays
        x_tip = np.zeros_like(self.blade_blading_parameters)
        radial_duct_coordinates = np.zeros_like(self.blade_blading_parameters)

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

        # Construct interpolant of the duct surface
        duct_interpolant = interpolate.make_splrep(lower_x,
                                             lower_y,
                                             k=3)

        # Loop over all stages
        duct_y = np.zeros_like(self.blade_blading_parameters)
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


    def _evaluate(self, x, out, *args, **kwargs):
        # Generate a unique analysis name
        pop_idx = kwargs.get("pop_idx", 0)
        gen_idx = kwargs.get("gen_idx", 0)
        self.analysis_name = self.GenerateAnalysisName(pop_idx,
                                                  gen_idx)
        
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
        # Note that the length of computed_objectives must equal obj_count
        computed_objectives = Objectives().ComputeObjective(outputs=MTFLOW_outputs, 
                                                            objective_IDs=config.objective_IDs)

        # Compute constraints


        # Cleanup the generated files
        self.CleanUpFiles()

        # Construct outputs
        out["F"] = np.column_stack(computed_objectives)

        return out
    


if __name__ == "__main__":
    test = OptimizationProblem(1,
                               )
    
    test._evaluate(0, {})
