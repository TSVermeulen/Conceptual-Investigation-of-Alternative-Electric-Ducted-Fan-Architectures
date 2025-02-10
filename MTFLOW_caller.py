"""
MTFLOW_caller
=================

Description
-----------
This module provides a complete interface with the MTFLOW programs, from input 
file generation through to output generation, including convergence handling, 
choking handling, crash detection, and grid refinement. It builds on the 
MTSET_call, MTFLO_call, and MTSOL_call classes to construct the complete interface.
The module is able to maintain a logging file to allow for debugging through the 
use of an optional control boolean. 

Classes
-------
ExitFlag
    A class with exit flags of the MTSOL interface to help with debugging. Effectively a copy of the MTSOL_call.ExitFlag() class
MTFLOW_caller
    A class to execute the complete MTFLOW interface. Effectively a detailed "wrapper" class of the MTSET_call, MTFLO_call, 
    and MTSOL_call classes. 

Examples
--------
>>> import time

>>> oper = {"Inlet_Mach": 0.6,
>>>         "Inlet_Reynolds": 5e6,
>>>         "N_crit": 9,
>>>         }

>>> analysisName = "test_case"
    
>>> # Roughly basing the blade design on the CFM Leap engine (approximate chord lengths)
>>> blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 0.75, "blade_count": 15, "radial_stations": [0.1, 1.15], "chord_length": [0.3, 0.2], "sweep_angle":[np.pi/16, np.pi/16], "twist_angle": [0, np.pi/8]},
>>>                       {"root_LE_coordinate": 1.1, "rotational_rate": 0., "blade_count": 15, "radial_stations": [0.1, 1.3], "chord_length": [0.15, 0.1], "sweep_angle":[np.pi/16, np.pi/16], "twist_angle": [0, np.pi/8]}]
    
>>> # Model the fan and stator blades using a uniform naca2415 profile along the blade span
>>> n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815}
>>> design_parameters = [[n2415_coeff, n2415_coeff],
>>>                      [n2415_coeff, n2415_coeff]]
    
>>> # Model the duct using a naca 6409 profile
>>> duct_parameters = {"b_0": 0.07979831, "b_2": 0.20013347, "b_8": 0.02901246, "b_15": 0.74993802, "b_17": 0.78496242, 'x_t': 0.30429947838135246, 'y_t': 0.0452171520304373, 'x_c': 0.4249653844429819, 'y_c': 0.06028051002570214, 'z_TE': -0.0003886462495685791, 'dz_TE': 0.0004425237127035188, 'r_LE': -0.009225474218611841, 'trailing_wedge_angle': 0.10293203348896998, 'trailing_camberline_angle': 0.21034003141636426, 'leading_edge_direction': 0.26559481057525414, "Chord Length": 3.0, "Leading Edge Coordinates": (0, 1.2)}
    
>>> # Model the centrebody using a naca 0025 profile
>>> centrebody_parameters = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0, 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0, 'y_c': 0, 'z_TE': 0, 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': np.float64(0.27689037361278407), 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 4, "Leading Edge Coordinates": (0.3, 0)}

>>> start_time = time.time()
>>> class_call = MTFLOW_caller(operating_conditions=oper,
>>>                         #    centrebody_params=centrebody_parameters,
>>>                            duct_params=duct_parameters,
>>>                            blading_parameters=blading_parameters,
>>>                            design_parameters=design_parameters,
>>>                            analysis_name=analysisName,
>>>                            ).caller()
>>> end_time = time.time()
    
>>> print(f"Execution of MTFLOW_call.caller() took {end_time - start_time} second")

Notes
-----
N/A

References
----------
The required input data, limitations, and structures are documented within the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID 4995309
Version: 1.0

Changelog:
- V1.0: Initial version. Lacks proper crash handling and choking handling in the HandleExitFlag() method, but otherwise complete. 
"""

import sys
import os
import logging
import random
import numpy as np
from enum import Enum

# Enable submodel relative imports 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Submodels.MTSET_call import MTSET_call
from Submodels.MTFLO_call import MTFLO_call
from Submodels.MTSOL_call import MTSOL_call
from Submodels.file_handling import fileHandling


class ExitFlag(Enum):
    """
    Enum class to define the exit flags for the MTFLOW interface. 

    The exit flags are used to determine the status of the interface execution. 

    Attributes
    ----------
    SUCCESS : int
        Successful completion of the interface execution. 
    CRASH : int
        MTSOL crash - likely related to the grid resolution. 
    NOT_PERFORMED : int
        Not performed, with no iterations executed or outputs generated. 
    COMPLETED: int
        Finished iteration/action, but no convergence. 
    CHOKING: int
        Choking occurs somewhere in solution, needs handling. 
    """

    SUCCESS = -1
    CRASH = 0
    NON_CONVERGENCE = 1
    NOT_PERFORMED = 2
    COMPLETED = 3
    CHOKING = 4


class MTFLOW_caller:
    """
    Wrapper class to execute the complete MTFLOW evaluation cycle.
    This class handles the setup, execution, and error handling for the MTFLOW evaluation process, which includes generating input files, constructing grids, and running solvers.
    """


    def __init__(self,
                 operating_conditions: dict,
                 centrebody_params: dict,
                 duct_params: dict,
                 blading_parameters: list[dict],
                 design_parameters: list[dict],
                 analysis_name: str
                 ) -> None:
        """
        Initialize the MTSOL_call class.

        This method sets up the initial state of the class.

        Parameters
        ----------
        - operating_conditions : dict
            A dictionary containing at least the following entries: Inlet_Mach, Inlet_Reynolds, N_crit, i.e. the inlet Mach number, Reynolds number, and critical amplification factor N
        - centrebody_params : dict
                Dictionary containing parameters for the centerbody.
        - duct_params : dict
            Dictionary containing parameters for the duct.
        - blading_parameters : list[dict]
            List containing the blading parameters for each stage. Each dictionary should include the following keys:
                - "root_LE_coordinate": The leading edge coordinate at the root of the blade.
                - "rotational_rate": The rotational rate of the blade.
                - "blade_count": The number of blades.
                - "radial_stations": List of the radial stations along the blade span.
                - "chord_length": List of the chord length distribution along the blade span.
                - "sweep_angle": List of the sweep angle distribution along the blade span.
                - "twist_angle": List of the twist angle distribution along the blade span.
        - design_parameters : list[list[dict]]
            List containing an equal number of nested lists as there are stages. Each nested list contains an equal number of dictionaries as there are radial stations. 
            Each dictionary must contain the following keys:
                - "b_0", "b_2", "b_8", "b_15", "b_17": Coefficients for the airfoil parameterization.
                - "x_t", "y_t", "x_c", "y_c": Coordinates for the airfoil parameterization.
                - "z_TE", "dz_TE": Trailing edge parameters.
                - "r_LE": Leading edge radius.
                - "trailing_wedge_angle": Trailing wedge angle.
                - "trailing_camberline_angle": Trailing camberline angle.
                - "leading_edge_direction": Leading edge direction.
                - "Chord Length": The chord length of the blade.
        - analysis_name : str
            String of the casename

        Returns
        -------
        None
        """

        # Unpack class inputs
        self.operating_conditions = operating_conditions
        self.analysis_name = analysis_name
        self.blading_parameters = blading_parameters
        self.design_parameters = design_parameters
        self.centrebody_params = centrebody_params
        self.duct_params = duct_params


    def HandleExitFlag(self,
                       exit_flag: int,
                       ) -> None:
        """
        Handle the exit flag of the MTSOL caller.

        Parameters
        ----------
        - exit_flag : int
            Exit flag indicating the status of the solver execution.
        
        Returns
        -------
        None
        """

        # If exit flag of the iteration indicates successful completion of the solver, simply return 
        if exit_flag in (ExitFlag.SUCCESS.value, ExitFlag.NON_CONVERGENCE.value):
            return
        
        # If the exit flag indicates an MTSOL crash, fix the issue causing the crash 
        elif exit_flag == ExitFlag.CRASH.value:
            # TODO: handling of crash
            print("Solver crashed, but no crash handling has been implemented!")
            return
        
        # If the exit flag indicates choking, reduce the rotor RPM to fix the issue
        elif exit_flag == ExitFlag.CHOKING.value:
            print("Choking occurs. Using a rudimentary fix....")
            
            #Use a 5% reduction in RPM as guess
            for i in range(len(self.blading_parameters)):
                reduction_factor = 0.05
                self.blading_parameters[i]["rotational_rate"] = self.blading_parameters[i]["rotational_rate"] * (1 - reduction_factor)
            
            return
        
        # Handle invalid exit flag returns
        elif exit_flag in (ExitFlag.COMPLETED.value, ExitFlag.NOT_PERFORMED.value):
            raise ValueError(f"Invalid exit flag {exit_flag} encountered following execution of MTSOL_call!") from None
        
        # Handle unknown exit flag returns
        else:
            raise ValueError(f"Unknown exit flag {exit_flag} encountered!") from None 


    def caller(self,
               *,
               debug: bool = False) -> tuple[int, list[tuple[int, int]]]:
        """ 
        Executes a complete MTSET-MTFLO-MTSOL evaluation, while handling grid issues and choking issues. 

        Parameters
        ----------
        - debug : bool, optional
            A boolean controlling the logging behaviour of the method. If True, the method generates a caller.log file. 
            Default value is False. 

        Returns
        -------
        Tuple[int, List[Tuple[int, int]]]
            A tuple containing the exit flag and a list of tuples with exit flags and iteration counts
            for the inviscid and viscous solves.
        """

        try:
            # --------------------
            # Set up logging for the caller execution
            # Writes log of the execution to the caller.log file using the 'w' mode. 
            # This empties out the caller file at the start of each MTFLOW_caller.caller() evaluation.
            # --------------------
            if debug:
                logging.basicConfig(level=logging.DEBUG,
                                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                    handlers=[logging.FileHandler("caller.log", 
                                                                mode='w'),
                                            logging.StreamHandler(),
                                            ],
                                    )
                logger = logging.getLogger(__name__)
            
            # --------------------
            # Change working directory to the submodels folder
            # --------------------

            current_dir = os.getcwd()
            subfolder_path = os.path.join(current_dir, 'Submodels')
            os.chdir(subfolder_path)
            
            # --------------------
            # First step is generating the MTSET input file - walls.analysis_name
            # As the MTFLO input file also contains a dependency on operating condition through Omega, it must be generated *within* the MTSOL loop. 
            # The walls.analysis_name file, which is the input to MTSET, does not contain any dependencies, and therefore can be generated outside of this loop here.
            # --------------------

            file_handler = fileHandling()
            file_handler.fileHandlingMTSET(params_CB=self.centrebody_params,
                                           params_duct=self.duct_params,
                                           case_name=self.analysis_name).GenerateMTSETInput()

            
            # --------------------
            # Construct the initial grid in MTSET
            # --------------------
            
            if debug:
                logger.info("Constructing the initial grid in MTSET")
            
            MTSET_call(analysis_name=self.analysis_name,
                       ).caller()
            
            # --------------------
            # Check the grid by running a simple, fan-less, inviscid low-Mach case. If there is an issue with the grid MTSOL will crash
            # Hence we can check the grid by checking the exit flag
            # --------------------
            
            if debug:
                logger.info("Checking the grid")
            
            first_check = True
            exit_flag_gridtest = ExitFlag.NOT_PERFORMED.value
            
            while exit_flag_gridtest != ExitFlag.SUCCESS.value:
                _, [(exit_flag_gridtest, _), _] = MTSOL_call(operating_conditions={"Inlet_Mach": 0.15, "Inlet_Reynolds": 0., "N_crit": self.operating_conditions["N_crit"]},
                                                             analysis_name=self.analysis_name,
                                                             ).caller(run_viscous=False,
                                                                      generate_output=False,
                                                                      )
                
                if exit_flag_gridtest == ExitFlag.SUCCESS.value:  # If the grid status is okay, break out of the checking loop and continue
                    if debug:
                        logger.info("Grid passed checks")
                    break

                # If the grid is incorrect, change grid parameters and rerun MTSET to update the grid. 
                if debug:
                    logger.warning("Grid crashed. Trying alternate grid parameters")

                # If first_check is true, we can try the suggested coefficients
                # The updated e and x coefficients reduce the number of streamwise points on the airfoil elements (by 0.1 * Npoints), 
                # while yielding a more "rounded/elliptic" grid due to the reduced x-coefficient.
                if first_check:
                    grid_e_coeff = 0.7
                    grid_x_coeff = 0.5
                    if debug:
                        logger.info(f"Trying suggested coefficients e={grid_e_coeff} and x={grid_x_coeff}")

                # If first_check is false, the suggested coefficients do not work, so we try a random number approach to brute-force a grid
                else:
                    grid_e_coeff = random.uniform(0.6, 1.0)
                    grid_x_coeff = random.uniform(0.2, 0.95)
                    if debug:
                        logger.info(f"Suggested Coefficients failed to yield a satisfactory grid. Trying bruteforce method with e={grid_e_coeff} and x={grid_x_coeff}")
                    
                MTSET_call(analysis_name=self.analysis_name,
                           grid_e_coeff=grid_e_coeff,
                           grid_x_coeff=grid_x_coeff,
                           ).caller()
                
                first_check = False

            # --------------------
            # Execute MTSOl solver
            # Passes the exit flag to determine if any issues have occurred. 
            # --------------------

            if debug:
                logger.info("Starting MTSOL execution loop")

            exit_flag = ExitFlag.NOT_PERFORMED.value  # Initialize exit flag
            while exit_flag != ExitFlag.SUCCESS.value:
                if debug:
                    logger.info("Loading blade row(s) from MTFLO")
                file_handler.fileHandlingMTFLO(self.analysis_name).GenerateMTFLOInput(blading_params=self.blading_parameters,
                                                                                      design_params=self.design_parameters)  # Create the MTFLO input file
                MTFLO_call(self.analysis_name).caller() #Load in the blade row(s) from MTFLO

                if debug:
                    logger.info("Executing MTSOL")
                
                exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)] = MTSOL_call(operating_conditions=self.operating_conditions,
                                                                                                                   analysis_name=self.analysis_name,
                                                                                                                   ).caller(run_viscous=True,
                                                                                                                            generate_output=True,
                                                                                                                            )
               
                if debug:
                    logger.info(f"MTSOL finished with exit flag {exit_flag}")
                    logger.info("Processing exit flag....")

                self.HandleExitFlag(exit_flag=exit_flag)  # Check completion status of MTSOL

            if debug:
                logger.info(f"MTSOL execution loop finished with final exit flag {exit_flag}")

            return exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)]

        # --------------------
        # If an error occured, handle it with the logger
        # --------------------
        except Exception as e:
            if debug:
                logger.critical(f"An error occurred: {e}")

            return ExitFlag.CRASH.value


if __name__ == "__main__":
    import time

    # Define some test inputs
    oper = {"Inlet_Mach": 0.6,
            "Inlet_Reynolds": 5e6,
            "N_crit": 9,
            }

    analysisName = "test_case"
    
    # Roughly basing the blade design on the CFM Leap engine (approximate chord lengths)
    blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 0.75, "blade_count": 15, "radial_stations": [0.1, 1.15], "chord_length": [0.3, 0.2], "sweep_angle":[np.pi/16, np.pi/16], "twist_angle": [0, np.pi/8]},
                          {"root_LE_coordinate": 1.1, "rotational_rate": 0., "blade_count": 15, "radial_stations": [0.1, 1.15], "chord_length": [0.15, 0.1], "sweep_angle":[np.pi/16, np.pi/16], "twist_angle": [0, np.pi/8]}]
    
    # Model the fan and stator blades using a uniform naca2415 profile along the blade span
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815}
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    # Model the duct using a naca 6409 profile
    duct_parameters = {"b_0": 0.07979831, "b_2": 0.20013347, "b_8": 0.02901246, "b_15": 0.74993802, "b_17": 0.78496242, 'x_t': 0.30429947838135246, 'y_t': 0.0452171520304373, 'x_c': 0.4249653844429819, 'y_c': 0.06028051002570214, 'z_TE': -0.0003886462495685791, 'dz_TE': 0.0004425237127035188, 'r_LE': -0.009225474218611841, 'trailing_wedge_angle': 0.10293203348896998, 'trailing_camberline_angle': 0.21034003141636426, 'leading_edge_direction': 0.26559481057525414, "Chord Length": 3.0, "Leading Edge Coordinates": (0, 1.2)}
    
    # Model the centrebody using a naca 0025 profile
    centrebody_parameters = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0, 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0, 'y_c': 0, 'z_TE': 0, 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': np.float64(0.27689037361278407), 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 4, "Leading Edge Coordinates": (0.3, 0)}

    start_time = time.time()
    class_call = MTFLOW_caller(operating_conditions=oper,
                               centrebody_params=centrebody_parameters,
                               duct_params=duct_parameters,
                               blading_parameters=blading_parameters,
                               design_parameters=design_parameters,
                               analysis_name=analysisName,
                               ).caller(debug=True)
    end_time = time.time()

    print(f"Execution of MTFLOW_call.caller() took {end_time - start_time} second")
