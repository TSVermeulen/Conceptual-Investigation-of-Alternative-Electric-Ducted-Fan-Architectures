"""


"""

import sys
import os
import logging
import random
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
    def __init__(self,
                 operating_conditions: dict,
                 blading_parameters: list[dict],
                 design_parameters: list[list[dict]],
                 analysis_name: str
                 ) -> None:
        """
        Initialize the MTSOL_call class.

        This method sets up the initial state of the class.

        Parameters
        ----------
        - operating_conditions : dict
            A dictionary containing at least the following entries: Inlet_Mach, Inlet_Reynolds, N_crit, i.e. the inlet Mach number, Reynolds number, and critical amplification factor N
        - blading_parameters : np.ndarray[dict]
            Array containing the blading parameters for each stage. Each dictionary should include the following keys:
                - "root_LE_coordinate": The leading edge coordinate at the root of the blade.
                - "rotational_rate": The rotational rate of the blade.
                - "blade_count": The number of blades.
                - "radial_stations": Numpy array of the radial stations along the blade span.
                - "chord_length": Numpy array of the chord length distribution along the blade span.
                - "sweep_angle": Numpy array of the sweep angle distribution along the blade span.
                - "twist_angle": Numpy array of the twist angle distribution along the blade span.
        - design_parameters : np.ndarray[dict]
            Array containing an equal number of dictionary entries as there are stages. Each dictionary must contain the following keys:
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


    def HandleExitFlag(self,
                       exit_flag: int,
                       iter_output: list[tuple],
                       ) -> None:
        """
        Handle the exit flag of (any of) the MTLOW executables.

        Parameters
        ----------
        - exit_flag : int
            Exit flag indicating the status of the solver execution.
        - iter_count : int
            Number of iterations performed up until failure of the solver.
        - case_type : str
            Type of case that was run.
        
        Returns
        -------
        None
        """

        # If exit flag of the iteration indicates successful completion of the solver, simply return 
        if exit_flag in (ExitFlag.SUCCESS.value, ExitFlag.NON_CONVERGENCE.value):
            return
        elif exit_flag == ExitFlag.CRASH.value:
            # TODO: handling of crash
            print("Solver crashed, but no crash handling has been implemented!")
            return
        elif exit_flag == ExitFlag.CHOKING.value:
            # TODO: handling of choking
            print("Choking occurs, but no choking handling has been implemented!")
            return
        elif exit_flag in (ExitFlag.COMPLETED.value, ExitFlag.NOT_PERFORMED.value):
            raise ValueError(f"Invalid exit flag {exit_flag} encountered following execution of MTSOL_call!") from None
        else:
            raise ValueError(f"Unknown exit flag {exit_flag} encountered!") from None
            
        


    def caller(self) -> int:
        """ 
        
        """
        try:
            # --------------------
            # Set up logging for the caller execution
            # Writes log of the execution to the caller.log file using the 'w' mode. 
            # This empties out the caller file at the start of each MTFLOW_caller.caller() evaluation.
            # --------------------

            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                handlers=[logging.FileHandler("caller.log", 
                                                              mode='w'),
                                          logging.StreamHandler(),
                                          ],
                                )
            logger = logging.getLogger(__name__)

            # --------------------
            # First step is generating the MTFLO input file - tflow.analysis_name
            # As the MTFLO input file also contains a dependency on operating condition through Omega, it must be generated *within* the MTFLOW caller. 
            # The walls.analysis_name file, which is the input to MTSET, does not contain any dependencies, and therefore can be generated outside of this caller function.
            # --------------------

            logger.info("Generating the MTFLO input file")
            file_handler = fileHandling().fileHandlingMTFLO(self.analysis_name)  # Create an instance of the file handler MTFLO subclass
            
            file_handler.GenerateMTFLOInput(self.blading_parameters,
                                            self.design_parameters,
                                            )  # Create tflow.analysis_name input file
            
            # --------------------
            # Construct the initial grid in MTSET
            # --------------------

            logger.info("Constructing the initial grid in MTSET")
            grid_e_coeff = 0.8  # Initialize default e coefficient for MTSET grid generation
            grid_x_coeff = 0.8  # Initialize default x coefficient for MTSET grid generation
            MTSET_call(self.analysis_name,
                       grid_e_coeff=grid_e_coeff,
                       grid_x_coeff=grid_x_coeff,
                       ).caller()
            
            # --------------------
            # Check the grid by running a simple, fan-less, inviscid low-Mach case. If there is an issue with the grid MTSOL will crash
            # Hence we can check the grid by checking the exit flag
            # --------------------

            logger.info("Checking the grid")
            checking_grid = True
            check_counter = 0
            
            while checking_grid:
                _, [(exit_flag_gridtest, _), _] = MTSOL_call({"Inlet_Mach": 0.15, "Inlet_Reynolds": 0., "N_crit": self.operating_conditions["N_crit"]},
                                                             self.analysis_name,
                                                             ).caller(run_viscous=False,
                                                                      generate_output=False,
                                                                      )
                
                if exit_flag_gridtest == ExitFlag.SUCCESS.value:  # If the grid status is okay, break out of the checking loop and continue
                    logger.info("Grid passed checks")
                    break

                # If the grid is incorrect, change grid parameters and rerun MTSET to update the grid. 
                logger.warning("Grid crashed. Trying alternate grid parameters")

                # If check counter is non-zero, the suggested coefficients do not work, so we try a random number approach to brute-force a grid
                if check_counter >= 1:
                    grid_e_coeff = random.uniform(0.6, 1.0)
                    grid_x_coeff = random.uniform(0.2, 0.95)
                    logger.info(f"Suggested Coefficients failed to yield a satisfactory grid. Trying bruteforce method with e={grid_e_coeff} and x={grid_x_coeff}")
                # If the check counter is zerom, we can try the suggested coefficients
                # The updated e and x coefficients reduce the number of streamwise points on the airfoil elements (by 0.1 * Npoints), 
                # while yielding a more "rounded/elliptic" grid due to the reduced x-coefficient.
                else:
                    grid_e_coeff = 0.7
                    grid_x_coeff = 0.5
                    logger.info(f"Trying suggested coefficients e={grid_e_coeff} and x={grid_x_coeff}")
                    
                MTSET_call(self.analysis_name,
                           grid_e_coeff=grid_e_coeff,
                           grid_x_coeff=grid_x_coeff,
                           ).caller()
                
                check_counter += 1

            # --------------------
            # Load in the blade row(s) from MTFLO
            # --------------------

            logger.info("Loading blade row(s) from MTFLO")
            MTFLO_call(self.analysis_name,
                       ).caller()

            # --------------------
            # Execute MTSOl solver
            # Passes the exit flags and iteration counts to the handle exit flag function to determine if any issues have occurred. 
            # --------------------

            logger.info("Executing MTSOL")
            exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)] = MTSOL_call(self.operating_conditions,
                                                                                                               self.analysis_name,
                                                                                                               ).caller(run_viscous=True,
                                                                                                                        generate_output=True,
                                                                                                                        )

            # --------------------
            # Check completion status of MTSOL
            # --------------------

            logger.info("Checking MTSOL exit flag")
            self.HandleExitFlag(exit_flag=exit_flag,
                                iter_output=[(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)],
                                )

            return exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)]

        # --------------------
        # If an error occured, handle it with the logger
        # --------------------
        except Exception as e:
            logger.critical(f"An error occurred: {e}")
            return ExitFlag.CRASH.value


if __name__ == "__main__":
    import time
    import numpy as np


    # Define some test inputs
    oper = {"Inlet_Mach": 0.5,
            "Inlet_Reynolds": 5e6,
            "N_crit": 9,
            }

    analysisName = "test_case"

    blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 0.05, "blade_count": 18, "radial_stations": [0.1, 1.8], "chord_length": [0.2, 0.4], "sweep_angle":[np.pi/16, np.pi/16], "twist_angle": [0, np.pi / 3]},
                          {"root_LE_coordinate": 1., "rotational_rate": 0., "blade_count": 10, "radial_stations": [0.1, 1.8], "chord_length": [0.2, 0.3], "sweep_angle":[np.pi/8, np.pi/8], "twist_angle": [0, np.pi/8]}]
    
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 1.5, "Leading Edge Coordinates": (0, 2)}
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    inputs = [oper, analysisName, ]
    
    start_time = time.time()
    class_call = MTFLOW_caller(operating_conditions=oper,
                               blading_parameters=blading_parameters,
                               design_parameters=design_parameters,
                               analysis_name=analysisName,
                               ).caller()
    end_time = time.time()

    print(f"Execution of MTFLOW_caller.caller() took {end_time - start_time} seconds")

