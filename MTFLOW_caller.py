"""


"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Submodels import MTSET_call, MTFLO_call, MTSOL_call
from enum import Enum


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
    NOT_PERFORMED = 1
    COMPLETED = 2
    CHOKING = 3


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
                       iter_count: int,
                       case_type: str,
                       ) -> None:
        """
        Handle the exit flag of (any of) the MTLOW executables.

        Parameters
        ----------
        exit_flag : int
            Exit flag indicating the status of the solver execution.
        iter_count : int
            Number of iterations performed up until failure of the solver.
        case_type : str
            Type of case that was run.
        
        Returns
        -------
        None
        """

        if exit_flag == ExitFlag.NON_CONVERGENCE.value:
            self.HandleNonConvergence()
        elif exit_flag == ExitFlag.CRASH.value:
            self.CrashRecovery(case_type,
                               iter_count,
                               )
        elif (exit_flag == ExitFlag.COMPLETED.value or 
              exit_flag == ExitFlag.SUCCESS.value or 
              exit_flag == ExitFlag.NOT_PERFORMED.value):
            return
        else:
            raise OSError(f"Unknown exit flag {exit_flag} encountered!") from None


    def caller(self) -> int:
        """ 
        
        """

        # --------------------
        # First step is generating the MTFLO input files - tflow.analysis_name
        # As the MTFLO input file also contains a dependency on operating condition through Omega, it must be generated *within* the MTFLOW caller. 
        # The walls.analysis_name file, which is the input to MTSET, does not contain any dependencies, and therefore can be generated outside of this caller function.
        # --------------------
        file_handler = file_handling.fileHandling().fileHandlingMTFLO(self.analysis_name)  # Create an instance of the file handler MTFLO subclass
        
        file_handler.GenerateMTFLOInput(self.blading_parameters,
                                        self.design_parameters,
                                        )  # Create tflow.analysis_name input file
        
        # --------------------
        # Construct the initial grid in MTSET
        # --------------------
        MTSET_call.MTSET_call(self.analysis_name,
                              ).caller()
        
        # --------------------
        # Check the grid by running a simple, fan-less, inviscid low-Mach case. If there is an issue with the grid MTSOL will crash
        # Hence we can check the grid by checking the exit flag
        # --------------------
        checking_grid = True
        
        while checking_grid:
            exit_flag_gridtest, [(exit_flag_gridtest, iter_count_gridtest), _] = MTSOL_call.MTSOL_call({"Inlet_Mach": 0.2, "Inlet_Reynolds": 0.},
                                                                                                       self.analysis_name,
                                                                                                        ).caller(run_viscous=False,
                                                                                                                 generate_output=False,
                                                                                                                 )

            if exit_flag_gridtest == ExitFlag.SUCCESS.value:  # If the grid status is okay, break out of the checking loop and continue
                break

            # If the grid is incorrect, change grid parameters and rerun MTSET to update the input file. 
            MTSET_call.MTSET_call(self.analysis_name,
                                  ).caller()

        # --------------------
        # Load in the blade row(s) from MTFLO
        # --------------------
        MTFLO_call.MTFLO_call(self.analysis_name,
                              ).caller()

        # --------------------
        # Execute MTSOl solver
        # Passes the exit flags and iteration counts to the handle exit flag function to determine if any issues have occurred. 
        # --------------------
        exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)] = MTSOL_call.MTSOL_call(self.operating_conditions,
                                                                                                                      self.analysis_name,
                                                                                                                      ).caller(run_viscous=True,
                                                                                                                               generate_output=True,
                                                                                                                               )


        # --------------------
        # Check completion status of MTSOL
        # --------------------
        self.HandleExitFlag()


        return




if __name__ == "__main__":
    import time
    from Submodels import file_handling
    import numpy as np


    # Define some test inputs
    oper = {"Inlet_Mach": 0.25,
            "Inlet_Reynolds": 5e6}

    analysisName = "test_case"

    blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 1., "blade_count": 18, "radial_stations": [0.1, 1.8], "chord_length": [0.2, 0.3], "sweep_angle":[np.pi/16, np.pi/16], "twist_angle": [0, np.pi / 3]},
                          {"root_LE_coordinate": 1., "rotational_rate": 0., "blade_count": 10, "radial_stations": [0.1, 1.8], "chord_length": [0.2, 0.3], "sweep_angle":[np.pi/8, np.pi/8], "twist_angle": [0, np.pi/8]}]
    
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 1.5, "Leading Edge Coordinates": (0, 2)}
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    inputs = [oper, analysisName, ]
    
    class_call = MTFLOW_caller(operating_conditions=oper,
                               blading_parameters=blading_parameters,
                               design_parameters=design_parameters,
                               analysis_name=analysisName,
                               ).caller()

