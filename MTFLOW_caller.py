"""


"""

from Submodels import MTSET_call, MTFLO_call, MTSOL_call, file_handling
from Submodels import file_handling
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
    """

    SUCCESS = -1
    CRASH = 0
    NOT_PERFORMED = 1
    COMPLETED = 2


class MTFLOW_caller:
    def __init__(self, *args) -> None:
        """
        Initialize the MTSOL_call class.

        This method sets up the initial state of the class.

        Returns
        -------
        None
        """

        # Unpack class inputs
        operating_conditions, file_path, analysis_name = args
        self.operating_conditions: dict = operating_conditions
        self.fpath: str = file_path
        self.analysis_name: str = analysis_name


    def CheckGrid(self,
                  exit_flag: int,
                  ) -> int:
        """
        
        """

        if exit_flag == ExitFlag.SUCCESS.value:
            return exit_flag
        else:
            return ExitFlag.CRASH.value

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
        # First step is generating the MTSET and MTFLO input files - walls.analysis_name and tflow.analysis_name
        # --------------------
        file_handler = file_handling.fileHandling()  # Create an instance of the file handler class

        file_handler.fileHandlingMTSET(params_CB, 
                                       params_duct, 
                                       ducted_fan_design_params, 
                                       self.analysis_name,
                                       )  # Create walls.analysis_name input file
        
        file_handler.fileHandlingMTFLO(stage_count, 
                                       self.analysis_name,
                                       ).GenerateMTFLOInput(blading_params,
                                                            design_params,
                                                            )  # Create tflow.analysis_name input file
        
        # --------------------
        # With the input file generated, construct the initial grid in MTSET
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
            grid_status = self.CheckGrid(exit_flag_gridtest)

            if grid_status == ExitFlag.SUCCESS.value:  # If the grid status is okay, break out of the checking loop and continue
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
        # --------------------
        exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)] = MTSOL_call.MTSOL_call(operating_conditions,
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