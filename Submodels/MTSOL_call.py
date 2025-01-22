"""

MTSOL calling class definition. 

The class definition contains all the required functions to interface the MTSOL code with Python. 

@author: T.S.Vermeulen
@email: thomas0708.vermeulen@gmail.com / T.S.Vermeulen@student.tudelft.nl
@version: 0

Changelog:
- V0: File created with empty class as placeholder

"""

import subprocess
import os
from typing import Any, Optional
from collections import deque

class MTSOL_call:
    """
    
    """

    def __init__(self,
                 *args: tuple[dict, Any],
                 ) -> None:
        """
        Initialize the MTSOL_call class.

        This method sets up the initial state of the class.

        Returns
        -------
        None
        """

        operating_conditions, file_path, analysis_name = args
        self.operating_conditions = operating_conditions
        self.fpath = file_path
        self.analysis_name = analysis_name

        # Define constants for the class
        self.ITER_STEP_SIZE = 2  # Step size in which iterations are performed in MTSOL


    def GenerateProcess(self,
                        ) -> None:
        """
        Create MTSOL subprocess

        Requires that the executable, mtsol.exe, and the input file, tdat.xxx are present in the same directory as this
        Python file. 
        """

        # Get the directory where the current Python file is located
        current_file_directory = os.path.dirname(os.path.abspath(__file__))

        # Change the working directory to the directory of the current Python file
        os.chdir(current_file_directory)

        # Generate the subprocess and write it to self
        self.process = subprocess.Popen([self.fpath, self.analysis_name], 
                                        stdin=subprocess.PIPE, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        shell=True, 
                                        text=True,
                                        )
        
        # Check if subprocess is started successfully
        if self.process.poll() is not None:
            raise ImportError(f"MTSOL or tdat.{self.analysis_name} not found in {self.fpath}") from None    
               

    def SetOperConditions(self,
                          ) -> None:
        """
        Set the inlet Mach number, Reynolds number, critical amplification factor, and disable the viscous toggles for all elements present. 

        Returns
        -------
        None
        """

        # Enter the Modify solution parameters menu
        self.process.stdin.write("m \n")
        self.process.stdin.flush()

        # Write inlet Mach number
        self.process.stdin.write(f"M {self.operating_conditions["Inlet_Mach"]} \n")
        self.process.stdin.flush()

        # Set critical amplification factor to N=9 rather than the default N=7
        self.process.stdin.write("N 9\n")
        self.process.stdin.flush()

        # Set the Reynolds number, calculated using the length self.LREF = 1!
        # Flush is required here to ensure console output is up-to-date before collecting it. 
        self.process.stdin.write(f"R {self.operating_conditions["Inlet_Reynolds"]} \n")
        self.process.stdin.flush()
       
        # Disable all viscous toggles to ensure inviscid analysis is run initially
        # To do this, we need to check what elements are present This is done by checking the console output of the menu and identifying the indices of all 'Tx' rows
        # Collect console output from MTSOL, stopping when the end of the menu is reached.
        interface_output = []
        while True:
            next_line = self.process.stdout.readline()  # Collect output and add to list
            interface_output.append(next_line)
            
            if next_line == "" and self.process.poll() is not None:  #Handle (unexpected) quitting of program
                break
            if next_line == ' V1,2..   Viscous side toggles\n':  # Stop collecting once end of MTSOL menu is reached
                break
        
        # Count the number of elements present and get the indices of the first and last element.  
        idx_first_element = interface_output.index(' G  2        Grid-move type\n') + 2
        idx_last_element = len(interface_output) - 3
        n_elements = idx_last_element - idx_first_element + 1

        # Disable the viscous toggles for each surface
        # Element surface numbers are stored to the toggles list, which is written to self to enable easy access later on when re-enabling the viscous toggles.
        # Note that odd numbers are the outer surfaces, while even numbers are the inner surfaces.
        toggles = []
        for i in range(n_elements):
            self.process.stdin.write(f"V {interface_output[idx_first_element + i][2]} \n")
            self.process.stdin.flush()	
            toggles.append(int(interface_output[idx_first_element + i][2]))
        self.element_counts = toggles
        
        # Exit the modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()
    

    def ToggleViscous(self,
                      elements: Optional[list[int]|int],
                      ) -> None:
        """
        Toggle the viscous settings for all elements.

        Parameters
        ----------
        elements : list[int] | int, optional
            An integer or list of integers representing the elements for which the viscous settings need to be toggled.
            If None, all elements are toggled. Default is None.

        Returns
        -------
        None
        """

        # Enter the Modify solution parameters menu
        self.process.stdin.write("m \n")
        self.process.stdin.flush()

        # Input Validation, together with setting the viscous settings for each element as desired
        if elements is not None:
            if not all(map(lambda v: v in self.element_counts, elements)):
                raise OSError(f"element is not in the element counted in the solution parameters menu!") from None
            self.process.stdin.write(f"V {','.join(map(str, elements))} \n")
        else:
            self.process.stdin.write(f"V {','.join(map(str, self.element_counts))} \n")
        self.process.stdin.flush()

        # Exit the Modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()

    def GenerateSolverOutput(self,
                             Viscous: bool = False,
                             ) -> None:
        """
        Generate all output files for the current analysis. 
        If a viscous analysis was performed, the boundary layer data is also dumped to the corresponding file.
        Requires that MTSOL is in the main menu when starting this function. 

        Parameters
        ----------
        Viscous : bool, optional
            If True, generates the outputs corresponding to a viscous analysis. Default is False.

        Returns
        -------
        None
        """

        # Update the solution state file
        self.process.stdin.write("W \n")
        self.process.stdin.flush()

        # If a viscous analysis was performed, dump the viscous data
        if Viscous:
            # If a viscous case was performed, dump the boundary layer data
            self.process.stdin.write("B \n")
            self.process.stdin.write(f"boundary_layer.{self.analysis_name} \n")
            self.process.stdin.flush()

            # Dump the flowfield data
            self.process.stdin.write("D \n")
            self.process.stdin.write(f"flowfield_viscous.{self.analysis_name} \n")
            self.process.stdin.flush()

            # Dump the forces data
            self.process.stdin.write("F \n")
            self.process.stdin.write(f"forces_viscous.{self.analysis_name} \n") 
            self.process.stdin.flush()
        else:
            # If an inviscid analysis was performed, dump the inviscid data
            # Dump the flowfield data
            self.process.stdin.write("D \n")
            self.process.stdin.write(f"flowfield.{self.analysis_name} \n")
            self.process.stdin.flush()

            # Dump the forces data
            self.process.stdin.write("F \n")
            self.process.stdin.write(f"forces.{self.analysis_name} \n") 
            self.process.stdin.flush()


    def ExecuteSolver(self,
                      Viscous: bool = False,
                      ) -> int:
        """
        Execute the solver for the current analysis.

        Parameters
        ----------
        Viscous : bool, optional
            If True, generates the outputs corresponding to a viscous analysis. Default is False.

        Returns
        -------
        tuple :
            exit_flag : int
                Exit flag indicating the status of the solver execution.
            iter_counter : int
                Number of iterations performed up until failure of the solver.
        """

        # Enter the execution menu. 
        # For each Newton iteration, write the residuals to dedicated lists for plotting/debugging purposes
        self.process.stdin.write("x \n")
        self.process.stdin.flush()
        self.process.stdin.write("1 \n")
        self.process.stdin.flush()  # Flush console input to ensure the execution process is started

        # Set exit flag to -1, which would indicate successful convergence
        # exit flag options are:
        # -1 : Successful
        # 0 : MTSOL crash - likely related to the grid resolution
        # Set the default exit flag to -1, as we assume succesful convergence unless issues occur
        exit_flag = -1 

        # Initialize iteration counter
        iter_counter = 2
        
        # Create output deque to store the last 20 lines of output to
        console_output = deque(maxlen=20)

        # Keep converging 
        while True:
            #Execute iteration
            self.process.stdin.write(f"{self.ITER_STEP_SIZE} \n")
            self.process.stdin.flush()

            # Increase iteration counter by step size
            iter_counter += self.ITER_STEP_SIZE

            while True:
                # Read the output line by line
                line = self.process.stdout.readline()
                console_output.append(line)
                
                # Handle successful convergence of case, which would be the end of the function, returning the exit flag
                if line.startswith(' Converged'):
                    # Exit the iteration subroutine
                    self.process.stdin.write("0 \n")
                    self.process.stdin.flush()

                    # Generate the solver output
                    self.GenerateSolverOutput(Viscous)

                    # return the exit flag and iteration counter
                    return exit_flag, iter_counter
                
                #Handle (unexpected) quitting of program - which would be a crash of MTSOL
                if line == "" and self.process.poll() is not None:  
                    exit_flag = 0
                    return exit_flag, iter_counter
                
                # Stop collecting once end of the iteration is reached, but read one more line
                if line.startswith('         dDoub:'):  
                    break

            

    def HandleShockWaves(self,
                         ) -> None:
        """
        Handle second order diffusion to help resolve shockwaves. 
        """

        pass


    def caller(self) -> int:
        """
        Main execution of MTSOL
        """

        # Generate MTSOL subprocess
        self.GenerateProcess()

        # Write operating conditions
        self.SetOperConditions()

        # Execute inviscid solve
        exit_flag_invisc, iter_count_invisc = self.ExecuteSolver()

        # Toggle viscous on centerbody and outer surface of duct
        self.ToggleViscous([1,3])

        # Execute initial viscous solve
        exit_flag_visci, iter_count_visci = self.ExecuteSolver(Viscous=True)

        # Toggle viscous on inner surface of duct
        self.ToggleViscous([4])

        # Execute complete viscous solve
        exit_flag_viscf, iter_count_viscf = self.ExecuteSolver(Viscous=True)     

        
        return self.process.returncode
    
    
    pass

if __name__ == "__main__": 
    import time
    start_time = time.time()

    filepath = r"mtsol.exe"
    analysisName = "test_case"
    oper = {"Inlet_Mach": 0.2,
            "Inlet_Reynolds": 5E6}

    test = MTSOL_call(oper, filepath, analysisName).caller()


    end_time = time.time()
    print(f"Execution of MTSOL_call took {end_time - start_time} seconds")