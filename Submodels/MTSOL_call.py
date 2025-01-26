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
import shutil
import glob
import re
from typing import Any, Optional
from enum import Enum
from collections import deque


class ExitFlag(Enum):
    """
    Enum class to define the exit flags for the MTSOL solver. 

    The exit flags are used to determine the status of the solver execution. 

    Attributes
    ----------
    SUCCESS : int
        Successful completion of the solver execution. 
    CRASH : int
        MTSOL crash - likely related to the grid resolution. 
    NON_CONVERGENCE : int
        Non-convergence, to be handled by the HandleNonConvergence function. 
    NOT_PERFORMED : int
        Not performed, with no iterations executed or outputs generated. 
    """

    SUCCESS = -1
    CRASH = 0
    NON_CONVERGENCE = 1
    NOT_PERFORMED = 2
    COMPLETED = 3


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
        self.ITER_STEP_SIZE = 1  # Step size in which iterations are performed in MTSOL
        self.SAMPLE_SIZE = 10  # Number of iterations to use to average over in case of non-convergence. 
        self.ITER_LIMIT = 50 # Maximum number of iterations to perform before non-convergence is assumed.


    

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
        interface_output = deque(maxlen=20)  # Create deque to store the last 20 lines of console output to
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
            # Only toggle if the element currently has the viscous option enabled. 
            if interface_output[idx_first_element + i][4] == "*":
                self.process.stdin.write(f"V {interface_output[idx_first_element + i][2]} \n")
                self.process.stdin.flush()	
            toggles.append(int(interface_output[idx_first_element + i][2]))
        self.element_counts = toggles
        
        # Exit the modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()
    

    def ToggleViscous(self,
                      elements: Optional[list[int]],
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
                print(self.element_counts)
                print(elements)
                raise OSError(f"element is not in the elements counted in the solution parameters menu!") from None
            self.process.stdin.write(f"V {','.join(map(str, elements))} \n")
        else:
            self.process.stdin.write(f"V {','.join(map(str, self.element_counts))} \n")
        self.process.stdin.flush()

        # Exit the Modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()


    def WaitForCompletion(self,
                          type: int = 1,
                          ) -> int:
        """
        Check the console output to ensure that iterations are completed.

        Parameters
        ----------
        type : str, optional
            Type of completion to check for. Default is 1, which corresponds to an iteration. 
            Other option is 2, which corresponds to output generation. 

        Returns
        -------
        exit_flag : int
            Exit flag indicating the status of the solver execution. -1 indicates successful completion, 0 indicates a crash, 
            and 3 indicates completion of the iteration, but NOT convergence.
        """

        # Check the console output to ensure that iterations are completed
        console_output = deque(maxlen=20)
        while True:
            # Read the output line by line
            line = self.process.stdout.readline()
            console_output.append(line)
            
            # Once iteration is complete, return the completed exit flag
            if line.startswith('         dDoub') and type == 1:
                print("condition 1")
                return ExitFlag.COMPLETED.value
            
            # Once the iteration is converged, return the converged exit flag
            elif line.startswith(' Converged') and type == 1:
                print("condition 2")
                return ExitFlag.SUCCESS.value
            
            # Once the solution is written to the state file, return the completed exit flag
            elif line.startswith(' Solution') and type == 2:
                print("condition 3")
                return ExitFlag.COMPLETED.value
            
            # Once the forces file is written, return the completed exit flag. 
            # This can be detected from the prompt to overwrite the file or to enter a filename
            elif (line.startswith(' File exists.  Overwrite?  Y') or line.startswith('Enter filename') or line.startswith(' =')) and type == 2:
                print("condition 4")
                return ExitFlag.COMPLETED.value
            
            # If the solver crashes, return the crash exit flag
            elif line == "" and self.process.poll() is not None:
                return ExitFlag.CRASH.value
            

    def GenerateSolverOutput(self,
                             ) -> None:
        """
        Generate all output files for the current analysis. 
        If a viscous analysis was performed, the boundary layer data is also dumped to the corresponding file.
        Requires that MTSOL is in the main menu when starting this function. 

        Returns
        -------
        None
        """

        # Update the solution state file
        self.process.stdin.write("W \n")
        self.process.stdin.flush()

        # Check if the solution state file is written successfully
        self.WaitForCompletion(type=2)

        # Dump the forces data
        self.process.stdin.write("F \n")
        self.process.stdin.flush()

        self.process.stdin.write(f"forces.{self.analysis_name} \n") 
        self.process.stdin.flush()

        self.process.stdin.write("Y \n")
        self.process.stdin.flush()
        
        # Check if the forces file is written successfully
        self.WaitForCompletion(type=2)

    def ExecuteSolver(self,
                      ) -> tuple[int, int]:
        """
        Execute the solver for the current analysis.

        Returns
        -------
        tuple :
            exit_flag : int
                Exit flag indicating the status of the solver execution.
            iter_counter : int
                Number of iterations performed up until failure of the solver.
        """

        # Initialize iteration counter
        iter_counter = 0
        
        # Keep converging until the iteration count exceeds the limit
        while iter_counter < self.ITER_LIMIT:
            #Execute iterations
            self.process.stdin.write(f"x {self.ITER_STEP_SIZE} \n")
            self.process.stdin.flush()

            # Increase iteration counter by step size
            iter_counter += self.ITER_STEP_SIZE

            # Wait for the current iteration to complete
            exit_flag = self.WaitForCompletion(type=1)

            # Check the exit flag to see if the solution has converged
            # If the solution has converged, break out of the iteration loop
            if exit_flag == ExitFlag.SUCCESS.value:
                return exit_flag, iter_counter

        # If the iteration limit is reached, return the non-convergence exit flag
        return ExitFlag.NON_CONVERGENCE.value, iter_counter


    def GetAverageValues(self,
                         file_name: str = 'forces',
                         skip_lines: dict = {2, 14, 26, 27, 32, 37, 42},
                         ) -> None:
        """
        Read the output files from the MTSOL_output_files directory and average the values to obtain the assumed true values in case of non-convergence.

        Parameters
        ----------
        file_name : str, optional
            The name of the file to read the values from. Default is 'forces'.
        skip_lines : dict, optional
            A set of line numbers to skip when averaging the values. Default is {2, 14, 26, 27, 32, 37, 42}. This corresponds to certain lines in the forces.xxx file which do not need to be averaged
        
        Returns
        -------
        None
        """

        # Construct output file name
        output_file = f'{file_name}.{self.analysis_name}'

        # Construct a simple local function to load in the contents of each file in the directory
        def read_file(filename):
            with open(filename, 'r') as file:
                return file.readlines()

        # Read all files in the directory
        file_pattern = f'MTSOL_output_files/{file_name}.{self.analysis_name}*'
        files = glob.glob(file_pattern)
        content = [read_file(file) for file in files]
        
        # Transpose content to group corresponding lines together
        transposed_content = list(map(list, zip(*content)))
        average_content = []

        # Regular expression to match variable = value pairs with varying spaces
        var_value_pattern = re.compile(r'([\w\s]+)\s*=\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)')

        # Regular expression to match scientific notation and numeric values
        value_pattern = re.compile(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?')

        # Process each group of corresponding lines
        for idx, lines in enumerate(transposed_content):
            line_num = idx + 1

            # Check if the current line should be skipped
            if line_num in skip_lines:
                average_content.append(lines[0])
                continue

            line_text = lines[0]

            # Handling single values after "="
            if all('=' in line and len(line.split('=')[1].split()) == 1 for line in lines):
                values = [float(value_pattern.search(line.split('=')[1]).group()) for line in lines]
                average_value = sum(values) / len(values)
                line_text = f'{line_text.split("=")[0].strip()} = {average_value:.5E}\n'
            
            # Handling multiple values in "variable=data1 variable=data2" structure
            elif all('=' in line for line in lines) and any(len(line.split('=')) > 2 for line in lines):
                var_values_dict = {}
                for line in lines:
                    var_values = var_value_pattern.findall(line)
                    for var, value, _ in var_values:
                        if var not in var_values_dict:
                            var_values_dict[var] = []
                        var_values_dict[var].append(float(value))
                avg_values = [f'{var} = {sum(values) / len(values):.5E}' for var, values in var_values_dict.items()]
                line_text = ' '.join(avg_values) + '\n'
            
            # Handling multiple values separated by spaces in "variable: data1 data2 data3 data4" structure
            elif all(':' in line for line in lines):
                text_part = lines[0].split(':')[0].strip() + ': '
                all_values = [list(map(float, line.split(':')[1].split())) for line in lines]
                avg_values = [sum(col) / len(col) for col in list(zip(*all_values))]
                line_text = text_part + '    '.join(f'{val:.5E}' for val in avg_values) + '\n'

            average_content.append(line_text)

            # Write the averaged content to a new file
            with open(output_file, 'w') as file:
                file.writelines(average_content)


    def HandleNonConvergence(self,
                             ) -> None:
        """
        Average over the last self.SAMPLE_SIZE iterations to determine flowfield variables.

        Returns
        -------
        None
        """

        # Create subfolder to put all output files into if the folder doesn't already exist
        dump_folder = r"\\MTSOL_output_files\\"
        os.makedirs(dump_folder, 
                    exist_ok=True)

        # Initialize iteration counter
        iter_counter = 0

        # Keep looping until iter_count exceeds the target value for number of iterations to average 
        while iter_counter <= self.SAMPLE_SIZE:
            #Execute iteration
            self.process.stdin.write("x 1 \n")
            self.process.stdin.flush()

            # Wait for current iteration to complete
            self.WaitForCompletion(type=1)

            # Generate solver outputs
            self.GenerateSolverOutput()

            # Rename file to indicate the iteration number, and avoid overwriting the same file. 
            # Also move the file to the output folder
            shutil.move(os.replace(f"forces.{self.analysis_name}", f"forces.{self.analysis_name}.{iter_counter}"),
                        dump_folder,
                        )

            # Increase iteration counter by step size
            iter_counter += self.ITER_STEP_SIZE

        # Average the data from all the iterations to obtain the assumed true values. This effectively assumes that the iterations are oscillating about the true value.
        # This is a simplification, but it is the best we can do in this case.
        # The average is calculated by summing all the values and dividing by the number of iterations.
        self.GetAverageValues()

        # After averaging, the individual generated files are no longer needed and can be deleted.
        shutil.rmtree(dump_folder)


    def CrashRecovery(self,
                      case_type: str,
                      iter_count: int,
                      ) -> None:
        """
        Recover from a crash of the MTSOL solver.

        Parameters
        ----------
        case_type : str
            Type of case that was run.
        iter_count : int
            Number of iterations performed up until failure of the solver.

        Returns
        -------
        None
        """

        # TODO: Write crash recovering code. 
        #  potentiall uses the number of iterations up to failure to check the flowfield and determine the cause of the crash.

        pass
    

    def HandleExitFlag(self,
                       exit_flag: int,
                       iter_count: int,
                       case_type: str,
                       ) -> None:
        """
        Handle the exit flag of the solver execution. Handle non-convergence, crashes, and successful completion accordingly.

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
        elif exit_flag == ExitFlag.SUCCESS.value or exit == ExitFlag.NOT_PERFORMED.value:
            return None
        else:
            raise OSError(f"Unknown exit flag {exit_flag} encountered!") from None


    def caller(self,
               Run_viscous: bool = False) -> tuple[int, list[tuple[int]]]:
        """
        Main execution of MTSOL
        """

        # Define initial exit flags and iteration counters
        # Note that a negative iteration count indicates that the solver did not run
        exit_flag_invisc = ExitFlag.NOT_PERFORMED.value
        iter_count_invisc = -1
        exit_flag_visci = ExitFlag.NOT_PERFORMED.value
        iter_count_visci = -1
        exit_flag_viscf = ExitFlag.NOT_PERFORMED.value
        iter_count_viscf = -1

        # Generate MTSOL subprocess
        self.GenerateProcess()

        # Write operating conditions
        self.SetOperConditions()

        # Execute inviscid solve
        exit_flag_invisc, iter_count_invisc = self.ExecuteSolver()

        # Handle solver based on exit flag
        self.HandleExitFlag(exit_flag_invisc, 
                            iter_count_invisc, 
                            "inviscid",
                            )
        
        print("inviscid done", iter_count_invisc)

        # Only run a viscous solve if required by the user and if the inviscid solve was successful
        if Run_viscous and exit_flag_invisc == ExitFlag.SUCCESS.value:
            # Toggle viscous on centerbody and outer surface of duct
            self.ToggleViscous([1, 3])

            # Execute initial viscous solve
            exit_flag_visci, iter_count_visci = self.ExecuteSolver()

            # Handle solver based on exit flag
            self.HandleExitFlag(exit_flag_visci,
                                iter_count_visci,
                                "initial_viscous",
                                )
            
            print("initial viscous done", iter_count_visci)
            
            # Execute complete viscous solve only if initial viscous run was successful
            if exit_flag_visci == ExitFlag.SUCCESS.value:
                # Toggle viscous on inner surface of duct
                self.ToggleViscous([4])

                # Execute final viscous solve
                exit_flag_viscf, iter_count_viscf = self.ExecuteSolver() 

                # Handle solver based on exit flag
                self.HandleExitFlag(exit_flag_viscf,
                                    iter_count_viscf,
                                    "final_viscous",
                                    )
                
                print("final viscous done", iter_count_viscf)
        
        # Generate the solver output
        self.GenerateSolverOutput()
        
        # Close the MTSOL tool
        self.process.stdin.write("Q \n")
        self.process.stdin.flush()

        # Check that MTSOL has closed successfully 
        if self.process.poll() is not None:
            try:
                self.process.wait(timeout=2)
            
            except subprocess.TimeoutExpired:
                self.process.kill()
                raise OSError("MTSOL did not close after completion.") from None
  
        return max(exit_flag_invisc, exit_flag_visci, exit_flag_viscf), [(exit_flag_invisc, iter_count_invisc), (exit_flag_visci, iter_count_visci), (exit_flag_viscf, iter_count_viscf)]


if __name__ == "__main__": 
    import time
    start_time = time.time()

    filepath = r"mtsol.exe"
    analysisName = "test_case"
    oper = {"Inlet_Mach": 0.2,
            "Inlet_Reynolds": 5E6}

    test = MTSOL_call(oper, filepath, analysisName).caller(Run_viscous=True)


    end_time = time.time()
    print(f"Execution of MTSOL_call took {end_time - start_time} seconds")