"""
MTSOL_call
=============

Description
-----------
This module provides an interface to interact with the MTSOL executable from Python. 
It creates a subprocess for the MSOL executable, executes an inviscid, and viscid if desired, solve, 
and writes the output data to the state file and the forces.xxx output file.

Classes
-------
ExitFlag
    An Enum class with exit flags for the MTSOL interface
MTSOL_call
    A class to handle the interface between Python and the MTSOL executable.

Examples
--------
>>> filepath = r"mtsol.exe"
>>> analysisName = "test_case"
>>> oper = {"Inlet_Mach": 0.2000,
>>>         "Inlet_Reynolds": 5.000E6,
>>>         "Pressure_Ratio_Exit": 0.99,
>>>         }
>>> test = MTSOL_call(oper, filepath, analysisName).caller(Run_viscous=True)

Notes
-----
This module is designed to work with the MTSOL executable. Ensure that the executable and the input state file, tdat.xxx, 
are present in the same directory as this Python file. When executing the file as a standalone, it uses the inputs 
and calls contained within the if __name__ == "__main__" section. This part also imports the time module to measure 
the time needed to perform each file generation call. This is beneficial in runtime optimization.

References
----------
The required input data, limitations, and structures are documented within the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 0.9

Changelog:
- V0.0: File created with empty class as placeholder.
- V0.9: Minimum Working Example. Lacks crash handling and pressure ratio definition. 
"""

import subprocess
import os
import shutil
import glob
import re
from typing import Any
from enum import Enum


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
    Class to handle the interface between MTSOL and Python
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
        self.operating_conditions: dict = operating_conditions
        self.fpath: str = file_path
        self.analysis_name: str = analysis_name

        # Define constants for the class
        self.ITER_STEP_SIZE = 2  # Step size in which iterations are performed in MTSOL
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
                                        bufsize=1,
                                        )
        
        # Check if subprocess is started successfully
        if self.process.poll() is not None:
            raise ImportError(f"MTSOL or tdat.{self.analysis_name} not found in {self.fpath}") from None    
               

    def SetOperConditions(self,
                          ) -> None:
        """
        Set the inlet Mach number and critical amplification factor, and set the Reynolds number equal to zero to ensure an inviscid case is obtained. 

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

        # Set the Reynolds number to 0 to ensure an inviscid solve is performed initially
        self.process.stdin.write("R 0 \n")
        self.process.stdin.flush()
        
        # Exit the modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()
    

    def ToggleViscous(self,
                      ) -> None:
        """
        Toggle the viscous setting for all elements by setting the inlet Reynolds number.
        Note that the Reynolds number is defined using the reference length LREF=1 m

        Returns
        -------
        None
        """

        # Enter the Modify solution parameters menu
        self.process.stdin.write("m \n")
        self.process.stdin.flush()

        # Set the viscous Reynolds number, calculated using the length self.LREF = 1!
        self.process.stdin.write(f"R {self.operating_conditions["Inlet_Reynolds"]} \n")
        self.process.stdin.flush()

        # Exit the Modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()

        # Wait for the change to be processed in MTSOL
        self.WaitForCompletion(type=3)
        

    def WaitForCompletion(self,
                          type: int = 1,
                          ) -> int:
        """
        Monitor the console output to verify the completion of a command.

        Parameters
        ----------
        type : int
            Specifies the type of completion to monitor. Default is 1, which corresponds to an iteration.
            Other options are: 2 for output generation and 3 for changing operating conditions.

        Returns
        -------
        exit_flag : int
            Exit flag indicating the status of the solver execution. -1 indicates successful completion, 0 indicates a crash,
            and 3 indicates the completion of the iteration without convergence.
        """

        # Check the console output to ensure that commands are completed
        while True:
            # Read the output line by line
            line = self.process.stdout.readline()
            #print(line)
            # Once iteration is complete, return the completed exit flag
            if line.startswith(' =') and type == 1:
                return ExitFlag.COMPLETED.value
            
            # Once the iteration is converged, return the converged exit flag
            elif 'Converged' in line and type == 1:
                return ExitFlag.SUCCESS.value
            
            # Once the solution is written to the state file, return the completed exit flag
            elif 'Solution written to state file' in line and type == 2:
                return ExitFlag.COMPLETED.value
            
            # Once the forces file is written, return the completed exit flag. 
            # This can be detected from the prompt to overwrite the file or to enter a filename
            elif (line.startswith(' File exists.  Overwrite?  Y') or line.startswith('Enter filename')) and type == 2:
                return ExitFlag.COMPLETED.value
            
            # When changing the operating conditions, check for the end of the modify parameters menu           
            elif line.startswith(' V1,2..') and type == 3:
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

        # Dump the flowfield data
        self.process.stdin.write("D \n")
        self.process.stdin.flush()

        self.process.stdin.write(f"flowfield.{self.analysis_name} \n")
        self.process.stdin.flush()

        self.process.stdin.write("Y \n")
        self.process.stdin.flush()      

        # Check if the flowfield file is written successfully
        self.WaitForCompletion(type=2)  


    def ExecuteSolver(self,
                      ) -> tuple[int, int]:
        """
        Execute the MTSOL solver for the current analysis.

        Returns
        -------
        tuple :
            exit_flag : int
                Exit flag indicating the status of the solver execution.
            iter_counter : int
                Number of iterations performed up until failure of the solver.
        """

        # Initialize iteration counter - set to -1 to ensure correct total count at end of convergence
        iter_counter = -1

        # Initialize exit flag
        exit_flag = ExitFlag.NON_CONVERGENCE.value

        # Keep converging until the iteration count exceeds the limit
        while iter_counter < self.ITER_LIMIT:
            # Check the exit flag to see if the solution has converged
            # If the solution has converged, break out of the iteration loop
            exit_flag = self.WaitForCompletion(type=1)
            if exit_flag == ExitFlag.SUCCESS.value:
                return exit_flag, iter_counter
            else:
                #Execute next iteration(s)
                self.process.stdin.write(f"x {self.ITER_STEP_SIZE} \n")
                self.process.stdin.flush()

                # Increase iteration counter by step size
                iter_counter += self.ITER_STEP_SIZE           

        # Return the exit flag and iteration counter
        return exit_flag, iter_counter


    def GetAverageValues(self,
                         file_name: str = 'forces',
                         ) -> None:
        """
        Read the output files from the MTSOL_output_files directory and average the values to obtain the assumed true values in case of non-convergence.

        Parameters
        ----------
        file_name : str, optional
            The name of the file to read the values from. Default is 'forces'.
        
        Returns
        -------
        None
        """

        # Construct output file name
        output_file = f'{file_name}.{self.analysis_name}'

        # Read all files in the directory
        file_pattern = f'MTSOL_output_files/{file_name}.{self.analysis_name}*'
        files = glob.glob(file_pattern)
        content = []
        for file in files:
            with open(file) as f:
                content.append(f.readlines())
        
        # Transpose content to group corresponding lines together
        transposed_content = list(map(list, zip(*content)))
        average_content = []

        # Regular expression to match variable = value pairs with varying spaces
        var_value_pattern = re.compile(r'([\w\s]+)\s*=\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)')

        # Regular expression to match scientific notation and numeric values
        value_pattern = re.compile(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?')

        # Process each group of corresponding lines
        for lines in transposed_content:
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
        elif (exit_flag == ExitFlag.COMPLETED.value or 
              exit_flag == ExitFlag.SUCCESS.value or 
              exit_flag == ExitFlag.NOT_PERFORMED.value):
            return
        else:
            raise OSError(f"Unknown exit flag {exit_flag} encountered!") from None
    

    def caller(self,
               Run_viscous: bool = False,
               ) -> tuple[int, list[tuple[int]]]:
        """
        Main execution interface of MTSOL.

        Parameters
        ----------
        Run_viscous : bool, optional
            Flag to indicate whether to run a viscous solve. Default is False.

        Returns
        -------
        tuple :
            maximum_exit_flag : int
                Exit flag indicating the status of the solver execution. Is equal to the maximum value of the inviscid and viscous exit flags, since exit_flag > -1 indicate failed/nonconverging solves.
                This is used as a one-variable status indicator, while the corresponding output list gives more details. 
            list :
                A list of tuples containing the exit flags and iteration counts for the inviscid and viscous solves.
        """

        # Define initial exit flags and iteration counters
        # Note that a negative iteration count indicates that the solver did not run
        exit_flag_invisc = ExitFlag.NOT_PERFORMED.value
        iter_count_invisc = -1
        exit_flag_visc = ExitFlag.NOT_PERFORMED.value
        iter_count_visc = -1

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

        # Only run a viscous solve if required by the user and if the inviscid solve was successful
        if Run_viscous and exit_flag_invisc == ExitFlag.SUCCESS.value:
            # Toggle viscous on all surfaces
            self.ToggleViscous()

            # Execute viscous solve
            exit_flag_visc, iter_count_visc = self.ExecuteSolver()

            # Handle solver based on exit flag
            self.HandleExitFlag(exit_flag_visc,
                                iter_count_visc,
                                "viscous",
                                )
        
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
  
        return max(exit_flag_invisc, exit_flag_visc), [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)]


if __name__ == "__main__": 
    import time
    
    filepath = r"mtsol.exe"
    analysisName = "test_case"
    oper = {"Inlet_Mach": 0.2000,
            "Inlet_Reynolds": 5.000E6}

    start_time = time.time()
    test = MTSOL_call(oper, filepath, analysisName).caller(Run_viscous=True)
    end_time = time.time()

    print(f"Execution of MTSOL_call took {end_time -  start_time} seconds")