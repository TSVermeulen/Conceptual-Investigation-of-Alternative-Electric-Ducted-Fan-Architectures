"""
MTSOL_call
=============

Description
-----------
This module provides an interface to interact with the MTSOL executable from Python. 
It creates a subprocess for the MTSOL executable, executes an inviscid, and viscid if desired, solve, 
and writes the output data to the state file and the forces.xxx output file.

Classes
-------
ExitFlag
    An Enum class with exit flags for the MTSOL interface
MTSOL_call
    A class to handle the interface between Python and the MTSOL executable.

Examples
--------
>>> analysisName = "test_case"
>>> oper = {"Inlet_Mach": 0.25,
>>>         "Inlet_Reynolds": 5.000E6,
>>>         "N_crit": 9,
>>>         }
>>> test = MTSOL_call(oper, analysisName).caller(run_viscous=True)

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
Version: 1.3

Changelog:
- V0.0: File created with empty class as placeholder.
- V0.9: Minimum Working Example. Lacks crash handling and pressure ratio definition. 
- V0.9.5: Cleaned up inputs, removing file_path and changing it to a constant.
- V1.0: With implemented choking handling, pressure ratio definition is no longer needed. Added choking exit flag. Cleaned up/updated HandleExitFlag() method. Added critical amplification factor as input. 
- V1.1: Added file processing check to ensure that the forces file is copied before it is deleted. Added a check to ensure that the MTSOL executable is present in the same directory as this Python file.
- V1.2: Added watchdog to check if the forces file is created before copying it.
- V1.3: Added crash handling for the inviscid solve. Added a function to set all values to zero in case of a crash during the inviscid solve. Added a function to handle non-convergence by averaging over the last self.SAMPLE_SIZE iterations. Added a function to handle the exit flag of the solver execution. Added a function to handle the convergence of individual surfaces in case of a crash during the viscous solve.
"""

import subprocess
import os
import shutil
import glob
import re
from collections import OrderedDict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from enum import Enum
from pathlib import Path
import time


class FileCreatedHandling(FileSystemEventHandler):
    """ 
    Simple class to handle checking if forces.analysis_name file has been generated. 
    """

    def __init__(self, 
                 file_path: str, 
                 destination: str) -> None:
        self.file_path = file_path
        self.destination = destination
        self.file_processed = False

    def on_created(self, event):
        """ Handle copying of the forces.analysis_name output file."""
        if event.src_path == self.file_path:
            shutil.copy(self.file_path, self.destination)
            os.remove(self.file_path)
            self.file_processed = True
        

    def on_modified(self, event):
        self.on_created(event)

    
    def is_file_processed(self) -> bool:
        """
        Return whether the file has been processed.

        Returns
        -------
        - bool
            True if the file has been processed, False otherwise.
        """
        return self.file_processed


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
    CHOKING : int
        Choking occurs somewhere in the solution, indicated by the 'QSHIFT' message in the MTSOL console output
    """

    SUCCESS = -1
    CRASH = 0
    NON_CONVERGENCE = 1
    NOT_PERFORMED = 2
    COMPLETED = 3
    CHOKING = 4


class MTSOL_call:
    """
    Class to handle the interface between MTSOL and Python
    """

    def __init__(self,
                 operating_conditions: dict,
                 analysis_name: str,
                 ) -> None:
        """
        Initialize the MTSOL_call class.

        This method sets up the initial state of the class.

        Parameters
        ----------
        - operating_conditions : dict
            A dictionary containing the operating conditions for the MTSOL analysis. The dictionary needs to contain:
                - Inlet_Mach: the inlet Mach number
                - Inlet_Reynolds: the inlet Reynolds number, calculated using L=1m
                - N_crit: the critical amplification factor
        - analysis_name : str
            A string of the analysis name. 

        Returns
        -------
        None
        """

        self.operating_conditions = operating_conditions
        self.analysis_name = analysis_name

        # Define constants for the class
        self.ITER_STEP_SIZE = 2  # Step size in which iterations are performed in MTSOL
        self.SAMPLE_SIZE = 10  # Number of iterations to use to average over in case of non-convergence. 
        self.ITER_LIMIT = 50  # Maximum number of iterations to perform before non-convergence is assumed.

        # Define filepath of MTSOL as being in the same folder as this Python file
        self.fpath: str = os.getenv('MTSOL_PATH', 'mtsol.exe')
        if not os.path.exists(self.fpath):
            raise FileNotFoundError(f"MTSOL executable not found at {self.fpath}")


    def StdinWrite(self,
                   command: str) -> None:
        """
        Simple function to write commands to the subprocess stdin in order to pass commands to MTSOL

        Parameters
        ----------
        - command : str
            The text-based command to pass to MTSOL

        Returns
        -------
        None
        """

        self.process.stdin.write(f"{command} \n")
        self.process.stdin.flush()


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
        self.StdinWrite("m")

        # Write inlet Mach number
        self.StdinWrite(f"M {self.operating_conditions['Inlet_Mach']}")

        # Set critical amplification factor to N=9 rather than the default N=7
        self.StdinWrite(f"N {self.operating_conditions['N_crit']}")

        # Set the Reynolds number to 0 to ensure an inviscid solve is performed initially
        self.StdinWrite("R 0")

        # Set momentum/entropy conservation Smom flag
        self.StdinWrite("S 4")

        # Exit the modify solution parameters menu
        self.StdinWrite("")
    

    def ToggleViscous(self,
                      ) -> None:
        """
        Toggle the viscous setting for the centrebody by setting the inlet Reynolds number. 
        Note that the Reynolds number must be defined using the MTFLOW reference length LREF. 

        Returns
        -------
        None
        """

        # Enter the Modify solution parameters menu
        self.StdinWrite("m")

        # Set the viscous Reynolds number, calculated using the length self.LREF = 1!
        self.StdinWrite(f"R {self.operating_conditions['Inlet_Reynolds']}")

        # Disable the viscous toggle on surfaces 3,4
        # This ensures the initial viscous run is only performed on the centerbody BL. 
        # Successive toggling of the other handles can then improve numerical stability
        self.StdinWrite("V3,4")

        # Exit the Modify solution parameters menu
        self.StdinWrite("")

        # Wait for the change to be processed in MTSOL
        self.WaitForCompletion(type=3)


    def SetViscous(self,
                   surface_ID: int,
                   ) -> None:
        """
        Set the viscous toggle for the given surface identifier.

        Parameters
        ----------
        - surface_ID : int
            ID of the surface which is to be toggled. For a ducted fan, the ID should be either 1, 3, or 4. 
        
        Returns
        -------
        None
        """
        
        # Enter the Modify solution parameters menu
        self.StdinWrite("m")

        # Toggle the given surface
        self.StdinWrite(f"V{surface_ID}")

        # Exit the Modify solution parameters menu
        self.StdinWrite("")

        # Wait for the change to be processed in MTSOL
        self.WaitForCompletion(type=3)
  

    def WaitForCompletion(self,
                          type: int = 1,
                          output_file: str = None
                          ) -> int:
        """
        Monitor the console output to verify the completion of a command.

        Parameters
        ----------
        - type : int
            Specifies the type of completion to monitor. Default is 1, which corresponds to an iteration.
            Other options are: 2 for output generation and 3 for changing operating conditions.
        - output_file : str, optional
            A string of the output file for which the completion is to be monitored. Either 'forces', 'flowfield', or 'boundary_layer'. 
            Note that the file extension (i.e.) casename, should not be included!

        Returns
        -------
        - exit_flag : int
            Exit flag indicating the status of the solver execution. -1 indicates successful completion, 0 indicates a crash,
            and 3 indicates the completion of the iteration without convergence.
        """

        # Check the console output to ensure that commands are completed
        while True:
            # Read the output line by line
            line = self.process.stdout.readline()
            # Once iteration is complete, return the completed exit flag
            if line.startswith(' =') and type == 1:
                exit_flag = ExitFlag.COMPLETED.value
                break
            
            # Once the iteration is converged, return the converged exit flag
            elif 'Converged' in line and type == 1:
                exit_flag = ExitFlag.SUCCESS.value
                break

            # If choking occurs, return the exit flag to choking
            elif ' *** QSHIFT: Mass flow or Pexit must be a DOF!' in line and type == 1:
                exit_flag = ExitFlag.CHOKING.value
                break
            
            # Once the solution is written to the state file, or the forces/flowfield file is written, return the completed exit flag
            # The succesful forces/flowfield writing can be detected from the prompt to overwrite the file or to enter a filename
            elif ('Solution written to state file' in line 
                  or line.startswith((' File exists.  Overwrite?  Y', 'Enter filename'))) and type == 2:
                exit_flag = ExitFlag.COMPLETED.value

                if output_file is not None:
                    max_wait_time = 5  # Maximum wait time in seconds
                    start_time = time.time()
                    # Wait for the file creation to be finished
                    while not os.path.exists(f'{output_file}.{self.analysis_name}'):
                        time.sleep(0.01) 
                        if time.time() - start_time > max_wait_time:
                            break 
                break
                        
            # When changing the operating conditions, check for the end of the modify parameters menu           
            elif line.startswith(' V1,2..') and type == 3:
                exit_flag = ExitFlag.COMPLETED.value
                break
            
            # If the solver crashes, return the crash exit flag
            elif line == "" and self.process.poll() is not None:
                exit_flag = ExitFlag.CRASH.value    
                break 

        return exit_flag 
    

    def WriteStateFile(self,
                       ) -> None:
        """
        Writes the current solution to the state file tdat.analysis_name.

        Returns
        -------
        None
        """

        # Update the solution state file
        self.StdinWrite("W")

        # Check if the solution state file is written successfully
        self.WaitForCompletion(type=2)
            

    def GenerateSolverOutput(self,
                             output_type: int = 0) -> None:
        """
        Generate all output files for the current analysis. 
        If a viscous analysis was performed, the boundary layer data is also dumped to the corresponding file.
        Requires that MTSOL is in the main menu when starting this function. 

        Parameters
        ----------
        - output_type : int
            A control integer to determine which output files need to be generated. 0 corresponds to the forces file, while 1 will generate all files.

        Returns
        -------
        None
        """

        # Update the solution state file
        self.WriteStateFile()

        # First delete the output files if they exist already
        forces_file = f"forces.{self.analysis_name}"
        flowfield_file = f"flowfield.{self.analysis_name}"
        boundary_layer_file = f"boundary_layer.{self.analysis_name}"

        if os.path.exists(forces_file):
            os.remove(forces_file)
        if os.path.exists(flowfield_file):
            os.remove(flowfield_file)
        if os.path.exists(boundary_layer_file):
            os.remove(boundary_layer_file)

        # Dump the forces data
        self.StdinWrite("F")
        self.StdinWrite(forces_file) 
        
        # Check if the forces file is written successfully
        self.WaitForCompletion(type=2,
                               output_file='forces')
    
        if output_type == 0:
            return

        # Dump the flowfield data
        self.StdinWrite("D")
        self.StdinWrite(flowfield_file)

        # Check if the flowfield file is written successfully
        self.WaitForCompletion(type=2,
                               output_file='flowfield')  

        # Dump the boundary layer data
        self.StdinWrite("B")
        self.StdinWrite(boundary_layer_file)

        # Check if the boundary layer file is written successfully
        self.WaitForCompletion(type=2,
                               output_file='boundary_layer')             


    def ExecuteSolver(self,
                      ) -> tuple[int, int]:
        """
        Execute the MTSOL solver for the current analysis.

        Returns
        -------
        - tuple :
            exit_flag : int
                Exit flag indicating the status of the solver execution.
            iter_count : int
                Number of iterations performed up until failure of the solver.
        """

        # Initialize iteration count 
        iter_count = 0    
        self.iter_counter = iter_count    

        # Keep converging until the iteration count exceeds the limit
        while iter_count < self.ITER_LIMIT:
            #Execute next iteration(s)
            self.StdinWrite(f"x {self.ITER_STEP_SIZE}")

            # Increase iteration counter by step size
            iter_count += self.ITER_STEP_SIZE     

            # Check the exit flag to see if the solution has converged
            # If the solution has converged, break out of the iteration loop
            exit_flag = self.WaitForCompletion(type=1)
            if exit_flag in (ExitFlag.SUCCESS.value, ExitFlag.CHOKING.value):
                break    

        # If the solver has not converged within self.ITER_LIMIT iterations, set the exit flag to non-convergence
        if exit_flag not in (ExitFlag.SUCCESS.value, ExitFlag.CHOKING.value):
            exit_flag = ExitFlag.NON_CONVERGENCE.value

        # Return the exit flag and iteration counter
        return exit_flag, iter_count


    def GetAverageValues(self,
                         ) -> None:
        """
        Read the output force files from the MTSOL_output_files directory and average the values to obtain the assumed true values in case of non-convergence.
        
        Returns
        -------
        None
        """

        # Creatge a file generator to read the output files
        def read_file_lines(file_pattern: str):
            # Generator to yield lines from files matching the pattern
            for file in glob.iglob(file_pattern):
                with open(file, "r") as f:
                    yield f.readlines()

        # Construct output file name and file pattern
        output_file = f'forces.{self.analysis_name}'
        file_pattern = f'MTSOL_output_files/forces.{self.analysis_name}*'

        # Read in all files (collect into a list so we can transpose later)
        content = list(read_file_lines(file_pattern))
        if not content:
            return

        # Transpose content to group corresponding lines together from all files
        transposed_content = list(map(list, zip(*content)))
        average_content = []

        # Regular expression for key=value pairs.
        # We use (?:...) for the exponent part so that findall returns just two groups.
        var_value_pattern = re.compile(
            r'([\w\s]+?)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        )
        # Regular expression for generic numeric value (scientific-notated numbers, etc.)
        value_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

        # Indices of header lines that should not be averaged
        skip_lines = [0, 1, 2, 3, 4, 5, 13, 25, 26, 31, 36, 41]

        # Process each group of corresponding lines
        for idx, lines in enumerate(transposed_content):
            # Start with the first file’s line as the base
            line_text = lines[0]

            # Skip header or pre-defined lines
            if idx in skip_lines:
                average_content.append(line_text)
                continue

            # Case 1: Single key=value pair per line
            if all('=' in line and len(line.split('=')[1].split()) == 1 for line in lines):
                values = [
                    float(value_pattern.search(line.split('=')[1]).group())
                    for line in lines
                ]
                average_value = sum(values) / len(values)
                key = line_text.split("=")[0].strip()
                line_text = f'{key} = {average_value:.5E}\n'

            # Case 2: Multiple key=value pairs in one line (e.g. "var1=data1 var2=data2")
            # Instead of checking for extra '=' via split, we count regex matches.
            elif all('=' in line for line in lines) and any(len(var_value_pattern.findall(line)) > 1 for line in lines):
                var_values_dict = OrderedDict()
                for line in lines:
                    # Find all key/value pairs in the line
                    for var, value in var_value_pattern.findall(line):
                        var = var.strip()  # ensure the variable name has no extra whitespace
                        var_values_dict.setdefault(var, []).append(float(value))
                # Compute average for each encountered key, preserving order from the first appearance.
                avg_values = [
                    f"{var} = {sum(vals) / len(vals):.5E}"
                    for var, vals in var_values_dict.items()
                ]
                line_text = " ".join(avg_values) + "\n"

            # Case 3: Lines with a colon and multiple space-separated values (e.g. "variable: data1 data2 data3 data4")
            elif all(':' in line for line in lines):
                text_part = lines[0].split(':')[0].strip() + ': '
                all_values = [list(map(float, line.split(':')[1].split())) for line in lines]
                # Average each column of values
                avg_values = [sum(col) / len(col) for col in zip(*all_values)]
                line_text = text_part + '    '.join(f'{val:.5E}' for val in avg_values) + '\n'

            # Case 4: Lines with unnamed values separated by varying spaces
            elif all(re.match(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)*', line) for line in lines):
                all_values = [list(map(float, re.split(r'\s+', line.strip()))) for line in lines]
                avg_values = [sum(col) / len(col) for col in zip(*all_values)]
                line_text = '    '.join(f'{val:.5E}' for val in avg_values) + '\n'

            average_content.append(line_text)

        # Write the averaged content to the output file
        with open(output_file, 'w') as file:
            file.writelines(average_content)


    def GenerateCrashOutputs(self,
                             ) -> None:
        """
        Generate a new output file with all values set to zero in case of a crash during the inviscid solve.

        Loads the output forces file and replace all numeric variable values with zero. 
            
        The function handles various line formats:
            - Single key/value lines: "variable = value"
            - Lines with multiple key/value pairs: "var1 = value1 var2 = value2" 
            - Colon-separated formats: "variable: num1 num2 num3 ..."
            - Lines with unnamed numbers separated by spaces.
            
        Header lines (determined by their index) are left unchanged. The modified content is written
        to the input file.

        Returns
        -------
        None
        """
            
        # Define the filepath
        file_path = f'forces.{self.analysis_name}'

        # To avoid mis-interpretation of files, delete the flowfield and boundary layer files, if they exist. 
        # For a crash output, these files are not needed, as we only use the forces.xxx file. 
        if os.path.exists(f"flowfield.{self.analysis_name}"):
            os.remove(f"flowfield.{self.analysis_name}")
        if os.path.exists(f"boundary_layer.{self.analysis_name}"):
            os.remove(f"boundary_layer.{self.analysis_name}")
            
        # Load in the forces file line-by-line
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Define the header (or skip) line indices which should remain unchanged.
        skip_lines = {0, 1, 2, 3, 4, 5, 13, 25, 26, 31, 36, 41}
            
        # Regular expression to match key/value pairs
        var_value_pattern = re.compile(r'([\w\s]+?)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
                        
        # Zero value used to replace all values
        zero_value = '0'
            
        new_content = []
            
        for idx, line in enumerate(lines):
            # Leave header lines unchanged
            if idx in skip_lines:
                new_content.append(line)
                continue
                
            # Case 1: Single key/value pair
            if '=' in line and len(line.split('=')[1].split()) == 1:
                key = line.split("=")[0].strip()
                new_line = f"{key} = {zero_value}\n"
                
            # Case 2: Multiple key/value pairs on a single line (e.g., "var1 = value1 var2 = value2")
            elif '=' in line and len(var_value_pattern.findall(line)) > 1:
                key_val_list = var_value_pattern.findall(line)
                new_line = " ".join(f"{var.strip()} = {zero_value}" for var, _ in key_val_list) + "\n"
                
            # Case 3: Colon-separated values (e.g., "variable: num1 num2 num3")
            elif ':' in line:
                # Preserve the text prior to colon and replace every following number with zero
                prefix = line.split(':', 1)[0].strip()
                tokens = line.split(':', 1)[1].split()
                new_line = f"{prefix}: " + "    ".join(zero_value for _ in tokens) + "\n"
                
            # Case 4: Lines that consist solely of unnamed numbers separated by spaces
            elif re.match(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)*', line):
                tokens = re.split(r'\s+', line.strip())
                new_line = "    ".join(zero_value for _ in tokens) + "\n"
                
            # Append the changed line to the new_content list
            new_content.append(new_line)
            
        # Write the modified content to a new output file.
        with open(file_path, 'w') as f:
            f.writelines(new_content)


    def HandleNonConvergence(self,
                             ) -> None:
        """
        Average over the last self.SAMPLE_SIZE iterations to determine flowfield variables.
        This method performs the following steps:
            - Creates a subfolder to store output files if it doesn't already exist.
            - Deletes any existing forces output file for the current analysis.
            - Executes a specified number of iterations and generates solver outputs for each.
            - Averages the data from the generated files to estimate the true values.
            - Deletes the individual output files after averaging.

        Returns
        -------
        None
        """

        # Create subfolder to put all output files into if the folder doesn't already exist
        dump_folder = Path("MTSOL_output_files")
        os.makedirs(dump_folder, 
                    exist_ok=True)

        # Delete the forces.analysisname file if it exists already
        if os.path.exists(f"forces.{self.analysis_name}"):
            os.remove(f"forces.{self.analysis_name}")

        # Initialize iteration counter
        iter_counter = 0

        # Initialize watchdog to check when output file has been created
        event_handler = FileCreatedHandling(f'forces.{self.analysis_name}',
                                            dump_folder / f'forces.{self.analysis_name}.{iter_counter}')

        observer = Observer()
        observer.schedule(event_handler,
                          path=os.getcwd(),
                          recursive=False,
                          )   
        observer.start()
    
        # Keep looping until iter_count exceeds the target value for number of iterations to average 
        while iter_counter <= self.SAMPLE_SIZE:
            #Execute iteration
            self.StdinWrite("x 1")

            # Wait for current iteration to complete
            self.WaitForCompletion(type=1)

            # Generate solver outputs
            self.GenerateSolverOutput()

            # Rename file to indicate the iteration number, and avoid overwriting the same file. 
            # Also move the file to the output folder
            # Waits for the file to exist before copying.
            init_time = time.time()
            timer = 0
            while not event_handler.is_file_processed() and timer < 10:
                current_time = time.time()
                timer = current_time - init_time
                time.sleep(0.1)
            
            # Increase iteration counter by step size
            iter_counter += 1

            # Re-intialise the event handler for the next iteration with an updated destination
            event_handler = FileCreatedHandling(f'forces.{self.analysis_name}',
                                                dump_folder / f'forces.{self.analysis_name}.{iter_counter}')
            observer.unschedule_all()
            observer.schedule(event_handler,
                              path=os.getcwd(),
                              recursive=False)            

        # Wrap up the watchdog
        observer.stop()
        observer.join()

        # Average the data from all the iterations to obtain the assumed true values. This effectively assumes that the iterations are oscillating about the true value.
        # This is a simplification, but it is the best we can do in this case.
        # The average is calculated by summing all the values and dividing by the number of iterations.
        self.GetAverageValues()


    def HandleExitFlag(self,
                       exit_flag: int,
                       type : int,
                       ) -> None:
        """
        Handle the exit flag of the solver execution. 

        Parameters
        ----------
        - exit_flag : int
            Exit flag indicating the status of the solver execution.
        - type : int
            A status integer indicating the type of solve being run:
                0: Inviscid
                1: Viscous CB
                2: 1 + Viscous outer duct
                3: Complete viscous
        
        Returns
        -------
        None
        """

        #If solver does not converge, call the non-convergence handler function.
        if exit_flag == ExitFlag.NON_CONVERGENCE.value:
            self.WriteStateFile()
            self.HandleNonConvergence()

        # Else if the solver has crashed during the inviscid solve after multiple iterations, set all force outputs to zero. This would mean an infeasible design. 
        elif exit_flag == ExitFlag.CRASH.value and type == 0:            
            # Restart MTSOL and create output file without running any iterations
            self.GenerateProcess()
            self.SetOperConditions()
            self.GenerateSolverOutput()

            # Open the forces output file and set all outputs to zero
            self.GenerateCrashOutputs()
        
        # Else if the solver has crashed during one of the viscous solves, use the last saved output state.
        elif exit_flag == ExitFlag.CRASH.value and type != 0:
            return

        # Else if the solver has finished, update the statefile.   
        elif exit_flag in (ExitFlag.COMPLETED.value, ExitFlag.SUCCESS.value, ExitFlag.NOT_PERFORMED.value, ExitFlag.CHOKING.value):
            self.WriteStateFile()
            return
          
        else:
            raise OSError(f"Unknown exit flag {exit_flag} encountered!") from None


    def TryExecuteViscousSolver(self,
                         surface_ID: int = None,
                         ) -> tuple[int, int]:
        """
        Try to execute the MTSOL solver for the current analysis on the viscous surface surface_ID.

        Parameters
        ----------
        - surface_ID : int, optional
            ID of the surface which is to be toggled. For a ducted fan, the ID should be either 1, 3, or 4. 
        
        Returns
        -------
        - exit_flag : int
            Exit flag indicating the status of the solver execution.
        - iter_counter : int
            Number of iterations performed up until failure of the solver.
        """
        
        try:
            # Restart MTSOL - this is required since MTSOL quits upon a solver crash, so we need to restart the subprocress. 
            self.GenerateProcess()

            # Set viscous if surface_ID is given
            if surface_ID is not None:
                self.SetViscous(surface_ID)

            # Execute the solver and get the exit flag and iteration count
            exit_flag, iter_count = self.ExecuteSolver()
        except (OSError, BrokenPipeError):
            # If the solver crashes, set the exit flag to crash
            exit_flag = ExitFlag.CRASH.value
            iter_count = self.iter_counter

        return exit_flag, iter_count    
    
    
    def ConvergeIndividualSurfaces(self,
                                   ) -> tuple[int, int]:
        """
        Should a complete viscous analysis fail and cause an MTSOL crash, a partial run, where each axisymmetric surface is toggled individually, 
        may sometimes improve performance and yield (partially) converged results.

        This function executes a consecutive partial run, where the centerbody, outer duct, and inner duct are converged in sequence.
        Note that this function requires MTSOL to be closed/in the crashed state. 
        
        Returns
        -------
        - total_exit_flag : int
            Exit flag indicating the overall status of the convergence. 
        - total_iter_count : int
            Count of the total number of iterations performed across the different analyses. 
        """

       # Define initial exit flags and iteration counters
        # Note that a negative iteration count indicates that the solver did not run
        exit_flag_visc_CB = ExitFlag.NOT_PERFORMED.value
        iter_count_visc_CB = 0
        exit_flag_visc_induct = ExitFlag.NOT_PERFORMED.value
        iter_count_visc_induct = 0
        exit_flag_visc_outduct = ExitFlag.NOT_PERFORMED.value
        iter_count_visc_outduct = 0
        total_exit_flag = ExitFlag.NOT_PERFORMED.value

        # Initialize a list to keep track of any surfaces which may fail convergence
        failed_surfaces = []

        # Execute the initial viscous solve, where we only solve for the boundary layer on the centerbody
        exit_flag_visc_CB, iter_count_visc_CB = self.TryExecuteViscousSolver(surface_ID=1)
        if exit_flag_visc_CB in (ExitFlag.SUCCESS.value, ExitFlag.COMPLETED.value):
            # If the viscous CB solve was successful, update the statefile. 
            self.WriteStateFile()
        elif exit_flag_visc_CB in (ExitFlag.CRASH.value, ExitFlag.NON_CONVERGENCE.value):
            # if the viscous CB solve caused a crash or doesn't converge, write it to the failed list for a later retry. 
            failed_surfaces.append(1)

        # Execute the viscous solve for the outside of the duct
        exit_flag_visc_outduct, iter_count_visc_outduct = self.TryExecuteViscousSolver(surface_ID=3)
        if exit_flag_visc_outduct in (ExitFlag.SUCCESS.value, ExitFlag.COMPLETED.value):
            # If the viscous solve was successful, update the statefile.
            self.WriteStateFile()
        elif exit_flag_visc_outduct in (ExitFlag.CRASH.value, ExitFlag.NON_CONVERGENCE.value):
            # if the viscous solve caused a crash or doesn't converge, write it to the failed list for a later retry. 
            failed_surfaces.append(3)
        
        # Execute the final viscous solve for the inside of the duct
        exit_flag_visc_induct, iter_count_visc_induct = self.TryExecuteViscousSolver(surface_ID=4)
        if exit_flag_visc_induct in (ExitFlag.SUCCESS.value, ExitFlag.COMPLETED.value):
            # If the viscous solve was successful, update the statefile.
            self.WriteStateFile()
        elif exit_flag_visc_induct in (ExitFlag.CRASH.value, ExitFlag.NON_CONVERGENCE.value):
            # if the viscous solve caused a crash or doesn't converge, write it to the failed list for a later retry. 
            failed_surfaces.append(4)
        
        retry_flags = {}
        retry_counts = {}
        for surface in failed_surfaces:
            # If the surface failed to converge, we need to toggle it and try again
            # Execute the viscous solve for the failed surface
            exit_flag_visc_retry, iter_count_visc_retry = self.TryExecuteViscousSolver(surface_ID=surface)

            # Check if the viscous solve was successful
            if exit_flag_visc_retry in (ExitFlag.SUCCESS.value, ExitFlag.COMPLETED.value):
                # If the viscous solve was successful, update the statefile.
                self.WriteStateFile()
            retry_flags[surface] = exit_flag_visc_retry
            retry_counts[surface] = iter_count_visc_retry
        
        # Compute total iteration count of the retry attempt
        retry_count = sum(retry_counts.values())
    
        # Compute the overall exit flag and total iteration count
        exit_flag_visc_CB = retry_flags.get(1, exit_flag_visc_CB)
        exit_flag_visc_outduct = retry_flags.get(3, exit_flag_visc_outduct)
        exit_flag_visc_induct = retry_flags.get(4, exit_flag_visc_induct)

        total_exit_flag = max(exit_flag_visc_CB, exit_flag_visc_outduct, exit_flag_visc_induct)
        total_iter_count = iter_count_visc_CB + iter_count_visc_outduct + iter_count_visc_induct + retry_count

        return total_exit_flag, total_iter_count
    

    def caller(self,
               run_viscous: bool = False,
               generate_output: bool = False,
               output_type: int = 0,
               ) -> tuple[int, int]:
        """
        Main execution interface of MTSOL.

        All executions of the MTSOL program are wrapped in try... except... finally... blocks to handle crashes of the solver

        Parameters
        ----------
        - run_viscous : bool, optional
            Flag to indicate whether to run a viscous solve. Default is False.
        - generate_output : bool, optional
            Flag to determine if MTFLOW outputs (forces, flowfield, boundary layer) should be generated. 
        - output_type : int, optional
            A control integer to determine which output files to generate. 0 corresponds to only the forces file, while any other integer generates all files. 

        Returns
        -------
        - total_exit_flag : int
            Exit flag indicating the status of the solver execution. Is equal to the maximum value of the inviscid and viscous exit flags, since exit_flag > -1 indicate failed/nonconverging solves.
            This is used as a one-variable status indicator, while the corresponding output list gives more details. 
        - total_iter_count : int
            An integer summation of the inviscid and viscous iteration counters. 
        """

        # Define initial exit flags and iteration counters
        # Note that a negative iteration count indicates that the solver did not run
        exit_flag_invisc = ExitFlag.NOT_PERFORMED.value
        iter_count_invisc = 0
        exit_flag_visc = ExitFlag.NOT_PERFORMED.value
        iter_count_visc = 0

        # Generate MTSOL subprocess
        self.GenerateProcess()

        # Write operating conditions
        self.SetOperConditions()

        # Execute inviscid solve
        try:  
            exit_flag_invisc, iter_count_invisc = self.ExecuteSolver()
        except (OSError, BrokenPipeError):
            # If the inviscid solve crashes, we need to set the exit flag to crash
            exit_flag_invisc = ExitFlag.CRASH.value
            iter_count_invisc = self.iter_counter

        finally:
            # Handle solver based on exit flag
            self.HandleExitFlag(exit_flag_invisc,
                                type=0)
            total_exit_flag = exit_flag_invisc
            total_iter_count = iter_count_invisc

        # Only run a viscous solve if required by the user
        # Theoretically there is the chance a viscous run may be started on a non-converged inviscid solve. 
        # This is acceptable, as we assume a steady state residual case has formed at the end of the inviscid case. 
        # There is a probability that by then running a viscous case, convergence to the correct solution may still be obtained.
        if run_viscous and exit_flag_invisc != ExitFlag.CRASH.value:
            # Toggle viscous on the centerbody and the inner and outer duct surfaces
            self.ToggleViscous()
            self.SetViscous(3)
            self.SetViscous(4)
            
            # First we try to run a complete viscous case. Only if this doesn't work and causes a crash do we try to converge each surface individually
            try:
                exit_flag_visc, iter_count_visc = self.ExecuteSolver()

                # Update the statefile
                self.WriteStateFile()
                
                total_exit_flag = max(exit_flag_visc, exit_flag_invisc)
                total_iter_count = iter_count_invisc + iter_count_visc

            except (OSError, BrokenPipeError):
                # If the complete viscous solve crashed, restart MTSOL and try to converge the individual surfaces separately
                total_exit_flag, individual_iter_count = self.ConvergeIndividualSurfaces()                
                total_iter_count += individual_iter_count
            
            finally:
                # Handle the exit flag
                self.HandleExitFlag(exit_flag_visc, 
                                    type=3)

        if generate_output:
            # Generate the solver output
            self.GenerateSolverOutput(output_type)
        
        # Close the MTSOL tool
        # If no output is generated, need to write an additional white line to close MTSOL
        self.StdinWrite("Q")
        if not generate_output:
            self.process.stdin.write("\n")
        self.process.stdin.flush()

        # Check that MTSOL has closed successfully 
        if self.process.poll() is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                raise OSError("MTSOL did not close after completion.") from None
              
        return total_exit_flag, total_iter_count


if __name__ == "__main__": 
    import time
    
    analysisName = "test_case"
    oper = {"Inlet_Mach": 0.2000,
            "Inlet_Reynolds": 5.000E6,
            "N_crit": 9,
            }

    start_time = time.time()
    test = MTSOL_call(oper, analysisName).caller(run_viscous=True,
                                                 generate_output=True)
    end_time = time.time()

    print(f"Execution of MTSOL_call took {end_time -  start_time} seconds")