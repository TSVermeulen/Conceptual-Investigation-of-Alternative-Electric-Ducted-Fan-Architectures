"""
MTFLO_call
=============

Description
-----------
This module provides an interface to interact with the MTFLO executable from Python. 
It creates a subprocess for the MTFLO executable, loads in the input file tflow.xxx, 
and writes the output to the tdat.xxx data file for use within MTSOL. 

Classes
-------
MTFLO_call
    A class to handle the interface between Python and the MTFLO executable.

Examples
--------
>>> filepath = r"mtflo.exe"
>>> analysisName = "test_case"
>>> test = MTFLO_call(filepath, analysisName)
>>> test.caller()

Notes
-----
This module is designed to work with the MTFLO executable. Ensure that the executable and the input file, tflow.xxx, 
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
Version: 1.0

Changelog:
- V1.0: Initial working version
"""

import subprocess
import os

class MTFLO_call():
    """
    Class to handle the interface between Python and the MTFLO executable.
    """

    def __init__(self, *args,
                 ) -> None:
        """
        Initialize the MTFLO_call class with the file path and analysis name.

        Parameters
        ----------
        file_path : str
            The path to the MTFLO executable.
        analysis_name : str
            The name of the analysis case.
        """

        file_path, analysis_name = args
        self.fpath = file_path
        self.analysis_name = analysis_name


    def GenerateProcess(self,
                        ) -> None:
        """
        Create MTFLO subprocess

        Requires that the executable, mtflo.exe, and the input file, tflow.xxx are present in the same directory as this
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
        if self.process.poll() != None:
            raise ImportError(f"The MTFLO program is not found in {self.fpath}")
        
    
    def LoadForcingField(self,
                         ) -> int:
        """
        Loads the tflow.xxx input file into MTFLO and checks it has been correctly processed

        Returns
        -------
        self.process.returncode : int
            The returncode of the subprocess upon completion. 
        """

        # Enter field parameter menu
        self.process.stdin.write("F \n")

        # Read parameter text file
        self.process.stdin.write("R \n")

        # Accept default filename
        self.process.stdin.write("\n")
        self.process.stdin.flush()

        # Check if file is loaded in successfully.
        # If error occured, MTFLO will have crashed, so we can check success by checking 
        # if the subprocess is still alive
        if self.process.poll() != None:
            raise ImportError(f"There was an issue with the input file {"tflow." + self.analysis_name}, which caused MTFLO to crash")
        
        # Exit the field parameter menu
        self.process.stdin.write("\n")

        # Write to the flowfield file tdat.xxx and check if writing was successful 
        self.process.stdin.write("W \n")
        self.process.stdin.flush()     

        if self.process.poll() != None:
            raise ImportError(f"There was an issue writing the field parameters to the file {"tdat." + self.analysis_name}, which caused MTFLO to crash")
        
        # Close the MTFLO program
        self.process.stdin.write("Q \n")
        self.process.stdin.flush()

         # Check that MTFLO has closed successfully 
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                raise OSError("Something went wrong in the MTSET call. \
                            MTSET was not closed following end of file generation. \
                            Run terminated.")
        else:    
            return self.process.returncode


    def caller(self
               ) -> int:
        """
        Full interfacing function between Python and MTFLO

        Requires that the input file, tflow.xxx, has been made and is available together with the mtflo.exe executable in the local directory.
        """
        
        # Create subprocess for the MTFLO tool
        self.GenerateProcess()  

        # Load the numerical grid
        self.LoadForcingField()

        return self.process.returncode       

if __name__ == "__main__":

    import time
    start_time = time.time()
    filepath = r"mtflo.exe"
    analysisName = "test_case"
    test = MTFLO_call(filepath, analysisName)
    test = test.caller()
    end_time = time.time()

    print(f"Execution of MTFLO_call({filepath, analysisName}).caller() took {end_time - start_time} seconds")