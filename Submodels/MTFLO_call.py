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
>>> analysisName = "test_case"
>>> test = MTFLO_call(analysisName)
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
Version: 1.0.5

Changelog:
- V1.0: Initial working version
- V1.0.5: Cleaned up inputs, removing file_path and changing it to a constant. 
- V1.1: Added file status check to ensure that the file has been written before proceeding. Added stdinwrite function to clean up process interactions.
"""

import subprocess
import os
import time

class MTFLO_call:
    """
    Class to handle the interface between Python and the MTFLO executable.
    """

    def __init__(self, 
                 analysis_name: str,
                 ) -> None:
        """
        Initialize the MTFLO_call class with the file path and analysis name.

        Parameters
        ----------
        - analysis_name : str
            The name of the analysis case.
        """

        self.analysis_name = analysis_name

        # Define filepath of MTFLO as being in the same folder as this Python file
        self.process_path: str = os.getenv('MTFLO_PATH', 'mtflo.exe')
        if not os.path.exists(self.process_path):
            raise FileNotFoundError(f"MTFLO executable not found at {self.process_path}")
        
        
    def StdinWrite(self,
                   command: str) -> None:
        """
        Simple function to write commands to the subprocess stdin in order to pass commands to MTFLO.

        Parameters
        ----------
        - command : str
            The text-based command to pass to MTFLO.

        Returns
        -------
        None
        """

        self.process.stdin.write(f"{command} \n")
        self.process.stdin.flush()


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
        self.process = subprocess.Popen([self.process_path, self.analysis_name], 
                                        stdin=subprocess.PIPE, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        shell=True, 
                                        text=True,
                                        bufsize=1,
                                        )
        
        # Check if subprocess is started successfully
        if self.process.poll() is not None:
            raise ImportError(f"MTFLO or tdat.{self.analysis_name} not found in {self.fpath}") from None
        
    
    def LoadForcingField(self,
                         ) -> None:
        """
        Loads the tflow.xxx input file into MTFLO and checks it has been correctly processed
        """

        # Enter field parameter menu
        self.StdinWrite("F")

        # Read parameter text file
        self.StdinWrite("R")

        # Accept default filename
        self.StdinWrite("")

        # Check if file is loaded in successfully.
        # If error occured, MTFLO will have crashed, so we can check success by checking 
        # if the subprocess is still alive
        if self.process.poll() is not None:
            raise ImportError(f"Issue with input file tflow.{self.analysis_name}, MTFLO crashed") from None
        
        # Exit the field parameter menu
        self.StdinWrite("")

        # Write to the flowfield file tdat.xxx and check if writing was successful 
        self.StdinWrite("W")

        if self.process.poll() is not None:
            raise OSError(f"Issue writing parameters to tdat.{self.analysis_name}, MTFLO crashed") from None
        
        # Close the MTFLO program
        self.StdinWrite("Q")

         # Check that MTFLO has closed successfully 
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            
            except subprocess.TimeoutExpired:
                self.process.kill()
                raise OSError("MTFLO did not close after file generation. Process was killed.") from None

    def FileStatus(self,
                   fpath: str,
                   ) -> bool:
        """ 
        Simple function to check if the file update/write has finished
        """
        try:
            with open(fpath, "a"):
                return True
        except IOError:
            return False


    def caller(self,
               ) -> int:
        """
        Full interfacing function between Python and MTFLO

        Requires that the input file, tflow.xxx, has been made and is available together with the mtflo.exe executable in the local directory.
        
        Returns
        -------
        - self.process.returncode : int
            self.process.returncode
        """
        
        # Create subprocess for the MTFLO tool
        self.GenerateProcess()  

        # Load the numerical grid
        self.LoadForcingField()

        # Wait until file has been processed
        fpath = f"tdat.{self.analysis_name}"
        while not self.FileStatus(fpath):
            time.sleep(0.01)

        return self.process.returncode      


if __name__ == "__main__":
    start_time = time.time()
    analysisName = "test_case"
    test = MTFLO_call(analysisName)
    test = test.caller()
    end_time = time.time()

    print(f"Execution of MTFLO_call({analysisName}).caller() took {end_time - start_time} seconds")