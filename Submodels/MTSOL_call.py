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
from typing import Any

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

        #Initialize reference quantities
        self.RSTRO = 1
        self.GAMMA = 1.4
        self.HSTRO = 1 / (self.GAMMA - 1)
        self.LREF = 1  # Use a reference length of 1 m
        self.ASTRO = 1
        self.PSTRO = 1 / self.GAMMA


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

        # Write inlet Mach number
        self.process.stdin.write(f"M {self.operating_conditions["Inlet_Mach"]} \n")

        # Set critical amplification factor to N=9 rather than the default N=7
        self.process.stdin.write("N 9\n")

        # Set the Reynolds number, calculated using the length self.LREF = 1!
        # Flush is required here to ensure console output is up-to-date before collecting it. 
        self.process.stdin.write(f"R {self.operating_conditions["Inlet_Reynolds"]} \n")
        self.process.stdin.flush()
       
        # Disable all viscous toggles to ensure inviscid analysis is run initially
        # To do this, we need to check what elements are present This is done by checking the console output of the menu and identifying the indices of all 'Tx' rows
        # Collect console output from MTSOL, stopping when the end of the menu is reached
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
        # Element surface numbers are stored to the toggles list, which is written to self to enable easy access later on when re-enabling the viscous toggles
        toggles = []
        for i in range(n_elements):
            self.process.stdin.write(f"V {interface_output[idx_first_element + i][2]}")
            toggles.append(int(interface_output[idx_first_element + i][2]))
        self.element_counts = toggles
        
        # Exit the Modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()

    

    def ToggleViscous(self,
                      elements: list[int]|int,
                      ) -> None:
        """
        Toggle the viscous settings for all elements.

        Parameters
        ----------
        elements : list[int] | int
            An integer or list of integers representing the elements for which the viscous settings need to be toggled.


        Returns
        -------
        None
        """

        # Input Validation
        if not all(map(lambda v: v in self.element_counts, elements)):
            raise OSError(f"element is not in the element counted in the solution parameters menu!") from None

        # Enter the Modify solution parameters menu
        self.process.stdin.write("m \n")

        # Toggle the viscous settings for each element in elements
        self.process.stdin.write(f"V {','.join(map(str, elements))}")

        # Exit the Modify solution parameters menu
        self.process.stdin.write("\n")
        self.process.stdin.flush()


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

        self.ToggleViscous([1, 3])
        input()

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