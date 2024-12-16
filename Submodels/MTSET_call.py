"""

"""

import subprocess

class MTSET_call():


    def __init__(self, *args):
        filePath, analysisName = args
        self.fPath = filePath
        self.analysisName = analysisName

        return


    def GenerateProcess(self, dummy):
        """
        -----
        Create MTSET subprocess
        -----

        Simple function to create an MTSET subprocress using the defined
        executable path and analysis name in the class initialization. 
        The defined subprocess has the inputs, outputs, and error file 
        handles sent to PIPE for direct interaction within the Python code.  
        """

        self.process = subprocess.Popen([self.fPath, self.analysisName], 
                                 stdin=subprocess.PIPE, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True
                                 )
        return
    

    def GridGenerator(self, dummy):
        """
        -----
        Automatic grid generator and grid refinement
        -----


        """

        # There are axisymmetric bodies - the center body and the duct. 
        # Both need to be loaded in the startup of MTSET

        # -----
        # TO DO 
        # include handling of updated spacing; detect number of bodies and handle appropriately
        # get current number of streamwise gridpoints
        # -----

        self.process.stdin.write("\n \n") 
        self.process.stdin.flush()  # Send return commands to MTSET

        # Exit grid spacing definition routine and modify grid parameters to
        # increase the number of streamlines and streamwise gridpoints
        self.process.stdin.write("\n")
        self.process.stdin.flush()

        pass
    

    def GridSmoothing(self, dummy):
        """
        -----
        Elliptic grid smoothing function
        -----
        
        Performs elliptic grid smoothing on the created grid until converged. 
        Convergence is measured by checking the last pass in the smoothing process 
        for the presence of Dmax in the terminal output. 
        If Dmax is no longer present within the grid, set smoothing to false 
        and exit the routine.
        """

        # Define controlling booleans for the smoothing process
        smoothing = True
        get_console_out = True

        # Control smoothing process, including detection when further smoothing is no longer needed
        while smoothing:
            self.process.stdin.write("e\n")  # Execute elliptic smoothing continue command
            self.process.stdin.flush()  # Send command to MTSET
            
            # Collect console output from MTSET, stopping when the end of the menu is reached
            interface_output = []
            while get_console_out:
                next_line = self.process.stdout.readline()  # Collect output and add to list
                interface_output.append(next_line)
                
                if next_line == "" and self.process.poll() is not None:  #Handle (unexpected) quitting of program
                    break
                if next_line == '   Q uit\n':  # Stop collecting once end of MTSET menu is reached
                    break
            
            # Find the index of 'Pass 10' (i.e. the final pass of the smoothing process). Checks from the back of the list
            pass_10_index = next((i for i, s in reversed(list(enumerate(interface_output))) if 'Pass          10' in s), -1)

            if "Dmax" not in interface_output[pass_10_index + 1]: #Check the row following Pass 10 to see if smoothing is converged
                break
        return

    def fileGenerator(self, dummy):
        """
        -----
        Generation of required files and outputs from MTSET to use in 
        further analyses
        -----

        This is a very simple function to handle the end-of-procedure steps in MTSET, generating the output files.
        These output files can then be used in later analyses. Two files are generated:
        mtgpar.xxx and tdat.xxx, where xxx is the analysis name

        mtgpar contains the grid parameters of the created grid, while tdat is a (binary) solution storage file, 
        containing an incompressible, inviscid flow solution as starting point for the MTFLO field parameter 
        specification or MTSOL solver. Both files are created using standard, built-in functions within MTSET.

        Function returns the exit code of the subprocess. 
        """

        # Create mtgpar.xxx file
        self.process.stdin.write("s\n")

        # Create tdat.xxx file
        # Note that MTSET automatically closes after writing the tdat file!
        self.process.stdin.write("w\n")
        self.process.stdin.flush()  # Flush inputs to make sure they are passed to MTSET immediately
        return

    def test(self):
        """
        
        """
        
        self.GenerateProcess(self)  # Create subprocess for the MTSET tool

        self.GridGenerator(self)

        self.GridSmoothing(self)  # Perform elliptical grid smoothing

        self.fileGenerator(self)  # Generate files           

        # Check that MTSET has closed successfully 
        if self.process.poll() == None:
            self.process.kill()
            raise OSError("Something went wrong in the MTSET call. \
                          MTSET was not closed following end of file generation. \
                          Run terminated.")
        else:    
            return self.process.returncode

if __name__ == "__main__":
    caller = MTSET_call(r"mtset.exe", "dprop")

    caller.test()
    