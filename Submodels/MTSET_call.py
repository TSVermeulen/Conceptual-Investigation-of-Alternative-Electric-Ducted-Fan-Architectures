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
        
        """

        self.process = subprocess.Popen([self.fPath, self.analysisName], 
                                 stdin=subprocess.PIPE, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True
                                 )
        return
    

    def GridRefinement(self, dummy):
        """
        
        """

        pass
    

    def GridSmoothing(self, dummy):
        """
        ----------
        Elliptic Grid Smoothing
        ----------
        Performs elliptic grid smoothing on the created grid until converged. 
        Convergence is measured by checking the last pass in the smoothing process 
        for the presence of Dmax in the terminal output. 
        If Dmax is no longer present within the grid, set smoothing to false 
        and continue.
        ----------
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
        
        """

        return

    def test(self):
        """
        
        """
        
        self.GenerateProcess(self)  # Create subprocess for the MTSET tool

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

        self.GridRefinement(self)  # Refine MTSET grid

        self.GridSmoothing(self)  # Perform elliptical grid smoothing

        self.fileGenerator(self)  # Generate files           


        self.process.stdin.write("s\n")  # Write current grid parameters to mtgpar.analysisName
        
        self.process.stdin.write("w\n")
        
        

        return None

if __name__ == "__main__":
    caller = MTSET_call(r"mtset.exe", "dprop")

    caller.test()
    