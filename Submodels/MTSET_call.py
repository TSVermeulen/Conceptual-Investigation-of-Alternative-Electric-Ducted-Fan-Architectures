"""

"""

import subprocess

class MTSET_call():
    def __init__(self, *args):

        filePath, analysisName = args
        self.fPath = filePath
        self.analysisName = analysisName


    def generateProcess(self, dummy):
        process = subprocess.Popen([self.fPath, self.analysisName], 
                                 stdin=subprocess.PIPE, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True
                                 )
        return process
    

    def test(self):
        process = self.generateProcess(self)

        # There are axisymmetric bodies - the center body and the duct. 
        # Both need to be loaded in the startup of MTSET
        process.stdin.write("\n \n") 
        process.stdin.flush()  # Send return commands to MTSET
        process.stdin.close()
        print(process.stdout.readlines())
        
        #stdout, stderr = process.communicate(input='\n')

        return None

if __name__ == "__main__":
    caller = MTSET_call("mtset.exe", "dprop")
    print(caller.test())