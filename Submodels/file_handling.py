"""

Class definitions for all file handling related actions required by the MTFLOW codes MTSET, MTSOL, and MTFLO.

@author: T.S.Vermeulen
@email: thomas0708.vermeulen@gmail.com / T.S.Vermeulen@student.tudelft.nl
@version: 0.0.1

Changelog:
- V0.0.1: Empty file generation with dummy placeholder class

"""

class AxiWallFileGenerator():
    """
    This class contains the required functionalities to generate the walls.xxx file which is loaded in MTSET. 
    """


    def __init__(self, *args):
        identifier, analysis_name, element_count = args

        self.identifier = identifier  # Identifier of the population member for which the walls.xxx file is being generated
        self.fileExtension
        pass

    
    def fileWriter(self):

        return 