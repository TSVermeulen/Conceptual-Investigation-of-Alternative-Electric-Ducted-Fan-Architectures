"""
config
======

This file contains the set of constants and external inputs to the GA optimisation. 


References
==========
The complete set of inputs needed to define the MTFLOW interface is: 
    operating_conditions: dict,
    centrebody_params: dict,
    duct_params: dict,
    blading_parameters: list[dict],
    design_parameters: list[dict],
    ref_length: float,
    analysis_name: str
"""

import numpy as np
from ambiance import Atmosphere
import os
import sys

# Get the parent directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
submodels_path = os.path.join(parent_dir, "Submodels")

# Add the submodels path to the system path
sys.path.append(submodels_path)


# Add the parent folder path to the system path
sys.path.append(parent_dir)

from X22A_validator import GenerateMTFLOBlading


# Define the altitude for the analysis and construct an atmosphere object from which atmospheric properties can be extracted. 
ALTITUDE = 0  # meters
atmosphere = Atmosphere(ALTITUDE)

# Define the objective IDs used to construct the objective functions
# Options for the IDs are:
# - 0: Efficiency
# - 1: Weight
# - 2: Frontal Area
# - 3: Pressure Ratio
# - 4: MultiPointTOCruise
# - 5: Centerbody transition location
# - 6: Duct inner transition location
# - 7: Duct outer transition location
# - 8: Duct thrust contribution
# - 9: Centerbody thrust contribution (i.e. minimize drag)
objective_IDs = [0]


# Define the operating conditions dictionary
oper = {"Inlet_Mach": 0.3,
        "N_crit": 9,
        "RPS": 25.237,
        "Omega": -9.666
        }

# Controls for the optimisation vector - CENTERBODY
OPTIMIZE_CENTERBODY = False  # Control boolean to determine if centerbody should be optimised. If false, code uses the default entry below.
CENTERBODY_VALUES = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0, 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0, 'y_c': 0, 'z_TE': 0, 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': np.float64(0.27689037361278407), 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 4, "Leading Edge Coordinates": (0.3, 0)}


# Controls for the optimisation vector - DUCT
OPTIMIZE_DUCT = False
DUCT_VALUES = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.004081758291374328), 'b_15': np.float64(0.735), 'b_17': np.float64(0.8), 'x_t': np.float64(0.2691129541223092), 'y_t': np.float64(0.084601317961794), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(-0.015685), 'dz_TE': np.float64(0.0005638524603968335), 'r_LE': np.float64(-0.06953901280141099), 'trailing_wedge_angle': np.float64(0.16670974950670672), 'trailing_camberline_angle': np.float64(0.003666809042006104), 'leading_edge_direction': np.float64(-0.811232599724247), 'Chord Length': 1.2446, "Leading Edge Coordinates": (0, 1.20968)}


# Controls for the optimisation vector - BLADES
OPTIMIZE_STAGE = [False, False, False]
NUM_RADIALSECTIONS = 2  # Define the number of radial sections at which the blade profiles for each stage will be defined. 
NUM_STAGES = 3  # Define the number of stages (i.e. total count of rotors + stators)
REFERENCE_SECTION_ANGLES = [np.deg2rad(19), 0, 0]  # Reference angles at the reference section (typically 75% of blade span)
BLADE_DIAMETERS = [2.1294, 2.1294, 2.1294]

blade_section = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815}

os.chdir(parent_dir)
STAGE_BLADING_PARAMETERS, STAGE_DESIGN_VARIABLES = GenerateMTFLOBlading(oper["Omega"],
                                                                        REFERENCE_SECTION_ANGLES[0],
                                                                        plot=False)
os.chdir(current_dir)






