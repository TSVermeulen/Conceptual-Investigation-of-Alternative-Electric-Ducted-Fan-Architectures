"""
config
======

Description
-----------
This module defines the configuration settings and parameters for the optimization problem, including aerodynamic 
analysis, design variables, and constraints. It integrates with the MTFLOW executable for aerodynamic analysis.

Notes
-----
Ensure that the MTFLOW executable and required input files are present in the appropriate directories. This module 
provides the necessary settings for optimization and analysis.

References
----------
For more details on the MTFLOW solver and its input/output requirements, refer to the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial implementation. 
"""

import numpy as np
from ambiance import Atmosphere
from contextlib import contextmanager
from enum import IntEnum, auto
import os
import sys
from pathlib import Path

# Add the parent and submodels paths to the system path if they are not already in the path
parent_path = str(Path(__file__).resolve().parent.parent)
submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

if parent_path not in sys.path:
    sys.path.append(parent_path)

if submodels_path not in sys.path:
    sys.path.append(submodels_path)


# Define a context manager for GenerateMTFLOBlading to ensure the working directory is set correctly
@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

# Define the altitude for the analysis and construct an atmosphere object from which atmospheric properties can be extracted. 
ALTITUDE = 0  # meters
atmosphere = Atmosphere(ALTITUDE)

# Define the objective IDs used to construct the objective functions
class ObjectiveID(IntEnum):
    """
    Enumeration of objective function identifiers for the optimization problem.
    Each member represents a different optimization objective that can be used in the genetic algorithm's fitness evaluation.
    """
    def _generate_next_value_(name, start, count, last_values):
        return count  # This makes the first member 0 rather than the default 1.

    EFFICIENCY = auto()
    WEIGHT = auto()
    FRONTAL_AREA = auto()
    PRESSURE_RATIO = auto()
    MULTIPOINT_TO_CRUISE = auto()
    CENTERBODY_TRANSITION_LOCATION = auto()
    DUCT_INNER_TRANSITION_LOCATION = auto()
    DUCT_OUTER_TRANSITION_LOCATION = auto()
    DUCT_THRUST_CONTRIBUTION = auto()
    CENTERBODY_THRUST_CONTRIBUTION = auto()
     
objective_IDs = [ObjectiveID.EFFICIENCY]


# Define the operating conditions dictionary
oper = {"Inlet_Mach": 0.10285224,
        "N_crit": 9,
        "RPS": 25.237,
        "Omega": -9.667
        }
oper["Vinl"] = atmosphere.speed_of_sound[0] * oper["Inlet_Mach"]

# Controls for the optimisation vector - CENTERBODY
OPTIMIZE_CENTERBODY = False  # Control boolean to determine if centerbody should be optimised. If false, code uses the default entry below.
CENTERBODY_VALUES = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0., 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0., 'y_c': 0., 'z_TE': 0., 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': 0.27689037361278407, 'trailing_camberline_angle': 0., 'leading_edge_direction': 0., "Chord Length": 1.5, "Leading Edge Coordinates": (0., 0.)}


# Controls for the optimisation vector - DUCT
OPTIMIZE_DUCT = True
DUCT_VALUES = {'b_0': 0., 'b_2': 0., 'b_8': 0.004081758291374328, 'b_15': 0.735, 'b_17': 0.8, 'x_t': 0.2691129541223092, 'y_t': 0.084601317961794, 'x_c': 0.5, 'y_c': 0., 'z_TE': -0.015685, 'dz_TE': 0.0005638524603968335, 'r_LE': -0.06953901280141099, 'trailing_wedge_angle': 0.16670974950670672, 'trailing_camberline_angle': 0.003666809042006104, 'leading_edge_direction': -0.811232599724247, 'Chord Length': 1.2446, "Leading Edge Coordinates": (0.093, 1.20968)}


# Controls for the optimisation vector - BLADES
OPTIMIZE_STAGE = [True, False, False]
ROTATING = [True, False, False]
NUM_RADIALSECTIONS = [5, 2, 2]  # Define the number of radial sections at which the blade profiles for each stage will be defined. 
NUM_STAGES = 3  # Define the number of stages (i.e. total count of rotors + stators)
REFERENCE_BLADE_ANGLES = [np.deg2rad(19), 0, 0]  # Reference angles at the reference section (typically 75% of blade span)
BLADE_DIAMETERS = [2.1336, 2.2098, 2.2098]
tipGap = 0.01016  # 1.016 cm tip gap


def GenerateMTFLOBlading(Omega: float,                        
                         ref_blade_angle: float,
                         ) -> tuple[list, list]:
    """
    Generate MTFLO blading.
    The blading parameters are based on Figure 3 in [1].

    Parameters
    ----------
    - Omega : float
        The non-dimensional rotational speed of the rotor, as defined in the MTFLOW documentation in units of Vinl/Lref
    - ref_blade_angle : float
        The blade set angle, in radians. 
    
    Returns
    -------
    - (list, list):
        A tuple of two lists:
        - blading_parameters : list
            A list containing dictionaries with the blading parameters.
        - design_parameters : list
            A list containing dictionaries with the design parameters for each radial station.
    """

    from Submodels.Parameterizations import AirfoilParameterization

    # Start defining the MTFLO blading inputs
    radial_stations = np.array([0., 0.3, 0.5, 0.7, 1]).astype(float) * BLADE_DIAMETERS[0] / 2
    chord_length = np.array([0.3510, 0.3152, 0.2528, 0.2367, 0.2205]).astype(float)
    blade_angle = np.array([np.deg2rad(53.6), np.deg2rad(46.8), np.deg2rad(32.3), np.deg2rad(22.3), np.deg2rad(15.5)]).astype(float)
    propeller_parameters = {"root_LE_coordinate": 0.1495672948767407, 
                            "rotational_rate": Omega, 
                            "ref_blade_angle": ref_blade_angle, 
                            "reference_section_blade_angle": np.deg2rad(20), 
                            "blade_count": 3, 
                            "radial_stations": radial_stations, 
                            "chord_length": chord_length, 
                            "blade_angle": blade_angle}
    
    horizontal_strut_parameters = {"root_LE_coordinate": 0.57785, 
                                   "rotational_rate": 0, 
                                   "ref_blade_angle": 0, 
                                   "reference_section_blade_angle": 0, 
                                   "blade_count": 4, 
                                   "radial_stations": np.array([0., 1]) * 1.15, 
                                   "chord_length": np.array([0.57658, 0.14224]), 
                                   "blade_angle": np.array([np.deg2rad(90), np.deg2rad(90)]),
                                   "sweep_angle": np.array([0, 0])}
    
    diagonal_strut_parameters = {"root_LE_coordinate": 0.577723, 
                                 "rotational_rate": 0, 
                                 "ref_blade_angle": 0, 
                                 "reference_section_blade_angle": 0, 
                                 "blade_count": 2, 
                                 "radial_stations": np.array([0., 1]) * 1.15, 
                                 "chord_length": np.array([0.10287, 0.10287]), 
                                 "blade_angle": np.array([np.deg2rad(90), np.deg2rad(90)]),
                                 "sweep_angle": np.array([0, 0])}
    
    # Construct blading list
    blading_parameters = [propeller_parameters,
                          horizontal_strut_parameters,
                          diagonal_strut_parameters]

    # Define the sweep angles
    # Note that this is approximate, since the rotation of the chord line is not completely accurate when rotating a complete profile
    sweep_angle = np.zeros_like(blading_parameters[0]["chord_length"])
    root_blade_angle = (np.deg2rad(53.6) + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0]["reference_section_blade_angle"])
    root_rotation_angle = np.pi / 2 - root_blade_angle

    root_LE = blading_parameters[0]["root_LE_coordinate"] # The location of the root LE is arbitrary for computing the sweep angles.
    root_mid_chord = root_LE + (0.3510 / 2) * np.cos(root_rotation_angle)
    for i in range(len(blading_parameters[0]["chord_length"])):
        blade_angle = (blading_parameters[0]["blade_angle"][i] + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0]["reference_section_blade_angle"])
        rotation_angle = np.pi / 2 - blade_angle

        # Compute sweep such that the midchord line is constant.
        local_LE = root_mid_chord - (blading_parameters[0]["chord_length"][i] / 2) * np.cos(rotation_angle)
        if blading_parameters[0]["radial_stations"][i] != 0:
            sweep_angle[i] = np.atan((local_LE - root_LE) / (blading_parameters[0]["radial_stations"][i]))
        else:
            sweep_angle[i] = 0
    blading_parameters[0]["sweep_angle"] = sweep_angle

    # Obtain the parameterizations for the profile sections. 
    local_dir_path = Path('Validation/Profiles')
    R02_fpath = local_dir_path / 'X22_02R.dat'
    R03_fpath = local_dir_path / 'X22_03R.dat'
    R04_fpath = local_dir_path / 'X22_04R.dat'
    R05_fpath = local_dir_path / 'X22_05R.dat'
    R06_fpath = local_dir_path / 'X22_06R.dat'
    R07_fpath = local_dir_path / 'X22_07R.dat'
    R08_fpath = local_dir_path / 'X22_08R.dat'
    R09_fpath = local_dir_path / 'X22_09R.dat'
    R10_fpath = local_dir_path / 'X22_10R.dat'
    Hstrut_fpath = local_dir_path / 'Hstrut.dat'
    Dstrut_fpath = local_dir_path / 'Dstrut.dat'

    filenames = [R02_fpath, R03_fpath, R04_fpath, R05_fpath, R06_fpath, R07_fpath, R08_fpath, R09_fpath, R10_fpath, Hstrut_fpath, Dstrut_fpath]

    # First check if all files are present
    missing_files = [f for f in filenames if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(map(str, missing_files))}")

    # Compute parameterization for the airfoil section at r=0.2R
    # Note that we keep this section constant for r=0.1R and r=0.15R and equal to that of r=0.2R
    R01_section = AirfoilParameterization().FindInitialParameterization(reference_file=R02_fpath,
                                                                        plot=False)
    # Compute parameterization for the airfoil section at r=0.3R
    R03_section = AirfoilParameterization().FindInitialParameterization(reference_file=R03_fpath,
                                                                        plot=False)
    # Compute parameterization for the mid airfoil section
    R05_section = AirfoilParameterization().FindInitialParameterization(reference_file=R05_fpath,
                                                                        plot=False)
    # Compute parameterization for the airfoil section at r=0.7R
    R07_section = AirfoilParameterization().FindInitialParameterization(reference_file=R07_fpath,
                                                                        plot=False)
    # Compute parameterization for the tip airfoil section
    R10_section = AirfoilParameterization().FindInitialParameterization(reference_file=R10_fpath,
                                                                        plot=False)
    # Compute parameterization for the horizontal struts
    Hstrut_section = AirfoilParameterization().FindInitialParameterization(reference_file=Hstrut_fpath,
                                                                           plot=False)
    # Compute parameterization for the diagonal struts
    Dstrut_section = AirfoilParameterization().FindInitialParameterization(reference_file=Dstrut_fpath,
                                                                           plot=False)

    # Construct blading list
    design_parameters = [[R01_section, R03_section, R05_section, R07_section, R10_section],
                         [Hstrut_section, Hstrut_section],
                         [Dstrut_section, Dstrut_section]]

    return blading_parameters, design_parameters

with pushd(parent_path):
        STAGE_BLADING_PARAMETERS, STAGE_DESIGN_VARIABLES = GenerateMTFLOBlading(oper["Omega"],
                                                                                REFERENCE_BLADE_ANGLES[0])

# Define the target thrust/power and efficiency for use in constraints
P_ref_constr = 0.16607 * (0.5 * atmosphere.density[0] * oper["Vinl"] ** 3 * BLADE_DIAMETERS[0] ** 2)  # Reference Power in Watts derived from baseline analysis
T_ref_constr = 0.13288 * (0.5 * atmosphere.density[0] * oper["Vinl"] ** 2 * BLADE_DIAMETERS[0] ** 2) # Reference Thrust in Newtons derived from baseline analysis
Eta_ref_constr = 0.80014  # Reference Propulsive efficiency derived from baseline analysis
deviation_range = 0.01  # +/- x% of the reference value for the constraints

# Define the constraint IDs used to construct the constraint functions
# constraint IDs are structured as a nested list, of the form:
# [[inequality constraint 1, inequality constraint 2, ...],
#  [equality constraint 1, equality constraint 2, ...]]
class InEqConstraintID(IntEnum):
    """
    Enumeration of the inequality constraint identifiers for the optimization problem.
    Each member represents a different inequality constraint that can be used in the genetic algorithm's evaluation.
    """
    def _generate_next_value_(name, start, count, last_values):
        return count  # This makes the first member 0 rather than the default 1.
    
    EFFICIENCY_GTE_ZERO = auto()
    EFFICIENCY_LEQ_ONE = auto()
    MINIMUM_THRUST = auto()
    MAXIMUM_THRUST = auto()
    
class EqConstraintID(IntEnum):
    """
    Enumeration of the equality constraint identifiers for the optimization problem.
    Each member represents a different equality constraint that can be used in the genetic algorithm's evaluation.
    """
    def _generate_next_value_(name, start, count, last_values):
        return count  # This makes the first member 0 rather than the default 1.
    
    CONSTANT_POWER = auto()

constraint_IDs = [[InEqConstraintID.EFFICIENCY_GTE_ZERO, InEqConstraintID.EFFICIENCY_LEQ_ONE, InEqConstraintID.MINIMUM_THRUST, InEqConstraintID.MAXIMUM_THRUST],
                  []]

# Define the population size
POPULATION_SIZE = 50
INIT_POPULATION_SIZE = 20  # Initial population size for the first generation
MAX_GENERATIONS = 20


# Define the initial population parameter spreads, used to construct a biased initial population 
SPREAD_CONTINUOUS = (0.20, 0.20)  # +/- x% of the reference value
SPREAD_DISCRETE = (-3, 6)  # +/- of the reference value