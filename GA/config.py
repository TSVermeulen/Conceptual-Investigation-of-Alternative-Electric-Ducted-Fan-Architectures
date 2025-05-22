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
Version: 1.2

Changelog:
- V1.0: Initial implementation. 
- V1.1: Removed need for context manager by using absolute paths. 
- V1.2: Wrapped blading generation routine in LRU cache to avoid 
        re-running the function at every GA worker import. 
"""

# Import standard libraries
import copy
import functools
from enum import IntEnum, auto
from pathlib import Path

# Import 3rd party libraries
import numpy as np
from ambiance import Atmosphere

# Ensure all paths are correctly setup
from utils import ensure_repo_paths  # type: ignore  # Module not in typeshed or py.typed missing
ensure_repo_paths()

# Import airfoil parameterization class
from Submodels.Parameterizations import AirfoilParameterization # type: ignore

# Define the seed used for randomisation
GLOBAL_SEED = 42

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
    FRONTAL_AREA = auto()
    WETTED_AREA = auto()
    PRESSURE_RATIO = auto()
    MULTIPOINT_TO_CRUISE = auto()
    # CENTERBODY_TRANSITION_LOCATION = auto()
    # DUCT_INNER_TRANSITION_LOCATION = auto()
    # DUCT_OUTER_TRANSITION_LOCATION = auto()
    # DUCT_THRUST_CONTRIBUTION = auto()
    # CENTERBODY_THRUST_CONTRIBUTION = auto()
     
objective_IDs = [ObjectiveID.EFFICIENCY, ObjectiveID.FRONTAL_AREA]

# Define the multi-point operating conditions
multi_oper = [{"Inlet_Mach": 0.10285225,
               "N_crit": 9,
               "atmos": Atmosphere(0),
               "Omega": -9.667,
               "RPS": 25.237},
            #    {"Inlet_Mach": 0.20,
            #     "N_crit": 9,
            #     "atmos": Atmosphere(0),
            #     "Omega": -20,
            #     "RPS": 0},
                ]

# Compute the inlet velocities and write them to the multi-point oper dict
for oper_dict in multi_oper:
    oper_dict["Vinl"] = oper_dict["atmos"].speed_of_sound[0] * oper_dict["Inlet_Mach"]

# Controls for the optimisation vector - CENTERBODY
OPTIMIZE_CENTERBODY = False  # Control boolean to determine if centerbody should be optimised. If false, code uses the default entry below.
CENTERBODY_VALUES = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0., 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0., 'y_c': 0., 'z_TE': 0., 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': 0.27689037361278407, 'trailing_camberline_angle': 0., 'leading_edge_direction': 0., "Chord Length": 1.5, "Leading Edge Coordinates": (0., 0.)}


# Controls for the optimisation vector - DUCT
OPTIMIZE_DUCT = True
# DUCT_VALUES = {'b_0': 0., 'b_2': 0., 'b_8': 0.004081758291374328, 'b_15': 0.735, 'b_17': 0.8, 'x_t': 0.2691129541223092, 'y_t': 0.084601317961794, 'x_c': 0.5, 'y_c': 0., 'z_TE': -0.015685, 'dz_TE': 0.0005638524603968335, 'r_LE': -0.06953901280141099, 'trailing_wedge_angle': 0.16670974950670672, 'trailing_camberline_angle': 0.003666809042006104, 'leading_edge_direction': -0.811232599724247, 'Chord Length': 1.2446, "Leading Edge Coordinates": (0.093, 1.20968)}
DUCT_VALUES = {'b_0': 0.05, 'b_2': 0.2, 'b_8': 0.0016112203781740767, 'b_15': 0.875, 'b_17': 0.8, 'x_t': 0.28390800787161385, 'y_t': 0.08503466788167842, 'x_c': 0.4, 'y_c': 0.0, 'z_TE': -0.015685, 'dz_TE': 0.0005625060663762559, 'r_LE': -0.06974976321495045, 'trailing_wedge_angle': 0.13161296013687374, 'trailing_camberline_angle': 0.003666809042006104, 'leading_edge_direction': -0.811232599724247, "Chord Length": 1.2446, "Leading Edge Coordinates": (0.093, 1.20968)}
REF_FRONTAL_AREA = 5.1712  # m^2

# Controls for the optimisation vector - BLADES
OPTIMIZE_STAGE = [True, False, False]
ROTATING = [True, False, False]
NUM_RADIALSECTIONS = [4, 2, 2]  # Define the number of radial sections at which the blade profiles for each stage will be defined. 
NUM_STAGES = 3  # Define the number of stages (i.e. total count of rotors + stators)
REFERENCE_BLADE_ANGLES = [np.deg2rad(14.5), 0, 0]  # Reference angles at the reference section, measured at the blade tip. The 14.5 degree angle is equivalent to a 19deg angle at the 75% span location.
BLADE_DIAMETERS = [2.1336, 2.2098, 2.2098]
tipGap = 0.01016  # 1.016 cm tip gap

@functools.lru_cache(maxsize=None, typed=True)  # Unlimited - adjust if memory becomes a concern. 
def _load_blading(Omega: float,  
                  RPS: float,                      
                  ref_blade_angle: float) -> tuple[list, list]:
    """
    Generate MTFLO blading.
    The blading parameters are based on Figure 3 in [1].

    Parameters
    ----------
    - Omega : float
        The non-dimensional rotational speed of the rotor, as defined in the MTFLOW documentation in units of Vinl/Lref
    - RPS : float
        The rotational rate of the rotor in rotations per second. 
    - ref_blade_angle : float
        The blade set angle, in radians. 
    
    Returns
    -------
    - (list, list):
        A tuple containing two lists:
        - blading_parameters : list
            A list containing dictionaries with the blading parameters.
        - design_parameters : list
            A list containing dictionaries with the design parameters for each radial station.
    """

    # Start defining the MTFLO blading inputs
    # radial_stations = np.array([0.0, 0.5334, 1.0668])  # 0, 0.5, 1
    # chord_length = np.array([0.3510, 0.2528, 0.2205])
    # blade_angle = np.array([np.deg2rad(53.6), np.deg2rad(32.3), np.deg2rad(15.5)])
    radial_stations = np.array([0.0, 0.32004, 0.74676, 1.0668])  # 0, 0.3, 0.7, 1
    chord_length = np.array([0.3510, 0.3152, 0.2367, 0.2205])
    blade_angle = np.array([np.deg2rad(38.1), np.deg2rad(30.9), np.deg2rad(16.8), np.deg2rad(0)])
    propeller_parameters = {"root_LE_coordinate": 0.1495672948767407, 
                            "rotational_rate": Omega, 
                            "RPS": RPS,
                            # "RPS_lst": [RPS, RPS * 3],
                            "RPS_lst": [RPS],
                            "ref_blade_angle": ref_blade_angle, 
                            "reference_section_blade_angle": 0, 
                            "blade_count": 3, 
                            "radial_stations": radial_stations, 
                            "chord_length": chord_length, 
                            "blade_angle": blade_angle}
    
    horizontal_strut_parameters = {"root_LE_coordinate": 0.57785, 
                                   "rotational_rate": 0, 
                                   "RPS": 0,
                                #    "RPS_lst": [0, 0],
                                   "RPS_lst": [0],
                                   "ref_blade_angle": 0, 
                                   "reference_section_blade_angle": 0, 
                                   "blade_count": 4, 
                                   "radial_stations": np.array([0.0, 1.15]), 
                                   "chord_length": np.array([0.57658, 0.14224]), 
                                   "blade_angle": np.array([np.deg2rad(90), np.deg2rad(90)]),
                                   "sweep_angle": np.array([0, 0])}
    
    diagonal_strut_parameters = {"root_LE_coordinate": 0.577723, 
                                 "rotational_rate": 0, 
                                 "RPS": 0,
                                #  "RPS_lst": [0, 0],
                                 "RPS_lst": [0],
                                 "ref_blade_angle": 0, 
                                 "reference_section_blade_angle": 0, 
                                 "blade_count": 2, 
                                 "radial_stations": np.array([0.0, 1.15]), 
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
    root_blade_angle = (np.deg2rad(38.1) + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0]["reference_section_blade_angle"])

    root_LE = blading_parameters[0]["root_LE_coordinate"] # The location of the root LE is arbitrary for computing the sweep angles.
    root_mid_chord = root_LE + (0.3510 / 2) * np.cos(np.pi / 2 - root_blade_angle)
    rotation_angle = np.pi / 2 - (blade_angle + ref_blade_angle - propeller_parameters["reference_section_blade_angle"])
    local_LE = root_mid_chord - (chord_length / 2) * np.cos(rotation_angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        sweep_angle = np.where(radial_stations != 0, np.arctan((local_LE - root_LE) / radial_stations), 0)
    blading_parameters[0]["sweep_angle"] = sweep_angle

    # Obtain the parameterizations for the profile sections. 
    profile_dir_path = Path(__file__).parent.parent / 'Validation/Profiles'
    file_names = ['X22_02R.dat', 'X22_03R.dat', 'X22_07R.dat', 'X22_10R.dat', 'Hstrut.dat', 'Dstrut.dat']
    filenames = [profile_dir_path / stem for stem in file_names]
    
    # First check if all files are present
    missing_files = [f for f in filenames if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(map(str, missing_files))}")

    # Instantiate the AirfoilParameterization class and compute the parameterisations for the profile sections
    param = AirfoilParameterization()
    R00_section = param.FindInitialParameterization(reference_file=filenames[0])
    R03_section = param.FindInitialParameterization(reference_file=filenames[1])
    R07_section = param.FindInitialParameterization(reference_file=filenames[2])
    R10_section = param.FindInitialParameterization(reference_file=filenames[3])
    Hstrut_section = param.FindInitialParameterization(reference_file=filenames[4])
    Dstrut_section = param.FindInitialParameterization(reference_file=filenames[5])

    # Construct blading list
    design_parameters = [[R00_section, R03_section, R07_section, R10_section],
                         [Hstrut_section, Hstrut_section],
                         [Dstrut_section, Dstrut_section]]

    return copy.deepcopy(blading_parameters), copy.deepcopy(design_parameters)

# Compute the blading and design parameters for the rotors/stators of the reference design
STAGE_BLADING_PARAMETERS, STAGE_DESIGN_VARIABLES = _load_blading(multi_oper[0]["Omega"],
                                                                 multi_oper[0]["RPS"],
                                                                 REFERENCE_BLADE_ANGLES[0])

# Define the target thrust/power and efficiency for use in constraints
P_ref_constr = [0.76169 * (0.5 * atmosphere.density[0] * multi_oper[0]["Vinl"] ** 3 * BLADE_DIAMETERS[0] ** 2),
                # 1.5592 * (0.5 * atmosphere.density[0] * multi_oper[1]["Vinl"] ** 3 * BLADE_DIAMETERS[0] ** 2),
                ]  # Reference Power in Watts derived from baseline analysis
T_ref_constr = [0.60736 * (0.5 * atmosphere.density[0] * multi_oper[0]["Vinl"] ** 2 * BLADE_DIAMETERS[0] ** 2),
                # 1.2002 * (0.5 * atmosphere.density[0] * multi_oper[1]["Vinl"] ** 2 * BLADE_DIAMETERS[0] ** 2),
                ] # Reference Thrust in Newtons derived from baseline analysis
Eta_ref_constr = 0.79738
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
POPULATION_SIZE = 40
# Larger initial population for better diversity, then reduced to standard size
INITIAL_POPULATION_SIZE = 60
MAX_GENERATIONS = 50
MAX_EVALUATIONS = 4000


# Compute the total number of objectives
n_objectives = len(objective_IDs) * len(multi_oper) - sum([1 for ID in objective_IDs if ID in (1, 2)]) * (len(multi_oper) - 1)

# Define the initial population parameter spreads, used to construct a biased initial population 
SPREAD_CONTINUOUS = 0.25  # Relative spread (+/- %) applied to continous variables around their reference values
ZERO_NOISE = 0.25  # % noise added to zero values to avoid stagnation
SPREAD_DISCRETE = (-3, 6)  # Absolute range for discrete variables (referene value -3 to reference value + 6)

# Repair operator controls
PROFILE_FEASIBILITY_OFFSET = 0.05  # Offset value to avoid bezier control points lying on x_t/x_c
MAX_ONE2ONE_ATTEMPTS = 200  # Maximum number of attempts to enforce one-to-one on the profile parameterization. 

PROBLEM_TYPE = "single_point"  # Either "single_point" or "multi_point". Defines the type of problem loaded in the main file. 
RESERVED_THREADS = 0  # Threads reserved for the operating system and any other programs.
THREADS_PER_EVALUATION = 2  # Number of threads per MTFLOW evaluation: one for running MTSET/MTSOL/MTFLO and one for polling outputs