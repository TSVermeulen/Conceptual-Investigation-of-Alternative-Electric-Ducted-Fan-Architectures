""" 
X22A_validation
===============

References
----------
[1] - https://ntrs.nasa.gov/api/citations/19670025554/downloads/19670025554.pdf 
[2] - https://apps.dtic.mil/sti/tr/pdf/AD0447814.pdf?form=MG0AV3 
[3] - https://arc.aiaa.org/doi/10.2514/1.C037541 

"""

import numpy as np
from pathlib import Path
from scipy import interpolate
from ambiance import Atmosphere
import matplotlib.pyplot as plt

from Submodels.Parameterizations import AirfoilParameterization
from MTFLOW_caller import MTFLOW_caller
from Submodels.file_handling import fileHandling
from Submodels.output_handling import output_processing

REFERENCE_BLADE_ANGLE = np.deg2rad(29)  # radians, converted from degrees
ANALYSIS_NAME = "X22A_validation"  # Analysis name for MTFLOW
FREESTREAM_VELOCITY = np.array([30, 30, 36, 44])  # m/s, tweaked to get acceptable values of RPS/OMEGA for the advance ratio range considered. 
ALTITUDE = 3352  # m
FAN_DIAMETER = 84.75 * 2.54 / 100  # m, taken from [3] and converted to meters from inches

# Advance ratio range to be used for the validation. Taken from figure 5 in [1]. Note that J is approximate, as the data is read from a (scanned) graph
J = np.array([0.3, 0.4, 0.5, 0.6])  

# Compute the rotational speed of the rotor in rotations per second
RPS = FREESTREAM_VELOCITY / (J * FAN_DIAMETER)
print(f"RPM (Should be between 1200-2590 RPM) [-]: {RPS * 60}")

# Use the calculated rotational speed to obtain the non-dimensional Omega used as input into MTFLOW
OMEGA = RPS * FAN_DIAMETER / FREESTREAM_VELOCITY
print(f"OMEGA [-]: {OMEGA}")

# Construct atmosphere object to obtain the atmospheric properties at the cruise altitude
# These properties can then be used to compute the inlet mach number and reynolds number
atmosphere = Atmosphere(ALTITUDE)
inlet_mach = (FREESTREAM_VELOCITY / atmosphere.speed_of_sound)
print(f"Inlet Mach Number [-]: {inlet_mach}")

reynolds_inlet = (FREESTREAM_VELOCITY * FAN_DIAMETER / (atmosphere.kinematic_viscosity))  # Uses the MTFLOW internal reference length! 
print(f"Inlet Reynolds Number [-]: {reynolds_inlet}")


def GenerateMTFLOBlading(Omega: float = 0.,
                         ref_blade_angle: float = np.deg2rad(19),
                         perform_parameterization: bool = False):
    """
    Generate MTFLO blading
    [2] mentions the use of a modified NASA 001-64 profile. We scale this profile to have the correct thickness for each radial station, 
    but that is the limit of approximations we can make with the limited data available. 
    
    The blading parameters are based on Figure 3 in [1].

    Parameters
    ----------
    - Omega : float
        The non-dimensional rotational speed of the rotor, as defined in the MTFLOW documentation in units of Vinl/Lref
    - ref_blade_angle : float
        The blade set angle, in radians. 
    """

    # Start defining the MTFLO blading inputs
    blading_parameters = [{"root_LE_coordinate": 0.181102, "rotational_rate": Omega, "ref_blade_angle": ref_blade_angle, ".75R_blade_angle": np.deg2rad(34.3), "blade_count": 3, "radial_stations": [0.0,
                                                                                                                                                                                                     0.10647, 
                                                                                                                                                                                                     0.21294,
                                                                                                                                                                                                     0.31941, 
                                                                                                                                                                                                     0.42588,
                                                                                                                                                                                                     0.53235,
                                                                                                                                                                                                     0.74529, 
                                                                                                                                                                                                     1.0647], 
                                                                                                                                                                                    "chord_length": [0.42256,
                                                                                                                                                                                                     0.3856,
                                                                                                                                                                                                    0.35052,
                                                                                                                                                                                                    0.3152,
                                                                                                                                                                                                    0.2794, 
                                                                                                                                                                                                    0.254, 
                                                                                                                                                                                                    0.235527,
                                                                                                                                                                                                    0.22098], 
                                                                                                                                                                                                    "sweep_angle": [0,
                                                                                                                                                                                                                    np.atan((0.42256 - 0.3856) / (2 * (0.10647 - 0.0))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.35052) / (2 * (0.21294 - 0.10647))), 
                                                                                                                                                                                                                    np.atan((0.3856 - 0.3152) / (2 * (0.31941 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.2794) / (2 * (0.42588 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.254) / (2 * (0.53235 - 0.10647))), 
                                                                                                                                                                                                                    np.atan((0.3856 - 0.235527) / (2 * (0.74529 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.22098) / (2 * (1.0647 - 0.10647)))], 
                                                                                                                                                                                                                    "blade_angle": [np.deg2rad(67.73),
                                                                                                                                                                                                                                    np.deg2rad(60.91),
                                                                                                                                                                                                                                    np.deg2rad(54), 
                                                                                                                                                                                                                                    np.deg2rad(50.2),
                                                                                                                                                                                                                                    np.deg2rad(46.9),
                                                                                                                                                                                                                                    np.deg2rad(43.3), 
                                                                                                                                                                                                                                    np.deg2rad(35.9),
                                                                                                                                                                                                                                    np.deg2rad(26.5)]}]
    
    if perform_parameterization:
        # Obtain the parameterizations for the profile sections. 
        local_dir_path = Path('Validation')
        root_fpath = local_dir_path / 'X22_root.dat'
        R01_fpath = local_dir_path / 'X22_01R.dat'
        R02_fpath = local_dir_path / 'X22_02R.dat'
        R03_fpath = local_dir_path / 'X22_03R.dat'
        R04_fpath = local_dir_path / 'X22_04R.dat'
        mid_fpath = local_dir_path / 'X22_mid.dat'
        R07_fpath = local_dir_path / 'X22_07R.dat'
        tip_fpath = local_dir_path / 'X22_tip.dat'

        # Compute parameterization for root airfoil section
        param_class = AirfoilParameterization()
        root_section = param_class.FindInitialParameterization(reference_file=root_fpath,
                                                            plot=True)
        print(root_section)
        # Compute parameterization for the airfoil section at r=0.1R
        R01_section = param_class.FindInitialParameterization(reference_file=R01_fpath,
                                                            plot=True)
        print(R01_section)
        # Compute parameterization for the airfoil section at r=0.2R
        R02_section = param_class.FindInitialParameterization(reference_file=R02_fpath,
                                                            plot=True)
        print(R02_section)
        # Compute parameterization for the airfoil section at r=0.3R
        R03_section = param_class.FindInitialParameterization(reference_file=R03_fpath,
                                                            plot=True)
        print(R03_section)
        # Compute parameterization for the airfoil section at r=0.4R
        R04_section = param_class.FindInitialParameterization(reference_file=R04_fpath,
                                                            plot=True)
        print(R04_section)
        # Compute parameterization for the mid airfoil section
        mid_section = param_class.FindInitialParameterization(reference_file=mid_fpath,
                                                            plot=True)
        print(mid_section)
        # Compute parameterization for the airfoil section at r=0.7R
        R07_section = param_class.FindInitialParameterization(reference_file=R07_fpath,
                                                            plot=True)
        print(R07_section)
        # Compute parameterization for the tip airfoil section
        tip_section = param_class.FindInitialParameterization(reference_file=tip_fpath,
                                                            plot=True)
        print(tip_section)
    else:
        # If we do not perform the parameterization, we can use the default data directly.

        # Uncomment below for parameterizations of the NASA 0010-64 modified airfoils, scaled to the correct t/c at each section as given by [1]
        # root_section = {'b_0': np.float64(0.0), 'b_2': np.float64(1.0494616332700592e-15), 'b_8': np.float64(0.09411341148635839), 'b_15': np.float64(0.8152267119721809), 'b_17': np.float64(0.8000000000000003), 'x_t': np.float64(0.35573319329500624), 'y_t': np.float64(0.22432627899827806), 'x_c': np.float64(9.999999988685286e-11), 'y_c': np.float64(7.5349244317312e-31), 'z_TE': np.float64(-6.718537758389443e-31), 'dz_TE': np.float64(0.0075424270184872326), 'r_LE': np.float64(-0.14007004061826644), 'trailing_wedge_angle': np.float64(0.4940307727103302), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # R01_section = {'b_0': np.float64(0.0), 'b_2': np.float64(-2.672828584724189e-17), 'b_8': np.float64(0.08585115547974777), 'b_15': np.float64(0.7953275053930488), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3579281911479124), 'y_t': np.float64(0.19020051373531507), 'x_c': np.float64(9.999999999017217e-11), 'y_c': np.float64(-3.3343083965088264e-32), 'z_TE': np.float64(-1.584857263205951e-32), 'dz_TE': np.float64(0.006341139075957525), 'r_LE': np.float64(-0.10521337071236224), 'trailing_wedge_angle': np.float64(0.44221908197314785), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # R02_section = {'b_0': np.float64(0.0), 'b_2': np.float64(-1.6015448245552788e-18), 'b_8': np.float64(0.07938850615648092), 'b_15': np.float64(0.7834186798106341), 'b_17': np.float64(0.8), 'x_t': np.float64(0.36329881056992414), 'y_t': np.float64(0.15714856661614984), 'x_c': np.float64(1.0000000000621005e-10), 'y_c': np.float64(2.8837455065765125e-31), 'z_TE': np.float64(1.5420585290878846e-31), 'dz_TE': np.float64(0.004979611635648663), 'r_LE': np.float64(-0.07670381417045467), 'trailing_wedge_angle': np.float64(0.38764252377470126), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # R03_section = {'b_0': np.float64(0.0), 'b_2': np.float64(1.931425156225862e-18), 'b_8': np.float64(0.06633556260351585), 'b_15': np.float64(0.7681322724996446), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3654308552023536), 'y_t': np.float64(0.12474390795031891), 'x_c': np.float64(9.99999999991782e-11), 'y_c': np.float64(1.1746330286395878e-21), 'z_TE': np.float64(-5.816115912830538e-23), 'dz_TE': np.float64(0.003922353074894919), 'r_LE': np.float64(-0.04969230569437875), 'trailing_wedge_angle': np.float64(0.31884907958902486), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # R04_section = {'b_0': np.float64(0.0), 'b_2': np.float64(-2.303349036188369e-18), 'b_8': np.float64(0.05909929469107383), 'b_15': np.float64(0.761590676673648), 'b_17': np.float64(0.8), 'x_t': np.float64(0.37016322127831186), 'y_t': np.float64(0.10314850608719811), 'x_c': np.float64(1.000000000039312e-10), 'y_c': np.float64(-2.9146761596293495e-32), 'z_TE': np.float64(-1.8445040397270788e-32), 'dz_TE': np.float64(0.0030969179265496628), 'r_LE': np.float64(-0.03533032261317036), 'trailing_wedge_angle': np.float64(0.2735843700675792), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # mid_section = {'b_0': np.float64(0.0), 'b_2': np.float64(-7.480007717645859e-19), 'b_8': np.float64(0.05366468158819504), 'b_15': np.float64(0.7581979955516953), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3758481979028474), 'y_t': np.float64(0.08693435994735386), 'x_c': np.float64(1.0000000000375219e-10), 'y_c': np.float64(3.1532033231399504e-31), 'z_TE': np.float64(1.1663796349056909e-31), 'dz_TE': np.float64(0.002455048446545025), 'r_LE': np.float64(-0.02605625915598885), 'trailing_wedge_angle': np.float64(0.2390393084978324), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # R07_section = {'b_0': np.float64(0.0), 'b_2': np.float64(-6.599169007689791e-19), 'b_8': np.float64(0.04130598342251118), 'b_15': np.float64(0.7518983627959437), 'b_17': np.float64(0.8), 'x_t': np.float64(0.389391509494732), 'y_t': np.float64(0.0581202522837689), 'x_c': np.float64(9.999999999935408e-11), 'y_c': np.float64(3.2549258126219314e-34), 'z_TE': np.float64(-9.757434897806068e-35), 'dz_TE': np.float64(0.0014372059257603737), 'r_LE': np.float64(-0.012418278748750618), 'trailing_wedge_angle': np.float64(0.1713112070773209), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        # tip_section = {'b_0': np.float64(0.0), 'b_2': np.float64(-1.8221176558032628), 'b_8': np.float64(0.04050328282737513), 'b_15': np.float64(0.7535544788925614), 'b_17': np.float64(0.8243303423622311), 'x_t': np.float64(0.3904202504820698), 'y_t': np.float64(0.0565726619312102), 'x_c': np.float64(0.008327798056102542), 'y_c': np.float64(-4.021740487047529e-17), 'z_TE': np.float64(-5.000000000015975e-06), 'dz_TE': np.float64(0.0013869233576762456), 'r_LE': np.float64(-0.01180173208324647), 'trailing_wedge_angle': np.float64(0.16756899676425166), 'trailing_camberline_angle': np.float64(4.336808689942014e-18), 'leading_edge_direction': np.float64(0.0)}

        # Uncomment below for parameterizations of the NACA 24xx airfoils where xx is the correct t/c at each section as given by [1]
        root_section = {'b_0': np.float64(0.043436799999999866), 'b_2': np.float64(0.2171839999999995), 'b_8': np.float64(0.11726758398228249), 'b_15': np.float64(0.8249999999999961), 'b_17': np.float64(0.8799999999995232), 'x_t': np.float64(0.31257230128178937), 'y_t': np.float64(0.22337313723665178), 'x_c': np.float64(0.43436799999999987), 'y_c': np.float64(0.017995500000004886), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.005203286392116128), 'r_LE': np.float64(-0.20118750070286548), 'trailing_wedge_angle': np.float64(0.4162288664907536), 'trailing_camberline_angle': np.float64(0.06014948230313067), 'leading_edge_direction': np.float64(0.08601042561658903)}
        R01_section = {'b_0': np.float64(0.043436800000000005), 'b_2': np.float64(0.21718400000000002), 'b_8': np.float64(0.1033965981555799), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8799999999426072), 'x_t': np.float64(0.31347926061091796), 'y_t': np.float64(0.193203872014576), 'x_c': np.float64(0.4343679999998398), 'y_c': np.float64(0.0179955), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.004508631764151873), 'r_LE': np.float64(-0.1559549692216496), 'trailing_wedge_angle': np.float64(0.3724798349891539), 'trailing_camberline_angle': np.float64(0.06664971979179658), 'leading_edge_direction': np.float64(0.0860104256165266)}
        R02_section = {'b_0': np.float64(0.043436800000000005), 'b_2': np.float64(0.21718400000000002), 'b_8': np.float64(0.0872200829966604), 'b_15': np.float64(0.824999999999641), 'b_17': np.float64(0.8799999999297953), 'x_t': np.float64(0.31455761646087627), 'y_t': np.float64(0.1579002508328601), 'x_c': np.float64(0.4343679999998469), 'y_c': np.float64(0.01809369346624768), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0035577328831318694), 'r_LE': np.float64(-0.11059317245818538), 'trailing_wedge_angle': np.float64(0.31511452613744084), 'trailing_camberline_angle': np.float64(0.07311710561474492), 'leading_edge_direction': np.float64(0.08601042561654795)}
        R03_section = {'b_0': np.float64(0.04337204470291165), 'b_2': np.float64(0.21272120200496192), 'b_8': np.float64(0.0710043463646321), 'b_15': np.float64(0.825), 'b_17': np.float64(0.8732266828645529), 'x_t': np.float64(0.31562781658742023), 'y_t': np.float64(0.12183457677329662), 'x_c': np.float64(0.4311965007745416), 'y_c': np.float64(0.017999382320190837), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0025526339030829876), 'r_LE': np.float64(-0.07304493839651645), 'trailing_wedge_angle': np.float64(0.25369493364847046), 'trailing_camberline_angle': np.float64(0.07298637804534369), 'leading_edge_direction': np.float64(0.08693031622038633)}
        R04_section = {'b_0': np.float64(0.043436800000000005), 'b_2': np.float64(0.21718399999998275), 'b_8': np.float64(0.06022029571699291), 'b_15': np.float64(0.8249999999993514), 'b_17': np.float64(0.8799999999999994), 'x_t': np.float64(0.3163685191920674), 'y_t': np.float64(0.09752558801509807), 'x_c': np.float64(0.43436799999999975), 'y_c': np.float64(0.019787545100461976), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.001894108490202252), 'r_LE': np.float64(-0.052418916083071386), 'trailing_wedge_angle': np.float64(0.2055523679181529), 'trailing_camberline_angle': np.float64(0.07235988269089702), 'leading_edge_direction': np.float64(0.09166773705192859)}
        mid_section = {'b_0': np.float64(0.04343679999382484), 'b_2': np.float64(0.2171839999977746), 'b_8': np.float64(0.05598511590076289), 'b_15': np.float64(0.8249999999999986), 'b_17': np.float64(0.8799594643153933), 'x_t': np.float64(0.31667987002507975), 'y_t': np.float64(0.08738195392982298), 'x_c': np.float64(0.43436799999984527), 'y_c': np.float64(0.02009683394747112), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0017047758093173887), 'r_LE': np.float64(-0.04526059257826665), 'trailing_wedge_angle': np.float64(0.18625096027106675), 'trailing_camberline_angle': np.float64(0.07311710570740025), 'leading_edge_direction': np.float64(0.08601042561688882)}
        R07_section = {'b_0': np.float64(0.043388767618650605), 'b_2': np.float64(0.217183999999723), 'b_8': np.float64(0.03781095908017323), 'b_15': np.float64(0.8249999999992228), 'b_17': np.float64(0.8799999999999821), 'x_t': np.float64(0.31758163011450247), 'y_t': np.float64(0.056155497269749194), 'x_c': np.float64(0.4306324551761999), 'y_c': np.float64(0.020402430492400765), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0011364650941242164), 'r_LE': np.float64(-0.029842340898292304), 'trailing_wedge_angle': np.float64(0.12540512697471534), 'trailing_camberline_angle': np.float64(0.07235988269089534), 'leading_edge_direction': np.float64(0.09166773705192825)}
        tip_section = {'b_0': np.float64(0.043388767618650605), 'b_2': np.float64(0.217183999999723), 'b_8': np.float64(0.03781095908017323), 'b_15': np.float64(0.8249999999992228), 'b_17': np.float64(0.8799999999999821), 'x_t': np.float64(0.31758163011450247), 'y_t': np.float64(0.056155497269749194), 'x_c': np.float64(0.4306324551761999), 'y_c': np.float64(0.020402430492400765), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0011364650941242164), 'r_LE': np.float64(-0.029842340898292304), 'trailing_wedge_angle': np.float64(0.12540512697471534), 'trailing_camberline_angle': np.float64(0.07235988269089534), 'leading_edge_direction': np.float64(0.09166773705192825)}

    # Construct blading list
    design_parameters = [[root_section, R01_section, R02_section, R03_section, R04_section, mid_section, R07_section, tip_section]]

    return blading_parameters, design_parameters


def GenerateMTFLOInput(blading_parameters,
                       design_parameters):
    """
    Generate MTFLO input file tflow.X22A_validation
    """
    
    fileHandling.fileHandlingMTFLO(case_name=ANALYSIS_NAME,
                                   ref_length=FAN_DIAMETER).GenerateMTFLOInput(blading_params=blading_parameters,
                                                                               design_params=design_parameters)


def GenerateMTSETGeometry():
    """
    Generate the duct and center body geometry

    Returns
    -------
    - xy_centerbody, xy_duct : tuple[np.ndarray, np.ndarray]
        tuple of the centerbody and duct x,y coordinates
    """

    # First define the upper surface based on the given coordinates in [1]
    # Note that we must flip the arrays to comply with the formatting expected by MTFLOW (TE-LE-TE)
    # Dimensions are originally provided in inches, so they are converted to meters
    upper_x = np.flip(np.array([0, 0.613, 1.225, 2.45, 3.675, 4.9, 7.35, 9.8, 10.25, 12.25, 14.7, 17.75, 19.6, 23.7, 24.5, 29.4, 34.3, 39.2, 44.1, 46.55, 49])) * 2.54 / 100  # x coordinates of the duct upper surface
    upper_y = np.flip(np.array([47.625, 48.695, 49.096, 49.609, 49.953, 50.205, 50.535, 50.710, 50.745, 50.779, 50.763, 50.6575, 50.552, 50.25, 50.164, 49.649, 49.038, 48.344, 47.576, 47.160, 46.722])) * 2.54 / 100  # y coordinates of the duct upper surface

    # The lower surface must be reconstructed from parts defined by the shapes and coordinates given in [1]
    lower_x = np.array([])
    lower_y = np.array([])

    # The nose is represented by a ellipse for which the points are given as:
    P1 = (0, 47.625)  # Vertex 1
    P2 = (10.25, 42.365)  # Vertex 2
    C = (10.25, 47.625)  # Center

    # We can then calculate the semi-major and semi-minor axes of the ellipse as:
    a = P2[0]
    b = C[1] - P2[1]
    h, k = C

    # Construct an array of x values along which to reconstruct the leading edge nose shape, 
    # and use it to calculate the corresponding y values
    # Both x_nose and y_nose are still in inches
    x_nose = np.linspace(P1[0], C[0], 30)
    y_nose = (k - b * np.sqrt(1 - ((x_nose - h) ** 2 / a ** 2))) 

    lower_x = np.append(lower_x, x_nose)
    lower_y = np.append(lower_y, y_nose)

    # Raw approximated data for aft section of the lower surface
    # Data taken from a graph digitizer program from Bram Meijerink.
    # Note that the y values are not matching the graph from [1], so the data is scaled. 
    # After scaling, a smoothing interpolation using a univariatespline is used to create a more physical curve.  
    aft_x_raw = np.array([17.75, 21.4, 24.2, 28, 31.1, 33.4, 35.9, 38, 39.6, 41.7, 43.9, 45.5, 47.2, 48.4, 49])
    aft_y_raw = np.array([42.375, 42.4562, 42.6226, 42.8506, 43.1218, 43.3298, 43.5878, 43.7874, 43.9954, 44.2618, 44.4946, 44.661, 44.869, 44.9938, 45.05])
    scale_factor = (46.65 - lower_y[-1]) / (aft_y_raw[-1] - lower_y[-1])
    aft_y_raw_scaled = lower_y[-1] + scale_factor * (aft_y_raw - lower_y[-1])

    # Smooth the aft y data using a univariatespline
    x = np.linspace(aft_x_raw[0], aft_x_raw[-1], 30)
    aft_y_smoothed = interpolate.UnivariateSpline(aft_x_raw, aft_y_raw_scaled, s=0.01)(x)

    # Construct full lower surface array and convert from inches to meters
    lower_x = np.append(lower_x, x) * 2.54 / 100
    lower_y = np.append(lower_y, aft_y_smoothed) * 2.54 / 100

    # Transform the xy arrays for the duct to the correct format
    x_duct = np.concatenate((upper_x, lower_x[1:]), axis=0)
    y_duct = np.concatenate((upper_y, lower_y[1:]), axis=0)
    xy_duct = np.vstack((x_duct, y_duct)).T

    # --------------------
    # Generate centre body geometry
    # --------------------

    # Data taken from [1]
    centerbody_x = (np.array([40.5, 36.6, 32.94, 26.75, 25.286, 21.89, 18.494, 17.03, 14.03, 10.98, 7.32, 3.4, 2.196, 0.8, 0.15, 0]) - 3.67) * 2.54 / 100 
    centerbody_y = np.array([5.5, 5.866, 6.588, 8.15, 8.53, 8.75, 8.53, 8.25, 7.5, 6.4, 5.05, 3.65, 3.0, 2.1, 0.732, 0]) * 2.54 / 100   

    centerbody_x = (np.array([40.5, 39, 36.6, 32.94, 26.75, 25.286, 21.89, 18.494, 17.03, 14.03, 10.98, 7.32, 3.4, 2.196, 0.8, 0.15, 0]) - 3.67) * 2.54 / 100 
    centerbody_y = np.array([2.1, 3.3, 4.5, 6.3, 8.15, 8.53, 8.75, 8.53, 8.25, 7.5, 6.4, 5.05, 3.65, 3.0, 2.1, 0.732, 0]) * 2.54 / 100  
     
    # centerbody_x = (np.array([40.5, 39, 36.6, 32.94, 26.75, 25.286, 21.89, 18.494, 17.03, 14.03, 10.98, 7.32, 3.4, 2.196, 0.8, 0.15, 0]) - 3.67) * 2.54 / 100 
    # centerbody_y = np.array([5.9, 5.9, 6.0, 6.3, 8.15, 8.53, 8.75, 8.53, 8.25, 7.5, 6.4, 5.05, 3.65, 3.0, 2.1, 0.732, 0]) * 2.54 / 100   

    # plt.figure()
    # plt.title("Centre body geometry")
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.plot(centerbody_x, centerbody_y)
    # plt.plot(centerbody_x, -centerbody_y)
    # plt.show()

    # plt.figure()
    # plt.title("Duct geometry")
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.plot(x_duct, y_duct)
    # plt.show()

    # Transform the data to the correct format
    # Ensures leading edge data point only occurs once to make sure a smooth spline is constructed, in accordance with the MTFLOW documentation. 
    centerbody_x_complete = np.concatenate((centerbody_x, np.flip(centerbody_x[:-2])), axis=0)
    centerbody_y_complete = np.concatenate((centerbody_y, np.flip(-centerbody_y[:-2])), axis=0)
    xy_centerbody = np.vstack((centerbody_x_complete, centerbody_y_complete)).T

    centerbody_x_0030 = np.array([1.0, 0.97927, 0.94287, 0.89780, 0.84691, 0.79199, 0.73438, 0.67512, 0.61510, 0.55507, 0.49571, 0.43759, 0.38128, 0.32727, 0.27602, 0.22796, 0.18350, 0.14303, 0.10691, 0.07549, 0.04909, 0.02805, 0.01265, 0.00321, 0.00000, 0.00321, 0.01265, 0.02805, 0.04909, 0.07549, 0.10691, 0.14303, 0.18350, 0.22796, 0.27602, 0.32727, 0.38128, 0.43759, 0.49571, 0.55507, 0.61510, 0.67512, 0.73438, 0.79199, 0.84691, 0.89780]) * 1.5 - 3.67 * 2.54 / 100
    centerbody_y_0030 = np.array([0.012, 0.01359, 0.02619, 0.04373, 0.06264, 0.08165, 0.09963, 0.11559, 0.12882, 0.13889, 0.14564, 0.14912, 0.14951, 0.14726, 0.14280, 0.13637, 0.12799, 0.11788, 0.10592, 0.09190, 0.07771, 0.05785, 0.04416, 0.01940, 0.00000, -0.01940, -0.04416, -0.05785, -0.07771, -0.09190, -0.10592, -0.11788, -0.12799, -0.13637, -0.14280, -0.14726, -0.14951, -0.14912, -0.14564, -0.13889, -0.12882, -0.11559, -0.09963, -0.08165, -0.06264, -0.04373]) * 1.5 
    #xy_centerbody = np.vstack((centerbody_x_0030, centerbody_y_0030)).T

    # --------------------
    # Generate MTSET input file walls.X22A_validation
    # To pass the class input validation, dummy inputs need to be provided
    # --------------------

    params_CB = {"Leading Edge Coordinates": (centerbody_x.min(),0), "Chord Length": (lower_x.max() - lower_x.min())}
    params_duct = {"Leading Edge Coordinates": (x_duct.min(), upper_y[-1]), "Chord Length": (lower_x.max() - lower_x.min())}

    fileHandling().fileHandlingMTSET(params_CB=params_CB,
                                     params_duct=params_duct,
                                     case_name=ANALYSIS_NAME,
                                     ref_length=FAN_DIAMETER,
                                     external_input=True).GenerateMTSETInput(xy_centerbody=xy_centerbody,
                                                                             xy_duct=xy_duct)


def RunMTFLOW(oper: dict,
              Omega: float,
              ref_blade_angle: float,
              perform_param: bool = False
              ):
    """
    Execute MTFLOW

    Parameters
    ----------

    Returns
    -------
    """
    
    # Create the MTSET geometry and write the input file walls.ANALYSIS_NAME
    GenerateMTSETGeometry()

    # Construct the MTFLO blading using the provided omega and reference blade angle. 
    # Perform parameterization can be optionally set to true in case 
    blading_parameters, design_parameters = GenerateMTFLOBlading(Omega,
                                                                 ref_blade_angle,
                                                                 perform_parameterization=perform_param)

    GenerateMTFLOInput(blading_parameters,
                       design_parameters)

    MTFLOW_caller(operating_conditions=oper,
                  centrebody_params={},
                  duct_params={},
                  blading_parameters=blading_parameters,
                  design_parameters=design_parameters,
                  ref_length=FAN_DIAMETER,
                  analysis_name=ANALYSIS_NAME).caller(debug=False,
                                                      external_inputs=True)
    
    CT, CP, etaP = output_processing(ANALYSIS_NAME).GetCTCPEtaP()

    return CT, CP, etaP


if __name__ == "__main__":
    
    for i in range(len(OMEGA)):
        # Define operating conditions
        oper = {"Inlet_Mach": inlet_mach[i],
                "Inlet_Reynolds": reynolds_inlet[i],
                "N_crit": 9,
                }
        
        CT, CP, etaP = RunMTFLOW(oper=oper,
                                 Omega=OMEGA[i],
                                 ref_blade_angle=REFERENCE_BLADE_ANGLE,
                                 perform_param=False)
        print(f"Omega: {OMEGA[i]}, CT: {CT}, CP: {CP}, etaP: {etaP}")
