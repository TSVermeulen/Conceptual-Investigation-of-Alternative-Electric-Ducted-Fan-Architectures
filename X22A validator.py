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
import os
import time

from Submodels.Parameterizations import AirfoilParameterization
from Submodels.file_handling import fileHandling
from Submodels.output_handling import output_processing
from Submodels.MTSOL_call import MTSOL_call
from Submodels.MTSET_call import MTSET_call
from Submodels.MTFLO_call import MTFLO_call

REFERENCE_BLADE_ANGLE = np.deg2rad(29)  # radians, converted from degrees
ANALYSIS_NAME = "X22A_validation"  # Analysis name for MTFLOW
ALTITUDE = 3352  # m
FAN_DIAMETER = 84.75 * 2.54 / 100  # m, taken from [3] and converted to meters from inches

# Advance ratio range to be used for the validation. Taken from figure 5 in [1]. Note that J is approximate, as the data is read from a (scanned) graph
J = (np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65]))  
FREESTREAM_VELOCITY = np.ones_like(J) * 40  # m/s, tweaked to get acceptable values of RPS/OMEGA for the advance ratio range considered. 

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


def GenerateMTFLOBlading(Omega: float,
                         ref_blade_angle: float,
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
    blading_parameters = [{"root_LE_coordinate": 0.181102, "rotational_rate": Omega, "ref_blade_angle": ref_blade_angle, ".75R_blade_angle": np.deg2rad(20), "blade_count": 3, "radial_stations": [0.0,
                                                                                                                                                                                                     0.10647, 
                                                                                                                                                                                                     0.21294,
                                                                                                                                                                                                     0.31941, 
                                                                                                                                                                                                     0.42588,
                                                                                                                                                                                                     0.53235,
                                                                                                                                                                                                     0.63882,
                                                                                                                                                                                                     0.74529, 
                                                                                                                                                                                                     0.85176,
                                                                                                                                                                                                     0.95823,
                                                                                                                                                                                                     1.0647], 
                                                                                                                                                                                    "chord_length": [0.42256,
                                                                                                                                                                                                     0.3856,
                                                                                                                                                                                                    0.35052,
                                                                                                                                                                                                    0.3152,
                                                                                                                                                                                                    0.2794, 
                                                                                                                                                                                                    0.254,
                                                                                                                                                                                                    0.2413, 
                                                                                                                                                                                                    0.235527,
                                                                                                                                                                                                    0.230332,
                                                                                                                                                                                                    0.2251364,
                                                                                                                                                                                                    0.22098], 
                                                                                                                                                                                                    "sweep_angle": [0,
                                                                                                                                                                                                                    np.atan((0.42256 - 0.3856) / (2 * (0.10647 - 0.0))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.35052) / (2 * (0.21294 - 0.10647))), 
                                                                                                                                                                                                                    np.atan((0.3856 - 0.3152) / (2 * (0.31941 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.2794) / (2 * (0.42588 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.254) / (2 * (0.53235 - 0.10647))), 
                                                                                                                                                                                                                    np.atan((0.3856 - 0.2413) / (2 * (0.63882 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.235527) / (2 * (0.74529 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.230332) / (2 * (0.85176 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.2251364) / (2 * (0.95823 - 0.10647))),
                                                                                                                                                                                                                    np.atan((0.3856 - 0.22098) / (2 * (1.0647 - 0.10647)))], 
                                                                                                                                                                                                                    "blade_angle": [np.deg2rad(67.73),
                                                                                                                                                                                                                                    np.deg2rad(60.91),
                                                                                                                                                                                                                                    np.deg2rad(53.64), 
                                                                                                                                                                                                                                    np.deg2rad(46.82),
                                                                                                                                                                                                                                    np.deg2rad(40.00),
                                                                                                                                                                                                                                    np.deg2rad(32.27),
                                                                                                                                                                                                                                    np.deg2rad(26.36), 
                                                                                                                                                                                                                                    np.deg2rad(21.82),
                                                                                                                                                                                                                                    np.deg2rad(19.09),
                                                                                                                                                                                                                                    np.deg2rad(16.82),
                                                                                                                                                                                                                                    np.deg2rad(15.45)]}]
    
    if perform_parameterization:
        print("Generating Section Parameterizations....")
        # Obtain the parameterizations for the profile sections. 
        local_dir_path = Path('Validation')
        R00_fpath = local_dir_path / 'X22_00R.dat'
        R01_fpath = local_dir_path / 'X22_01R.dat'
        R02_fpath = local_dir_path / 'X22_02R.dat'
        R03_fpath = local_dir_path / 'X22_03R.dat'
        R04_fpath = local_dir_path / 'X22_04R.dat'
        R05_fpath = local_dir_path / 'X22_05R.dat'
        R06_fpath = local_dir_path / 'X22_06R.dat'
        R07_fpath = local_dir_path / 'X22_07R.dat'
        R08_fpath = local_dir_path / 'X22_08R.dat'
        R09_fpath = local_dir_path / 'X22_09R.dat'
        R10_fpath = local_dir_path / 'X22_10R.dat'

        # Compute parameterization for root airfoil section
        param_class = AirfoilParameterization()
        R00_section = param_class.FindInitialParameterization(reference_file=R00_fpath,
                                                            plot=True)
        print(R00_section)
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
        R05_section = param_class.FindInitialParameterization(reference_file=R05_fpath,
                                                            plot=True)
        print(R05_section)
        # Compute parameterization for the airfoil section at r=0.6R
        R06_section = param_class.FindInitialParameterization(reference_file=R06_fpath,
                                                            plot=True)
        print(R06_section)
        # Compute parameterization for the airfoil section at r=0.7R
        R07_section = param_class.FindInitialParameterization(reference_file=R07_fpath,
                                                            plot=True)
        print(R07_section)
        # Compute parameterization for the airfoil section at r=0.8R
        R08_section = param_class.FindInitialParameterization(reference_file=R08_fpath,
                                                            plot=True)
        print(R08_section)
        # Compute parameterization for the airfoil section at r=0.9R
        R09_section = param_class.FindInitialParameterization(reference_file=R09_fpath,
                                                            plot=True)
        print(R09_section)
        # Compute parameterization for the tip airfoil section
        R10_section = param_class.FindInitialParameterization(reference_file=R10_fpath,
                                                            plot=True)
        print(R10_section)
        print("Sections successfully parameterized...")
    else:
        # If we do not perform the parameterization, we can use the default data directly.

        # Uncomment below for parameterizations of the NASA 0010-64 modified airfoils, scaled to the correct t/c at each section as given by [1]
        R00_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.12398236503839008), 'b_15': np.float64(0.8249999999999986), 'b_17': np.float64(0.8), 'x_t': np.float64(0.319385), 'y_t': np.float64(0.22272254900122362), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.00519199999999953), 'r_LE': np.float64(-0.22009027242727328), 'trailing_wedge_angle': np.float64(0.4052859613533125), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R01_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.10886988285035995), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8), 'x_t': np.float64(0.31938500000000003), 'y_t': np.float64(0.1926416222727293), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.004202149127453786), 'r_LE': np.float64(-0.16970573777797837), 'trailing_wedge_angle': np.float64(0.3626844170472994), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R02_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.09158884682085988), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3193849999999999), 'y_t': np.float64(0.15752903893022369), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0030414982160082457), 'r_LE': np.float64(-0.12010641297556184), 'trailing_wedge_angle': np.float64(0.30807507013856983), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R03_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.07493804137479351), 'b_15': np.float64(0.8249999999998017), 'b_17': np.float64(0.8), 'x_t': np.float64(0.31938500000000003), 'y_t': np.float64(0.12226435701015367), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0023580000000000025), 'r_LE': np.float64(-0.08040548775481944), 'trailing_wedge_angle': np.float64(0.248871152206594), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R04_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.06299369999999999), 'b_15': np.float64(0.8249999999998896), 'b_17': np.float64(0.8), 'x_t': np.float64(0.319385), 'y_t': np.float64(0.09684839124421991), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.00189), 'r_LE': np.float64(-0.05817190574018529), 'trailing_wedge_angle': np.float64(0.20280490950701618), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R05_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.056700000001260874), 'b_15': np.float64(0.8249999999847404), 'b_17': np.float64(0.8), 'x_t': np.float64(0.31938499998257885), 'y_t': np.float64(0.08639770413693541), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0017010000000769743), 'r_LE': np.float64(-0.050746982198026466), 'trailing_wedge_angle': np.float64(0.18346909680831783), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R06_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.0441), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3193849999999842), 'y_t': np.float64(0.0655563955174534), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.001323000000000001), 'r_LE': np.float64(-0.038589159832080076), 'trailing_wedge_angle': np.float64(0.1437477569719401), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R07_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.0378), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8), 'x_t': np.float64(0.31938500000000003), 'y_t': np.float64(0.055222838904078426), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0011340000000000386), 'r_LE': np.float64(-0.033962787690348176), 'trailing_wedge_angle': np.float64(0.12462875900123835), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R08_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.025200000000038122), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3193849999993754), 'y_t': np.float64(0.03600000000002191), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0007560000000044782), 'r_LE': np.float64(-0.028456466460144818), 'trailing_wedge_angle': np.float64(0.08327107566449986), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R09_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.0189), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8), 'x_t': np.float64(0.3193849999996647), 'y_t': np.float64(0.027000000000000322), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0005670000000000002), 'r_LE': np.float64(-0.029008119264659373), 'trailing_wedge_angle': np.float64(0.06230012067313717), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}
        R10_section = {'b_0': np.float64(0.0), 'b_2': np.float64(0.0), 'b_8': np.float64(0.00945), 'b_15': np.float64(0.8249999995130213), 'b_17': np.float64(0.8), 'x_t': np.float64(0.31938500000000003), 'y_t': np.float64(0.0135), 'x_c': np.float64(0.0), 'y_c': np.float64(0.0), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0002790000001227517), 'r_LE': np.float64(-0.04298164745722162), 'trailing_wedge_angle': np.float64(0.032471786876037884), 'trailing_camberline_angle': np.float64(-0.0), 'leading_edge_direction': np.float64(0.0)}

        # Uncomment below for parameterizations of the NACA 24xx airfoils where xx is the correct t/c at each section as given by [1]
        R00_section = {'b_0': np.float64(0.043436799999999894), 'b_2': np.float64(0.21718399999989457), 'b_8': np.float64(0.11726758398228249), 'b_15': np.float64(0.825), 'b_17': np.float64(0.8799999999999977), 'x_t': np.float64(0.31257230128178937), 'y_t': np.float64(0.22337313795225675), 'x_c': np.float64(0.43436799999999137), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.005203286392132102), 'r_LE': np.float64(-0.20118750070286548), 'trailing_wedge_angle': np.float64(0.41622924910547937), 'trailing_camberline_angle': np.float64(0.06015380862316172), 'leading_edge_direction': np.float64(0.08601042561657472)}
        R01_section = {'b_0': np.float64(0.043436799999999894), 'b_2': np.float64(0.21718399999989457), 'b_8': np.float64(0.11726758398228249), 'b_15': np.float64(0.825), 'b_17': np.float64(0.8799999999999977), 'x_t': np.float64(0.31257230128178937), 'y_t': np.float64(0.22337313795225675), 'x_c': np.float64(0.43436799999999137), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.005203286392132102), 'r_LE': np.float64(-0.20118750070286548), 'trailing_wedge_angle': np.float64(0.41622924910547937), 'trailing_camberline_angle': np.float64(0.06015380862316172), 'leading_edge_direction': np.float64(0.08601042561657472)}
        R02_section = {'b_0': np.float64(0.043436799999976905), 'b_2': np.float64(0.21718399999979043), 'b_8': np.float64(0.0872200829966604), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8799999999986879), 'x_t': np.float64(0.31455761646087627), 'y_t': np.float64(0.15805045670186377), 'x_c': np.float64(0.43436800000000003), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.003312476954544271), 'r_LE': np.float64(-0.11059317245818538), 'trailing_wedge_angle': np.float64(0.31511452613791263), 'trailing_camberline_angle': np.float64(0.07311710570741244), 'leading_edge_direction': np.float64(0.0860104256165266)}
        R03_section = {'b_0': np.float64(0.043436799999999734), 'b_2': np.float64(0.2171839999999994), 'b_8': np.float64(0.0710043463646321), 'b_15': np.float64(0.8249999999999982), 'b_17': np.float64(0.8799999999982436), 'x_t': np.float64(0.3156278165874202), 'y_t': np.float64(0.1228105274087834), 'x_c': np.float64(0.4343679999999997), 'y_c': np.float64(0.019323202393123108), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0023632341906802226), 'r_LE': np.float64(-0.07304493839651645), 'trailing_wedge_angle': np.float64(0.25369493364862317), 'trailing_camberline_angle': np.float64(0.07311710570741237), 'leading_edge_direction': np.float64(0.08601042561652661)}
        R04_section = {'b_0': np.float64(0.043436799999991185), 'b_2': np.float64(0.21718399999958032), 'b_8': np.float64(0.06022029571699291), 'b_15': np.float64(0.8249999999999984), 'b_17': np.float64(0.8799999999997878), 'x_t': np.float64(0.3163685191919138), 'y_t': np.float64(0.09752560792091045), 'x_c': np.float64(0.4343679999999996), 'y_c': np.float64(0.019786992122682503), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0018941084902022582), 'r_LE': np.float64(-0.052418916083071386), 'trailing_wedge_angle': np.float64(0.20555236791833145), 'trailing_camberline_angle': np.float64(0.07235988269088045), 'leading_edge_direction': np.float64(0.09166773705192793)}
        R05_section = {'b_0': np.float64(0.043436800000000005), 'b_2': np.float64(0.21718399999981555), 'b_8': np.float64(0.055985115900761086), 'b_15': np.float64(0.8250000000000001), 'b_17': np.float64(0.8800000000000001), 'x_t': np.float64(0.3166798700248241), 'y_t': np.float64(0.08738273661728485), 'x_c': np.float64(0.43436800000000003), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.001704775809307488), 'r_LE': np.float64(-0.045260592578268165), 'trailing_wedge_angle': np.float64(0.1862509602710665), 'trailing_camberline_angle': np.float64(0.07311710570741246), 'leading_edge_direction': np.float64(0.08601042561652611)}
        R06_section = {'b_0': np.float64(0.0434367999999999), 'b_2': np.float64(0.21718399999983917), 'b_8': np.float64(0.0441133107895112), 'b_15': np.float64(0.8249999999998703), 'b_17': np.float64(0.8799999999996602), 'x_t': np.float64(0.3172810434180286), 'y_t': np.float64(0.06650838994007396), 'x_c': np.float64(0.43224434335031214), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0013258759431415772), 'r_LE': np.float64(-0.0340257104268357), 'trailing_wedge_angle': np.float64(0.14615001928109467), 'trailing_camberline_angle': np.float64(0.07235988269089653), 'leading_edge_direction': np.float64(0.09166773705195394)}
        R07_section = {'b_0': np.float64(0.04343679999999999), 'b_2': np.float64(0.21718399999999946), 'b_8': np.float64(0.03781095908017323), 'b_15': np.float64(0.8249999999999997), 'b_17': np.float64(0.8800000000000001), 'x_t': np.float64(0.3175816301145028), 'y_t': np.float64(0.05616229103205138), 'x_c': np.float64(0.4301062545016437), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0011364650941213513), 'r_LE': np.float64(-0.02984234089829805), 'trailing_wedge_angle': np.float64(0.12540512697470668), 'trailing_camberline_angle': np.float64(0.072359882690897), 'leading_edge_direction': np.float64(0.0916677370519278)}
        R08_section = {'b_0': np.float64(0.043435840490908366), 'b_2': np.float64(0.21717156107333008), 'b_8': np.float64(0.025206327358623846), 'b_15': np.float64(0.7198439596493504), 'b_17': np.float64(0.879999999850039), 'x_t': np.float64(0.3181782295043778), 'y_t': np.float64(0.03609913118415931), 'x_c': np.float64(0.42887440271799854), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0007576781374716192), 'r_LE': np.float64(-0.024971909460576956), 'trailing_wedge_angle': np.float64(0.08395161994845322), 'trailing_camberline_angle': np.float64(0.07311710570741244), 'leading_edge_direction': np.float64(0.09166773705192754)}
        R09_section = {'b_0': np.float64(0.0434368), 'b_2': np.float64(0.21718400000000002), 'b_8': np.float64(0.018907108904881547), 'b_15': np.float64(0.675), 'b_17': np.float64(0.879999999997568), 'x_t': np.float64(0.31847980901197753), 'y_t': np.float64(0.02701015557843469), 'x_c': np.float64(0.42772801400972266), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.0005682325470607155), 'r_LE': np.float64(-0.025664754474186754), 'trailing_wedge_angle': np.float64(0.06333992083729073), 'trailing_camberline_angle': np.float64(0.07235988269086556), 'leading_edge_direction': np.float64(0.09193464971585504)}
        R10_section = {'b_0': np.float64(0.043436799978288414), 'b_2': np.float64(0.21345586161713498), 'b_8': np.float64(0.009453554452440775), 'b_15': np.float64(0.6750000000000098), 'b_17': np.float64(0.7200000001174014), 'x_t': np.float64(0.3189324045059888), 'y_t': np.float64(0.013505077789201183), 'x_c': np.float64(0.38800644116728733), 'y_c': np.float64(0.01932320189174007), 'z_TE': np.float64(0.0), 'dz_TE': np.float64(0.00027960649157178285), 'r_LE': np.float64(-0.03969016947523491), 'trailing_wedge_angle': np.float64(0.031990793322947644), 'trailing_camberline_angle': np.float64(0.07169022917751999), 'leading_edge_direction': np.float64(0.08601042564128364)}

    # Construct blading list
    design_parameters = [[R00_section, R01_section, R02_section, R03_section, R04_section, R05_section, R06_section, R07_section, R08_section, R09_section, R10_section]]

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
    centerbody_x = np.flip(np.array([40.5, 36.6, 32.94, 26.75, 25.286, 21.89, 18.494, 17.03, 14.03, 10.98, 7.32, 3.4, 2.196, 0.8, 0.15, 0]) - 3.67) * 2.54 / 100 
    centerbody_y = np.flip(np.array([5.5, 5.866, 6.588, 8.15, 8.53, 8.75, 8.53, 8.25, 7.5, 6.4, 5.05, 3.65, 3.0, 2.1, 0.732, 0]) * 2.54 / 100)

    centerbody_x = np.flip(np.array([63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 43, 40, 34, 32, 30, 25, 22, 19, 15, 11, 6.3, 2.3, 0.7, 0.5, 0]) * 2.54 / 100) - 3.67 * 2.54 / 100
    centerbody_y = np.flip(np.array([1.7, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.6, 6.2, 7.1, 7.8, 8.5, 8.7, 8.2, 7.5, 6.5, 5.6, 4.4, 3.3, 1.7, 1.0, 0]) * 2.54 / 100)



    # centerbody_x = np.flip((np.array([40.5, 40.4, 40.3, 40, 39, 36.6, 32.94, 26.75, 25.286, 21.89, 18.494, 17.03, 14.03, 10.98, 7.32, 3.4, 2.196, 0.8, 0.15, 0]) - 3.67) * 2.54 / 100)
    # centerbody_y = np.flip(np.array([2.1, 2.1, 2.2, 2.3, 3.3, 4.5, 6.3, 8.15, 8.53, 8.75, 8.53, 8.25, 7.5, 6.4, 5.05, 3.65, 3.0, 2.1, 0.732, 0]) * 2.54 / 100)
    
    # centerbody_x = np.array([-3.237188708, -3.133349816, -3.055470646, -2.624044772, -2.458397015, -1.537444934, -1.454621055, -0.285197339, 
    #                          0.795221613, 3.208239684, 4.539602625, 6.202261081, 8.449631394, 10.6104693, 12.44248404, 
    #                          14.18796638, 15.76903713, 16.35127663, 17.43293176, 19.1784141, 19.34406185, 21.16742336, 
    #                          21.49871887, 23.07237256, 23.65213971, 24.31473074, 25.88220354, 26.95273308, 27.94043873, 30.41773802, 
    #                          30.66744583, 31.98768318, 32.15456711, 33.39321676, 35.21163355, 36.12022386, 36.86316641, 38.35152387, 
    #                          39.09446642, 39.59017351, 40.41717612, 40.45, 40.47, 40.5]) * 2.54 / 100
    
    # centerbody_y = np.array([0, 1.245572238, 0.916749078, 2.063427133, 2.062438191, 2.713656388, 2.713161917, 3.362896701, 
    #                          3.602715095, 4.327114987, 4.729614313, 5.130135755, 5.855524589, 6.335161377, 
    #                          6.980940394, 7.380967365, 7.86406545, 8.024768498, 8.346669064, 8.746696035, 8.745707093, 8.816910905, 
    #                          8.814933022, 8.805538074, 8.802076778, 8.798121011, 8.378315203, 7.96147622, 7.545131709, 7.037804549, 
    #                          7.118403308, 6.782163085, 6.863256316, 6.609592736, 6.352467859, 6.182864335, 6.014249753, 5.841184932, 
    #                          5.67257035, 5.587521352, 5.500494471, 5.5, 5.5, 5.5]) * 2.54 / 100
    

    # Perform smoothing interpolation on the centerbody geometry
    interpolated_centerbody_x = centerbody_x[0] + ((1 - np.cos(np.linspace(0, np.pi, 30))) / 2) * (centerbody_x[-1] - centerbody_x[0])  #  cosine spacing for increased resolution at LE and TE
    

    interpolated_centerbody_y = interpolate.LSQUnivariateSpline(centerbody_x,
                                                                centerbody_y,
                                                                t=np.linspace(centerbody_x[1], centerbody_x[-2], 5),
                                                                k=3,
                                                                )(interpolated_centerbody_x)  
    
    plt.figure()
    plt.title("Centre body geometry")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(centerbody_x, centerbody_y)
    plt.plot(interpolated_centerbody_x, interpolated_centerbody_y)
    plt.show()

    # plt.figure()
    # plt.title("Duct geometry")
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.plot(x_duct, y_duct)
    # plt.show()

    # Transform the data to the correct format
    # Ensures leading edge data point only occurs once to make sure a smooth spline is constructed, in accordance with the MTFLOW documentation. 
    centerbody_x_complete = np.concatenate((np.flip(interpolated_centerbody_x), interpolated_centerbody_x[:-2]), axis=0)
    centerbody_y_complete = np.concatenate((np.flip(interpolated_centerbody_y), -interpolated_centerbody_y[:-2]), axis=0)
    xy_centerbody = np.vstack((centerbody_x_complete, centerbody_y_complete)).T


    # NACA 0030-based centerbody coordinates, scaled and shifted to the correct size/location.
    # centerbody_x = np.array([1.0, 0.97927, 0.94287, 0.89780, 0.84691, 0.79199, 0.73438, 0.67512, 0.61510, 0.55507, 0.49571, 0.43759, 0.38128, 0.32727, 0.27602, 0.22796, 0.18350, 0.14303, 0.10691, 0.07549, 0.04909, 0.02805, 0.01265, 0.00321, 0.00000, 0.00321, 0.01265, 0.02805, 0.04909, 0.07549, 0.10691, 0.14303, 0.18350, 0.22796, 0.27602, 0.32727, 0.38128, 0.43759, 0.49571, 0.55507, 0.61510, 0.67512, 0.73438, 0.79199, 0.84691, 0.89780]) - 3.67 * 2.54 / 100
    # centerbody_y = np.array([0.005, 0.01359, 0.02619, 0.04373, 0.06264, 0.08165, 0.09963, 0.11559, 0.12882, 0.13889, 0.14564, 0.14912, 0.14951, 0.14726, 0.14280, 0.13637, 0.12799, 0.11788, 0.10592, 0.09190, 0.07771, 0.05785, 0.04416, 0.01940, 0.00000, -0.01940, -0.04416, -0.05785, -0.07771, -0.09190, -0.10592, -0.11788, -0.12799, -0.13637, -0.14280, -0.14726, -0.14951, -0.14912, -0.14564, -0.13889, -0.12882, -0.11559, -0.09963, -0.08165, -0.06264, -0.04373])  
    
    # centerbody_x = (np.array([0.79199, 0.73438, 0.67512, 0.61510, 0.55507, 0.49571, 0.43759, 0.38128, 0.32727, 0.27602, 0.22796, 0.18350, 0.14303, 0.10691, 0.07549, 0.04909, 0.02805, 0.01265, 0.00321, 0.00000, 0.00321, 0.01265, 0.02805, 0.04909, 0.07549, 0.10691, 0.14303, 0.18350, 0.22796, 0.27602, 0.32727, 0.38128, 0.43759, 0.49571, 0.55507, 0.61510, 0.67512, 0.73438, 0.79199, 0.84691, 0.89780]) - 3.67 * 2.54 / 100) * 1.2
    # centerbody_y = np.array([0.08165, 0.09963, 0.11559, 0.12882, 0.13889, 0.14564, 0.14912, 0.14951, 0.14726, 0.14280, 0.13637, 0.12799, 0.11788, 0.10592, 0.09190, 0.07771, 0.05785, 0.04416, 0.01940, 0.00000, -0.01940, -0.04416, -0.05785, -0.07771, -0.09190, -0.10592, -0.11788, -0.12799, -0.13637, -0.14280, -0.14726, -0.14951, -0.14912, -0.14564, -0.13889, -0.12882, -0.11559, -0.09963, -0.08165, -0.06264, -0.04373]) * 1.33

    # xy_centerbody = np.vstack((centerbody_x, centerbody_y)).T
    

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


def ChangeOMEGA(omega):
    """
    Rather than regenerating the tflow.xxx file from scratch, simply change omega 
    """

    with open(f"tflow.{ANALYSIS_NAME}", "r") as file:
        lines = file.readlines()

    omega_line = 11
    updated_omega = f"{omega} \n"
    lines[omega_line] = updated_omega

    with open(f"tflow.{ANALYSIS_NAME}", "w") as file:
        file.writelines(lines)


if __name__ == "__main__":
    # Create the MTSET geometry and write the input file walls.ANALYSIS_NAME
    GenerateMTSETGeometry()

    # Construct the MTFLO blading using omega=0 and reference blade angle. 
    # Perform parameterization can be optionally set to true in case different profiles are used compared to the default inputs
    blading_parameters, design_parameters = GenerateMTFLOBlading(Omega=0,
                                                                 ref_blade_angle=REFERENCE_BLADE_ANGLE,
                                                                 perform_parameterization=False)
    
    # Generate the MTFLO input file
    GenerateMTFLOInput(blading_parameters,
                       design_parameters)

    # Change working directory to the submodels folder
    try:
        current_dir = os.getcwd()
        subfolder_path = os.path.join(current_dir, 'Submodels')
        os.chdir(subfolder_path)
    except OSError as e:
        raise OSError from e
    
    # Create the grid
    MTSET_call(analysis_name=ANALYSIS_NAME,
               #streamwise_points=141
               ).caller()
    
    # Perform analysis for all omega, Mach, and Re combinations defined at the top of the file
    for i in range(len(OMEGA)):
        # Update the blade parameters to the correct omega 
        ChangeOMEGA(OMEGA[i])      

        # Create the grid
        # MTSET_call(analysis_name=ANALYSIS_NAME,
        #         #streamwise_points=141
        #         ).caller()
        
        # Wait for the grid file to be loaded
        time.sleep(1)

        #Load in the blade row(s) from MTFLO 
        MTFLO_call(ANALYSIS_NAME).caller() 

        # wait to ensure blade rows are loaded in
        time.sleep(1)

        # Define operating conditions
        oper = {"Inlet_Mach": inlet_mach[i],
                "Inlet_Reynolds": reynolds_inlet[i],
                "N_crit": 9,
                }
        
        # Execute MTSOL
        exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)] = MTSOL_call(operating_conditions=oper,
                                                                                                           analysis_name=ANALYSIS_NAME,
                                                                                                           ).caller(run_viscous=True,
                                                                                                                    generate_output=True,
                                                                                                                    )
        
        print(f"MTSOL finished with exit flags: {exit_flag, [(exit_flag_invisc, iter_count_invisc), (exit_flag_visc, iter_count_visc)]}")

        # Collect outputs from the forces.xxx file
        CT, CP, etaP = output_processing(ANALYSIS_NAME).GetCTCPEtaP()
        print(f"Omega: {OMEGA[i]}, CT: {CT}, CP: {CP}, etaP: {etaP}")
