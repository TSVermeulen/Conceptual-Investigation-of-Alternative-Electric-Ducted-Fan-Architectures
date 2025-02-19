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

REFERENCE_BLADE_ANGLE = np.radians(19)  # radians
ANALYSIS_NAME = "X22A_validation"  # Analysis name for MTFLOW
FREESTREAM_VELOCITY = 95.83  # m/s, taken from [3]
ALTITUDE = 3355  # m, taken from [3]
FAN_DIAMETER = 84.75 * 2.54 / 100  # m, taken from [3] and converted to meters from inches
RPM = []
J = []

def GenerateMTFLOBlading(Omega: float = 0.,
                         ref_blade_angle: float = np.deg2rad(19)):
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
    blading_parameters = [{"root_LE_coordinate": 0.26035, "rotational_rate": Omega, "ref_blade_angle": ref_blade_angle, ".75R_blade_angle": np.deg2rad(34.3), "blade_count": 3, "radial_stations": [0.21294, 0.53235, 1.0647], 
                                                                                                                                                                        "chord_length": [0.35052, 
                                                                                                                                                                                        0.254, 
                                                                                                                                                                                        0.22098], 
                                                                                                                                                                                        "sweep_angle": [0, 
                                                                                                                                                                                                        np.atan((0.35052 - 0.254) / (0.53235 - 0.21294)), 
                                                                                                                                                                                                        np.atan((0.35052 - 0.22098) / (1.0647 - 0.21294))], 
                                                                                                                                                                                                        "blade_angle": [np.deg2rad(53), 
                                                                                                                                                                                                                        np.deg2rad(32), 
                                                                                                                                                                                                                        np.deg2rad(15)]}]

    # Obtain the parameterizations for the profile sections. 
    local_dir_path = Path('Validation')
    root_fpath = local_dir_path / 'X22_root.dat'
    mid_fpath = local_dir_path / 'X22_mid.dat'
    tip_fpath = local_dir_path / 'X22_tip.dat'

    # Compute parameterization for root airfoil section
    param_class = AirfoilParameterization()
    root_section = param_class.FindInitialParameterization(reference_file=root_fpath,
                                                        plot=False)

    # Compute parameterization for the mid airfoil section
    mid_section = param_class.FindInitialParameterization(reference_file=mid_fpath,
                                                        plot=False)

    # Compute parameterization for the tip airfoil section
    tip_section = param_class.FindInitialParameterization(reference_file=tip_fpath,
                                                        plot=False)

    # Construct blading list
    design_parameters = [[root_section, mid_section, tip_section]]

    return blading_parameters, design_parameters


def GenerateMTFLOInput(blading_parameters,
                       design_parameters):
    """
    Generate MTFLO input file tflow.X22A_validation
    """
    
    fileHandling.fileHandlingMTFLO(case_name=ANALYSIS_NAME).GenerateMTFLOInput(blading_params=blading_parameters,
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
    centerbody_x = (np.array([40.5, 40.4, 39, 36.6, 32.94, 26.75, 25.286, 21.89, 18.494, 17.03, 14.03, 10.98, 7.32, 3.4, 2.196, 0.8, 0.15, 0]) - 3.67) * 2.54 / 100 
    centerbody_y = np.array([5.5, 5.5, 5.6, 5.856, 6.588, 8.15, 8.53, 8.75, 8.53, 8.25, 7.5, 6.4, 5.05, 3.65, 3.0, 2.1, 0.732, 0]) * 2.54 / 100

    # Raw data from Bram:
    #centerbody_x = np.array([56, 45, 43, 40, 34, 32, 30, 25, 22, 19, 15, 11, 6.3, 2.3, 0, 0, 0]) * 2.54 / 100
    #centerbody_y = np.array([5, 5.2, 5.6, 6.2, 7.1, 7.8, 8.5, 8.7, 8.2, 7.5, 6.5, 5.6, 4.4, 3.3, 1.7, 0, 0]) * 2.54 / 100

    plt.figure()
    plt.title("Centre body geometry")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(centerbody_x, centerbody_y)
    plt.plot(centerbody_x, -centerbody_y)
    plt.show()

    plt.figure()
    plt.title("Duct geometry")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(x_duct, y_duct)
    plt.show()

    # Transform the data to the correct format
    # Ensures leading edge data point only occurs once to make sure a smooth spline is constructed, in accordance with the MTFLOW documentation. 
    centerbody_x_complete = np.concatenate((centerbody_x, np.flip(centerbody_x[:-2])), axis=0)
    centerbody_y_complete = np.concatenate((centerbody_y, np.flip(-centerbody_y[:-2])), axis=0)
    xy_centerbody = np.vstack((centerbody_x_complete, centerbody_y_complete)).T

    # --------------------
    # Generate MTSET input file walls.X22A_validation
    # To pass the class input validation, dummy inputs need to be provided
    # --------------------

    params_CB = {"Leading Edge Coordinates": (centerbody_x.min(),0), "Chord Length": (lower_x.max() - lower_x.min())}
    params_duct = {"Leading Edge Coordinates": (x_duct.min(), upper_y[-1]), "Chord Length": (lower_x.max() - lower_x.min())}

    fileHandling().fileHandlingMTSET(params_CB=params_CB,
                                     params_duct=params_duct,
                                     case_name=ANALYSIS_NAME,
                                     external_input=True).GenerateMTSETInput(xy_centerbody=xy_centerbody,
                                                                             xy_duct=xy_duct)

    return xy_centerbody, xy_duct


def RunMTFLOW(oper: dict,
              Omega: float = 0.,
              ref_blade_angle: float = np.deg2rad(19),
              
              ):
    """
    Execute MTFLOW

    Parameters
    ----------

    Returns
    -------
    """

    xy_centerbody, xy_duct = GenerateMTSETGeometry()

    blading_parameters, design_parameters = GenerateMTFLOBlading(Omega,
                                                                 ref_blade_angle)

    GenerateMTFLOInput(blading_parameters,
                       design_parameters)

    MTFLOW_caller(operating_conditions=oper,
                  centrebody_params={},
                  duct_params={},
                  blading_parameters=blading_parameters,
                  design_parameters=design_parameters,
                  analysis_name=ANALYSIS_NAME).caller(debug=True,
                                                      external_inputs=True)
    
    CT, CP, etaP = output_processing(ANALYSIS_NAME).GetCTCPEtaP()

    return CT, CP, etaP


if __name__ == "__main__":

    # Define operating conditions
    oper = {"Inlet_Mach": 0.2,
            "Inlet_Reynolds": 0,
            "N_crit": 9,
            }

    # RunMTFLOW(oper=oper,
    #           )

    CT, CP, etaP = output_processing(ANALYSIS_NAME).GetCTCPEtaP()