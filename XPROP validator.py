""" 
XPROP_validation
===============

Validation of the developed MTFLOW code against an XPROP unswept propeller


References
----------


"""

import numpy as np
from pathlib import Path
from scipy import interpolate
from ambiance import Atmosphere
import matplotlib.pyplot as plt
import os
import time
import sys

# Enable submodel relative imports 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Submodels.Parameterizations import AirfoilParameterization
from Submodels.file_handling import fileHandling
from Submodels.output_handling import output_processing
from Submodels.MTSOL_call import MTSOL_call
from Submodels.MTSET_call import MTSET_call
from Submodels.MTFLO_call import MTFLO_call


def GenerateMTFLOBlading(Omega: float,                        
                         ref_blade_angle: float,
                         ) -> tuple[list]:
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
    - blading_parameters : list
        A list containing dictionaries with the blading parameters.
    - design_parameters : list
        A list containing dictionaries with the design parameters for each radial station.
    """

    # Start defining the MTFLO blading inputs
    blading_parameters = [{"root_LE_coordinate": 0.079128, "rotational_rate": Omega, "ref_blade_angle": ref_blade_angle, ".75R_blade_angle": np.deg2rad(0), "blade_count": 6, "radial_stations": np.array([0.16,
                                                                                                                                                                                                                      0.195,
                                                                                                                                                                                                                      0.23,
                                                                                                                                                                                                                      0.265,
                                                                                                                                                                                                                      0.3,
                                                                                                                                                                                                                      0.335,
                                                                                                                                                                                                                      0.370,
                                                                                                                                                                                                                      0.405,
                                                                                                                                                                                                                      0.440,
                                                                                                                                                                                                                      0.475,
                                                                                                                                                                                                                      0.510,
                                                                                                                                                                                                                      0.545,
                                                                                                                                                                                                                      0.58,
                                                                                                                                                                                                                      0.615,
                                                                                                                                                                                                                      0.650,
                                                                                                                                                                                                                      0.685,
                                                                                                                                                                                                                      0.72,
                                                                                                                                                                                                                      0.755,
                                                                                                                                                                                                                      0.790,
                                                                                                                                                                                                                      0.825,
                                                                                                                                                                                                                      0.860,
                                                                                                                                                                                                                      0.895,
                                                                                                                                                                                                                      0.930,
                                                                                                                                                                                                                      0.965,
                                                                                                                                                                                                                      1.0
                                                                                                                                                                                                                      ]) * 0.20320, 
                                                                                                                                                                                    "chord_length": np.array([32.5741,
                                                                                                                                                                                                              32.1,
                                                                                                                                                                                                              31.6261,
                                                                                                                                                                                                              31.1506,
                                                                                                                                                                                                              30.6776,
                                                                                                                                                                                                              30.2311,
                                                                                                                                                                                                              29.8529,
                                                                                                                                                                                                              29.5839,
                                                                                                                                                                                                              29.4857,
                                                                                                                                                                                                              29.5811,
                                                                                                                                                                                                              29.8379,
                                                                                                                                                                                                              30.176,
                                                                                                                                                                                                              30.5150,
                                                                                                                                                                                                              30.7752,
                                                                                                                                                                                                              30.9372,
                                                                                                                                                                                                              31.0523,
                                                                                                                                                                                                              31.1658,
                                                                                                                                                                                                              31.1599,
                                                                                                                                                                                                              30.7789,
                                                                                                                                                                                                              29.7761,
                                                                                                                                                                                                              28.0360,
                                                                                                                                                                                                              25.5175,
                                                                                                                                                                                                              22.2309,
                                                                                                                                                                                                              18.5338,
                                                                                                                                                                                                              13.8289]) / 1000, 
                                                                                                                                                                                    "blade_angle": np.deg2rad(np.array([26.82,
                                                                                                                                                                                                                        24.83,
                                                                                                                                                                                                                        22.89,
                                                                                                                                                                                                                        20.99,
                                                                                                                                                                                                                        19.15,
                                                                                                                                                                                                                        17.38,
                                                                                                                                                                                                                        15.62,
                                                                                                                                                                                                                        13.87,
                                                                                                                                                                                                                        12.14,
                                                                                                                                                                                                                        10.43,
                                                                                                                                                                                                                        8.73,
                                                                                                                                                                                                                        7.05,
                                                                                                                                                                                                                        5.39,
                                                                                                                                                                                                                        3.75,
                                                                                                                                                                                                                        2.12,
                                                                                                                                                                                                                        0.6,
                                                                                                                                                                                                                        -0.73,
                                                                                                                                                                                                                        -1.87,
                                                                                                                                                                                                                        -2.91,
                                                                                                                                                                                                                        -3.90,
                                                                                                                                                                                                                        -4.80,
                                                                                                                                                                                                                        -5.67,
                                                                                                                                                                                                                        -6.53,
                                                                                                                                                                                                                        -7.37,
                                                                                                                                                                                                                        -8.00]))}]
    # Define the sweep angles
    # Note that this is approximate, since the rotation of the chord line is not completely accurate when rotating a complete profile
    sweep_angle = np.zeros_like(blading_parameters[0]["chord_length"])
    root_rotation_angle = np.pi / 2 - (np.deg2rad(68.18) + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0][".75R_blade_angle"])
    root_LE = blading_parameters[0]["root_LE_coordinate"] # The location of the root LE is arbitrary for computing the sweep angles.
    root_mid_chord = root_LE + (0.4226 / 2) * np.cos(root_rotation_angle)
    for i in range(len(blading_parameters[0]["chord_length"])):
        rotation_angle = np.pi/2 - (blading_parameters[0]["blade_angle"][i] + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0][".75R_blade_angle"])

        # Compute sweep such that the midchord line is constant.
        local_LE = root_mid_chord - (blading_parameters[0]["chord_length"][i] / 2) * np.cos(rotation_angle)
        sweep_angle[i] = np.atan((local_LE - root_LE) / (blading_parameters[0]["radial_stations"][i]))      
    blading_parameters[0]["sweep_angle"] = sweep_angle
    
    # Create plot of the propeller blade
    # Chord lengths are approximate due to the incomplete rotation implementation. 
    plt.figure()
    plt.xlabel("Axial Location [m]")
    plt.ylabel("Radial location [m]")
    plt.title("Propeller Blade Input Geometry")
    plt.grid()
    x_LE_arr = np.zeros_like(blading_parameters[0]["chord_length"])
    x_TE_arr = np.zeros_like(x_LE_arr)
    x_mid_arr = np.zeros_like(x_LE_arr)

    for section in range(len(blading_parameters[0]["radial_stations"])):
        rotation_angle = np.pi/2 - (blading_parameters[0]["blade_angle"][section] + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0][".75R_blade_angle"])
        x_LE = root_LE + blading_parameters[0]["radial_stations"][section] * np.tan(blading_parameters[0]["sweep_angle"][section])
        x_TE = x_LE + blading_parameters[0]["chord_length"][section] * np.cos(rotation_angle)
        x_mid_arr[section] = (x_LE + x_TE) / 2
        x_LE_arr[section] = x_LE
        x_TE_arr[section] = x_TE
        plt.plot([x_LE, x_TE], [blading_parameters[0]["radial_stations"][section], blading_parameters[0]["radial_stations"][section]], label=f"r={round(blading_parameters[0]["radial_stations"][section], 2)} m")

    plt.plot(x_LE_arr, blading_parameters[0]["radial_stations"], "-.k")
    plt.plot(x_TE_arr, blading_parameters[0]["radial_stations"], "-.k")
    plt.plot(x_mid_arr, blading_parameters[0]["radial_stations"], "-.k")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

    # Define filepaths for all stations
    local_dir_path = Path('XPROP_validation')     

    A_fpath = local_dir_path / "N250_airfoilStation1.dat"
    B_fpath = local_dir_path / "N250_airfoilStation2.dat"
    C_fpath = local_dir_path / "N250_airfoilStation3.dat"
    D_fpath = local_dir_path / "N250_airfoilStation4.dat"
    E_fpath = local_dir_path / "N250_airfoilStation5.dat"
    F_fpath = local_dir_path / "N250_airfoilStation6.dat"
    G_fpath = local_dir_path / "N250_airfoilStation7.dat"
    H_fpath = local_dir_path / "N250_airfoilStation8.dat"
    I_fpath = local_dir_path / "N250_airfoilStation9.dat"
    J_fpath = local_dir_path / "N250_airfoilStation10.dat"
    K_fpath = local_dir_path / "N250_airfoilStation11.dat"
    L_fpath = local_dir_path / "N250_airfoilStation12.dat"
    M_fpath = local_dir_path / "N250_airfoilStation13.dat"
    N_fpath = local_dir_path / "N250_airfoilStation14.dat"
    O_fpath = local_dir_path / "N250_airfoilStation15.dat"
    P_fpath = local_dir_path / "N250_airfoilStation16.dat"
    Q_fpath = local_dir_path / "N250_airfoilStation17.dat"
    R_fpath = local_dir_path / "N250_airfoilStation18.dat"
    S_fpath = local_dir_path / "N250_airfoilStation19.dat"
    T_fpath = local_dir_path / "N250_airfoilStation20.dat"
    U_fpath = local_dir_path / "N250_airfoilStation21.dat"
    V_fpath = local_dir_path / "N250_airfoilStation22.dat"
    W_fpath = local_dir_path / "N250_airfoilStation23.dat"
    X_fpath = local_dir_path / "N250_airfoilStation24.dat"
    Y_fpath = local_dir_path / "N250_airfoilStation25.dat"

    # Compute parameterization for the airfoil stations
    A_section = AirfoilParameterization().FindInitialParameterization(reference_file=A_fpath,
                                                        plot=True)
    B_section = AirfoilParameterization().FindInitialParameterization(reference_file=B_fpath,
                                                        plot=True)
    C_section = AirfoilParameterization().FindInitialParameterization(reference_file=C_fpath,
                                                        plot=True)
    D_section = AirfoilParameterization().FindInitialParameterization(reference_file=D_fpath,
                                                        plot=True)
    E_section = AirfoilParameterization().FindInitialParameterization(reference_file=E_fpath,
                                                        plot=True)
    F_section = AirfoilParameterization().FindInitialParameterization(reference_file=F_fpath,
                                                        plot=True)
    G_section = AirfoilParameterization().FindInitialParameterization(reference_file=G_fpath,
                                                        plot=True)
    H_section = AirfoilParameterization().FindInitialParameterization(reference_file=H_fpath,
                                                        plot=True)
    I_section = AirfoilParameterization().FindInitialParameterization(reference_file=I_fpath,
                                                        plot=True)
    J_section = AirfoilParameterization().FindInitialParameterization(reference_file=J_fpath,
                                                        plot=True)
    K_section = AirfoilParameterization().FindInitialParameterization(reference_file=K_fpath,
                                                        plot=True)
    L_section = AirfoilParameterization().FindInitialParameterization(reference_file=L_fpath,
                                                        plot=True)
    M_section = AirfoilParameterization().FindInitialParameterization(reference_file=M_fpath,
                                                        plot=True)
    N_section = AirfoilParameterization().FindInitialParameterization(reference_file=N_fpath,
                                                        plot=True)
    O_section = AirfoilParameterization().FindInitialParameterization(reference_file=O_fpath,
                                                        plot=True)
    P_section = AirfoilParameterization().FindInitialParameterization(reference_file=P_fpath,
                                                        plot=True)
    Q_section = AirfoilParameterization().FindInitialParameterization(reference_file=Q_fpath,
                                                        plot=True)
    R_section = AirfoilParameterization().FindInitialParameterization(reference_file=R_fpath,
                                                        plot=True)
    S_section = AirfoilParameterization().FindInitialParameterization(reference_file=S_fpath,
                                                        plot=True)
    T_section = AirfoilParameterization().FindInitialParameterization(reference_file=T_fpath,
                                                        plot=True)
    U_section = AirfoilParameterization().FindInitialParameterization(reference_file=U_fpath,
                                                        plot=True)
    V_section = AirfoilParameterization().FindInitialParameterization(reference_file=V_fpath,
                                                        plot=True)
    W_section = AirfoilParameterization().FindInitialParameterization(reference_file=W_fpath,
                                                        plot=True)
    X_section = AirfoilParameterization().FindInitialParameterization(reference_file=X_fpath,
                                                        plot=True)
    Y_section = AirfoilParameterization().FindInitialParameterization(reference_file=Y_fpath,
                                                        plot=True)
    # Construct blading list
    design_parameters = [[A_section, B_section, C_section, D_section, E_section, F_section, G_section, H_section, I_section, J_section, K_section, L_section, M_section, N_section, O_section, P_section, Q_section, R_section, S_section, T_section, U_section, V_section, W_section, X_section, Y_section]]

    return blading_parameters, design_parameters


def GenerateMTFLOInput(blading_parameters,
                       design_parameters) -> None:
    """
    Generate the MTFLO input file tflow.X22A_validation

    Parameters
    ----------
   - blading_parameters : list
        A list containing dictionaries with the blading parameters.
    - design_parameters : list
        A list containing dictionaries with the design parameters for each radial station.

    Returns
    -------
    None
    """
    
    fileHandling.fileHandlingMTFLO(case_name=ANALYSIS_NAME,
                                   ref_length=FAN_DIAMETER).GenerateMTFLOInput(blading_params=blading_parameters,
                                                                               design_params=design_parameters)


def GenerateMTSETGeometry() -> None:
    """
    Generate the duct and center body geometry. Uses a combination of analytical representation, and smoothing interpolations to obtain the axisymmetric geometries. 

    Returns
    -------
    None
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
    # Data taken from a graph digitized by Bram Meijerink.
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
    centerbody_x = np.flip(np.array([66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 43, 40, 34, 32, 30, 25, 22, 19, 15, 11, 6.3, 2.3, 0.7, 0.5, 0]) * 2.54 / 100) - 3.67 * 2.54 / 100
    centerbody_y = np.flip(np.array([1.72, 1.72, 1.73, 1.74, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.6, 6.2, 7.1, 7.8, 8.5, 8.7, 8.2, 7.5, 6.5, 5.6, 4.4, 3.3, 1.7, 1.0, 0]) * 2.54 / 100)
    
    # centerbody_x = np.flip(np.array([56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 43, 40, 34, 32, 30, 25, 22, 19, 15, 11, 6.3, 2.3, 0.7, 0.5, 0]) * 2.54 / 100) - 3.67 * 2.54 / 100
    # centerbody_y = np.flip(np.array([3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.6, 6.2, 7.1, 7.8, 8.5, 8.7, 8.2, 7.5, 6.5, 5.6, 4.4, 3.3, 1.7, 1.0, 0]) * 2.54 / 100)

    # Perform smoothing interpolation on the centerbody geometry
    interpolated_centerbody_x = centerbody_x[0] + ((1 - np.cos(np.linspace(0, np.pi, 30))) / 2) * (centerbody_x[-1] - centerbody_x[0])  #  cosine spacing for increased resolution at LE and TE
    

    interpolated_centerbody_y = interpolate.UnivariateSpline(centerbody_x,
                                                             centerbody_y,
                                                             k=3,
                                                             s=5
                                                             )(interpolated_centerbody_x)  
    
    # plt.figure()
    # plt.title("Centre body geometry")
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.plot(centerbody_x, centerbody_y)
    # plt.plot(interpolated_centerbody_x, interpolated_centerbody_y)
    # plt.show()

    # Transform the data to the correct format
    # Ensures leading edge data point only occurs once to make sure a smooth spline is constructed, in accordance with the MTFLOW documentation.    
    centerbody_x_complete = np.concatenate((np.flip(interpolated_centerbody_x), interpolated_centerbody_x[:-2]), axis=0)
    centerbody_y_complete = np.concatenate((np.flip(interpolated_centerbody_y), -interpolated_centerbody_y[:-2]), axis=0)
    xy_centerbody = np.vstack((centerbody_x_complete, centerbody_y_complete)).T    

    # Generate MTSET input file walls.X22A_validation
    params_CB = {"Leading Edge Coordinates": (centerbody_x.min(),0), "Chord Length": (lower_x.max() - lower_x.min())}
    params_duct = {"Leading Edge Coordinates": (x_duct.min(), upper_y[-1]), "Chord Length": (lower_x.max() - lower_x.min())}

    fileHandling().fileHandlingMTSET(params_CB=params_CB,
                                     params_duct=params_duct,
                                     case_name=ANALYSIS_NAME,
                                     ref_length=FAN_DIAMETER,
                                     external_input=True).GenerateMTSETInput(xy_centerbody=xy_centerbody,
                                                                             xy_duct=xy_duct)


def ChangeOMEGA(omega) -> None:
    """
    Rather than regenerating the tflow.xxx file from scratch, simply change omega in the tflow.xxx file. 

    Parameters
    ----------
    - omega : float
        The non-dimensional rotational speed to be entered into the tflow.xxx input file. 

    Returns
    -------
    None
    """

    with open(f"tflow.{ANALYSIS_NAME}", "r") as file:
        lines = file.readlines()

    omega_line = 11
    updated_omega = f"{omega} \n"
    lines[omega_line] = updated_omega

    with open(f"tflow.{ANALYSIS_NAME}", "w") as file:
        file.writelines(lines)


def ExecuteParameterSweep(OMEGA: np.ndarray[float],
                          inlet_mach: np.ndarray[float],
                          reynolds_inlet: np.ndarray[float],
                          reference_angle: float,
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a parameter sweep over a range of OMEGA, inlet Mach numbers, and Reynolds numbers.

    Parameters
    ----------
    - OMEGA : np.ndarray[float]
        Array of non-dimensional rotational speeds of the rotor.
    - inlet_mach : np.ndarray[float]
        Array of inlet Mach numbers.
    - reynolds_inlet : np.ndarray[float]
        Array of inlet Reynolds numbers.

    Returns
    -------
    - CT_outputs : np.ndarray[float]
        Array of thrust coefficients.
    - CP_outputs : np.ndarray[float]
        Array of power coefficients.
    - EtaP_outputs : np.ndarray[float]
        Array of propulsive efficiencies.
    """

    # Create the MTSET geometry and write the input file walls.ANALYSIS_NAME
    GenerateMTSETGeometry()

    # Construct the MTFLO blading using omega=0 and reference blade angle. 
    # Perform parameterization can be optionally set to true in case different profiles are used compared to the default inputs
    blading_parameters, design_parameters = GenerateMTFLOBlading(Omega=0,
                                                                 ref_blade_angle=reference_angle)
    
    # Change working directory to the submodels folder
    try:
        current_dir = os.getcwd()
        subfolder_path = os.path.join(current_dir, 'Submodels')
        os.chdir(subfolder_path)
    except OSError as e:
        raise OSError from e
    
    # Generate the MTFLO input file
    GenerateMTFLOInput(blading_parameters,
                       design_parameters)
    
    # # Create the grid
    # MTSET_call(analysis_name=ANALYSIS_NAME,
    #            streamwise_points=250,
    #            ).caller()
    
    # Perform analysis for all omega, Mach, and Re combinations defined at the top of the file
    CT_outputs = np.zeros_like(OMEGA)
    CP_outputs = np.zeros_like(OMEGA)
    EtaP_outputs = np.zeros_like(OMEGA)

    for i in range(len(OMEGA)):
        # Update the blade parameters to the correct omega 
        ChangeOMEGA(OMEGA[i])      

        # Create the grid
        MTSET_call(analysis_name=ANALYSIS_NAME,
                   streamwise_points=200
                   ).caller()
        
        # Wait for the grid file to be loaded
        time.sleep(1)

        #Load in the blade row(s) from MTFLO 
        MTFLO_call(ANALYSIS_NAME).caller() 

        # Wait to ensure blade rows are loaded in
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
        
        # Wait to ensure outpit files have been loaded in
        time.sleep(1)

        # Collect outputs from the forces.xxx file
        CT, CP, etaP = output_processing(ANALYSIS_NAME).GetCTCPEtaP()
        print(f"Omega: {OMEGA[i]}, CT: {CT}, CP: {CP}, etaP: {etaP}")

        CT_outputs[i] = CT
        CP_outputs[i] = CP 
        EtaP_outputs[i] = etaP
    
    # Return back to current dir
    os.chdir(current_dir)
    
    return CT_outputs, CP_outputs, EtaP_outputs


if __name__ == "__main__":

    # First we define some constants and the operating conditions which will be analysed
    REFERENCE_BLADE_ANGLE = np.array([np.deg2rad(30), np.deg2rad(45)])  # radians, converted from degrees
    ANALYSIS_NAME = "XPROP_validation"  # Analysis name for MTFLOW
    ALTITUDE = 0  # m
    FAN_DIAMETER = 0.203200

    # Advance ratio range to be used for the validation, together with freestream velocity.
    J = np.flip(np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]))  # -
    FREESTREAM_VELOCITY = np.ones_like(J) * 28  # m/s, tweaked to get acceptable values of RPS/OMEGA for the advance ratio range considered. 

    # Compute the rotational speed of the rotor in rotations per second
    RPS = FREESTREAM_VELOCITY / (J * FAN_DIAMETER)  # Hz
    print(f"RPM (Should be between 1200-2590 RPM) [-]: {RPS * 60}")

    # Use the calculated rotational speed to obtain the non-dimensional Omega used as input into MTFLOW
    OMEGA = (RPS * 2 * np.pi) * FAN_DIAMETER / FREESTREAM_VELOCITY
    print(f"Omega [-]: {OMEGA}")

    # Construct atmosphere object to obtain the atmospheric properties at the cruise altitude
    # These properties can then be used to compute the inlet mach number and reynolds number
    atmosphere = Atmosphere(ALTITUDE)
    inlet_mach = (FREESTREAM_VELOCITY / atmosphere.speed_of_sound)
    print(f"Mach [-]: {inlet_mach}")
    reynolds_inlet = (FREESTREAM_VELOCITY * FAN_DIAMETER / (atmosphere.kinematic_viscosity))
    print(f"Reynolds [-]: {reynolds_inlet}")

    # Initialize output dictionaries and perform parameter sweep. 
    CT = {}
    CP = {}
    etaP = {}
    for i in range(len(REFERENCE_BLADE_ANGLE)):
        print(f"Analysing beta_{75}={round(np.rad2deg(REFERENCE_BLADE_ANGLE[i]), 2)} deg")
        CT_out, CP_out, eta_out = ExecuteParameterSweep(OMEGA=OMEGA,
                                                        inlet_mach=inlet_mach,
                                                        reynolds_inlet=reynolds_inlet,
                                                        reference_angle=REFERENCE_BLADE_ANGLE[i])
        
        key = f"beta_75 = {REFERENCE_BLADE_ANGLE[i]}"
        CT[key] = CT_out
        CP[key] = CP_out
        etaP[key] = eta_out

    