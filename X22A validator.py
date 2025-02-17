""" 
X22A_validation
===============

References
----------
[1] - https://ntrs.nasa.gov/api/citations/19670025554/downloads/19670025554.pdf 
[2] - https://apps.dtic.mil/sti/tr/pdf/AD0447814.pdf?form=MG0AV3 

"""

import numpy as np
from pathlib import Path
from scipy import interpolate

from Submodels.Parameterizations import AirfoilParameterization
from MTFLOW_caller import MTFLOW_caller
from Submodels.file_handling import fileHandling

REFERENCE_BLADE_ANGLE = np.radians(19)  # radians
ANALYSIS_NAME = "X22A_validation"  # Analysis name for MTFLOW

# --------------------
# Generate MTFLO blading
# [2] mentions the use of a modified NASA 001-64 profile. We scale this profile to have the correct thickness for each radial station, 
# but that is the limit of approximations we can make with the limited data available. 
#
# The blading parameters are based on Figure 3 in [1].
# --------------------

# Start defining the MTFLO blading inputs
# The rotational rate will be redefined later on when performing the MTFLOW analyses
blading_parameters = [{"root_LE_coordinate": 0.3556, "rotational_rate": 0., "blade_count": 3, "radial_stations": [0.21294, 0.53235, 1.0647], 
                                                                                                                    "chord_length": [0.35052, 
                                                                                                                                     0.254, 
                                                                                                                                     0.22098], 
                                                                                                                                     "sweep_angle": [0, 
                                                                                                                                                     np.atan2((0.35052 - 0.254), (0.53235 - 0.21294)), 
                                                                                                                                                     np.atan2((0.254 - 0.22098), (1.0647 - 0.53235))], 
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

# --------------------
# Generate MTFLO input file tflow.X22A_validation
# --------------------

file_handler = fileHandling()
file_handler.fileHandlingMTFLO(case_name=ANALYSIS_NAME).GenerateMTFLOInput(blading_params=blading_parameters,
                                                                           design_params=design_parameters)

# --------------------
# Generate duct geometry
# --------------------

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
x_duct = np.concatenate((upper_x, lower_x), axis=0)
y_duct = np.concatenate((upper_y, lower_y), axis=0)
xy_duct = np.vstack((x_duct, y_duct)).T

# --------------------
# Generate centre body geometry
# --------------------

# Data taken from a digitized graph from Bram Meijerink
centerbody_x = np.array([46, 45, 43, 40, 34, 32, 30, 25, 22, 19, 15, 11, 6.3, 2.3, 0, 0, 0]) * 2.54 / 100
centerbody_y = np.array([5, 5.2, 5.6, 6.2, 7.1, 7.8, 8.5, 8.7, 8.2, 7.5, 6.5, 5.6, 4.4, 3.3, 1.7, 0, 0]) * 2.54 / 100

# Smooth the y data to obtain a more physical geometry
x = np.linspace(centerbody_x.min(), centerbody_x.max(), 30)
centerbody_y_interpd = interpolate.UnivariateSpline(np.flip(centerbody_x), np.flip(centerbody_y), s=0.01)(x)

centerbody_x_reconstructed = np.concatenate((np.flip(x), x), axis=0)
centerbody_y_reconstructed = np.concatenate((np.flip(centerbody_y_interpd), -centerbody_y_interpd), axis=0)

# Transform the data to the correct format
xy_centerbody = np.vstack((centerbody_x_reconstructed, centerbody_y_reconstructed)).T

# --------------------
# Generate MTSET input file walls.X22A_validation
# To pass the class input validation, dummy inputs need to be provided
# --------------------

params_CB = {"Leading Edge Coordinates": (centerbody_x.min(),0), "Chord Length": (lower_x.max() - lower_x.min())}
params_duct = {"Leading Edge Coordinates": (x_duct.min(), y_duct.max()), "Chord Length": (lower_x.max() - lower_x.min())}

file_handler.fileHandlingMTSET(params_CB=params_CB,
                               params_duct=params_duct,
                               case_name=ANALYSIS_NAME,
                               external_input=True).GenerateMTSETInput(xy_centerbody=xy_centerbody,
                                                                       xy_duct=xy_duct)
