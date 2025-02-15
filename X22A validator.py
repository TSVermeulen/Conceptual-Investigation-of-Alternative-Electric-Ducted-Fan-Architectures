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

from Submodels.Parameterizations import AirfoilParameterization

# --------------------
# Generate MTFLO blading
# --------------------

# Start defining the MTFLO blading inputs
blading_parameters = [{"root_LE_coordinate": 0.3556, "rotational_rate": 0.75, "blade_count": 3, "radial_stations": [0.21294, 
                                                                                                                    0.53235, 
                                                                                                                    1.0647], 
                                                                                                                    "chord_length": [0.35052, 
                                                                                                                                     0.254, 
                                                                                                                                     0.22098], 
                                                                                                                                     "sweep_angle": [0, 
                                                                                                                                                     np.atan2((0.35052 - 0.254), (0.53235 - 0.21294)), 
                                                                                                                                                     np.atan2((0.254 - 0.22098), (1.0647 - 0.53235))], 
                                                                                                                                                     "twist_angle": [np.deg2rad(53), 
                                                                                                                                                                     np.deg2rad(32), 
                                                                                                                                                                     np.deg2rad(15)]}]

# Obtain the parameterizations for the profile sections. [2] mentions the use of a modified NASA 001-64 profile. 
# We scale this profile to have the correct thickness for each radial station, but that is the limit of approximations we can make with the limited data available. 
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

# Construct 

