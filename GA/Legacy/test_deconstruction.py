"""
Test of the DeconstructDesignVector method in problem definition
"""

import config
import unittest
import numpy as np

class DesignVectorAccessor:
    """ Simple class to provide efficient access to design vector elements without repeated string formatting. """
    
    def __init__(self,
                 x_dict: dict[str, float|int],
                 x_keys: list[str]) -> None:
        self.x_dict = x_dict
        self.x_keys = x_keys

    
    def get(self,
            base_idx: int,
            offset: int = 0,
            default=None) -> float|int:
        """ Get value at base_idx + offset position"""
        try:
            key = self.x_keys[base_idx + offset]
            return self.x_dict[key]
        except (IndexError, KeyError) as err:
            if default is not None:
                return default
            raise KeyError(f"Design vector key at position {base_idx + offset} missing") from err

def DeconstructDesignVector(x: dict[str, float|int]):
    """
        Decompose the design vector x into dictionaries of all the design variables to match the expected input formats for 
        the MTFLOW code interface. 
        The design vector has the format: [centerbody, duct, blades]
        
        Parameters
        ----------
        - x : dict[str, float|int]
            Dictionary representation of the design vector received from pymoo.

        Returns
        -------
        None
    """

    # Create a design vector accessor instance for more efficient access

    vector = DesignVectorAccessor(x, list(x.keys()))
    vget = vector.get  # Create a local alias for vector.get - this is marginally quicker per call

    # Define a helper function to compute parameter b_8 using the mapping design variable
    def Getb8(b_8_map: float, 
              r_le: float, 
              x_t: float, 
              y_t: float) -> float:
        """
        Helper function to compute the bezier parameter b_8 using the mapping parameter 0 <= b_8_map <= 1
        """

        term = -2 * r_le * x_t / 3
        sqrt_term = 0 if term <= 0 else np.sqrt(term)

        return b_8_map * min(y_t, sqrt_term)
    
    # Define a pointer to count the number of variable parameters
    centerbody_designvar_count = 8
    duct_designvar_count = 17
    if config.OPTIMIZE_CENTERBODY:
        centerbody_start = 0
        duct_start = centerbody_designvar_count 
        stage_start = centerbody_designvar_count + (duct_designvar_count if config.OPTIMIZE_DUCT else 0)
    else:
        duct_start = 0
        stage_start = duct_designvar_count if config.OPTIMIZE_DUCT else 0

    # Deconstruct the centerbody values if it's variable.
    # If the centerbody is constant, read in the centerbody values from config.
    # Note that if the centerbody is variable, we keep the LE coordinate fixed, as the LE coordinate of the duct would already be free to move. 
    if config.OPTIMIZE_CENTERBODY:
        idx = centerbody_start
        centerbody_variables = {"b_0": 0.,
                                     "b_2": 0., 
                                     "b_8": Getb8(vget(idx), vget(idx, 5), vget(idx, 2), vget(idx, 3)),
                                     "b_15": vget(idx, 1),
                                     "b_17": 0.,
                                     "x_t": vget(idx, 2),
                                     "y_t": vget(idx, 3),
                                     "x_c": 0.,
                                     "y_c": 0.,
                                     "z_TE": 0.,
                                     "dz_TE": vget(idx, 4),
                                     "r_LE": vget(idx, 5),
                                     "trailing_wedge_angle": vget(idx, 6),
                                     "trailing_camberline_angle": 0.,
                                     "leading_edge_direction": 0., 
                                     "Chord Length": vget(idx, 7),
                                     "Leading Edge Coordinates": (0., 0.)}
    else:
        centerbody_variables = config.CENTERBODY_VALUES

    # Deconstruct the duct values if it's variable.
    # If the duct is constant, read in the duct values from config.
    if config.OPTIMIZE_DUCT:
        idx = duct_start
        duct_variables = {"b_0": vget(idx),
                               "b_2": vget(idx, 1), 
                               "b_8": Getb8(vget(idx, 2), vget(idx, 11), vget(idx, 5), vget(idx, 6)),
                               "b_15": vget(idx, 3),
                               "b_17": vget(idx, 4),
                               "x_t": vget(idx, 5),
                               "y_t": vget(idx, 6),
                               "x_c": vget(idx, 7),
                               "y_c": vget(idx, 8),
                               "z_TE": vget(idx, 9),
                               "dz_TE": vget(idx, 10),
                               "r_LE": vget(idx, 11),
                               "trailing_wedge_angle": vget(idx, 12),
                               "trailing_camberline_angle": vget(idx, 13),
                               "leading_edge_direction": vget(idx, 14), 
                               "Chord Length": vget(idx, 15),
                               "Leading Edge Coordinates": (vget(idx, 16), 0)}
        print(duct_variables)
    else:
        duct_variables = config.DUCT_VALUES
                
    # Deconstruct the rotorblade parameters if they are variable.
    # If the rotorblade parameters are constant, read in the parameters from config.
    blade_design_parameters = []
    idx = stage_start
    for i in range(config.NUM_STAGES):
        # Initiate empty list for each stage
        stage_design_parameters = []
        if config.OPTIMIZE_STAGE[i]:
            # If the stage is to be optimized, read in the design vector for the blade profiles
            for _ in range(config.NUM_RADIALSECTIONS[i]):
                # Loop over the number of radial sections and append each section to stage_design_parameters
                section_parameters = {"b_0": vget(idx),
                                    "b_2": vget(idx, 1), 
                                    "b_8": Getb8(vget(idx, 2), vget(idx, 11), vget(idx, 5), vget(idx, 6)), 
                                    "b_15": vget(idx, 3),
                                    "b_17": vget(idx, 4),
                                    "x_t": vget(idx, 5),
                                    "y_t": vget(idx, 6),
                                    "x_c": vget(idx, 7),
                                    "y_c": vget(idx, 8),
                                    "z_TE": vget(idx, 9),
                                    "dz_TE": vget(idx, 10),
                                    "r_LE": vget(idx, 11),
                                    "trailing_wedge_angle": vget(idx, 12),
                                    "trailing_camberline_angle": vget(idx, 13),
                                    "leading_edge_direction": vget(idx, 14)}
                idx += 15
                stage_design_parameters.append(section_parameters)
        else:
            # If the stage is meant to be constant, read it in from config. 
            stage_design_parameters = config.STAGE_DESIGN_VARIABLES[i]
        # Write the stage nested list to blade_design_parameters
        blade_design_parameters.append(stage_design_parameters)

    blade_blading_parameters = []
    blade_diameters = []
    for i in range(config.NUM_STAGES):
        # Initiate empty list for each stage
        stage_blading_parameters = {}
        if config.OPTIMIZE_STAGE[i]:
            # If the stage is to be optimized, read in the design vector for the blading parameters
            stage_blading_parameters["root_LE_coordinate"] = vget(idx)
            stage_blading_parameters["ref_blade_angle"] = config.REFERENCE_BLADE_ANGLES[i]
            stage_blading_parameters["reference_section_blade_angle"] = vget(idx, 2)
            stage_blading_parameters["blade_count"] = int(round(vget(idx, 1)))
            stage_blading_parameters["radial_stations"] = np.linspace(0, 1, config.NUM_RADIALSECTIONS[i]) * vget(idx, 3)  # Radial stations are defined as fraction of blade radius * local radius
            blade_diameters.append(vget(idx, 3) * 2)

            # Initialize sectional blading parameter lists
            stage_blading_parameters["chord_length"] = [None] * config.NUM_RADIALSECTIONS[i]
            stage_blading_parameters["sweep_angle"] = [None] * config.NUM_RADIALSECTIONS[i]
            stage_blading_parameters["blade_angle"] = [None] * config.NUM_RADIALSECTIONS[i]

            base_idx = idx + 4
            for j in range(config.NUM_RADIALSECTIONS[i]):
                # Loop over the number of radial sections and write their data to the corresponding lists
                stage_blading_parameters["chord_length"][j]= vget(base_idx, j)
                stage_blading_parameters["sweep_angle"][j] = vget(base_idx, config.NUM_RADIALSECTIONS[i] + j)
                stage_blading_parameters["blade_angle"][j] = vget(base_idx, config.NUM_RADIALSECTIONS[i] * 2 + j)
            idx = base_idx + 3 * config.NUM_RADIALSECTIONS[i]               
        else:
            stage_blading_parameters = config.STAGE_BLADING_PARAMETERS[i]
            blade_diameters.append(config.BLADE_DIAMETERS[i])
            
        # Append the stage blading parameters to the main list
        blade_blading_parameters.append(stage_blading_parameters)
        
    # Write the reference length for MTFLOW
    Lref = blade_diameters[0]

    return centerbody_variables, duct_variables, blade_blading_parameters, blade_design_parameters, blade_diameters, Lref


CB = config.CENTERBODY_VALUES
centerbody_vector = [CB['b_8'] / min(CB['y_t'], np.sqrt(-2 * CB['r_LE'] * CB['x_t'] / 3)),
                     CB['b_15'],
                     CB['x_t'],
                     CB['y_t'],
                     CB['dz_TE'],
                     CB['r_LE'],
                     CB['trailing_wedge_angle'],
                     CB['Chord Length']]
centerbody_dict = {f'x{i}': centerbody_vector[i] for i in range(len(centerbody_vector))}

duct = config.DUCT_VALUES
duct_vector = [duct["b_0"],
               duct["b_2"],
               duct["b_8"] /  min(duct['y_t'], np.sqrt(-2 * duct['r_LE'] * duct['x_t'] / 3)),
               duct["b_15"],
               duct["b_17"],
               duct["x_t"],
               duct["y_t"],
               duct["x_c"],
               duct["y_c"],
               duct["z_TE"],
               duct["dz_TE"],
               duct["r_LE"],
               duct["trailing_wedge_angle"],
               duct["trailing_camberline_angle"],
               duct["leading_edge_direction"],
               duct["Chord Length"],
               duct["Leading Edge Coordinates"][0]]
duct["Leading Edge Coordinates"] = (duct["Leading Edge Coordinates"][0], 0)
duct_dict = {f'x{i}': duct_vector[i] for i in range(len(duct_vector))}


rotor_design = config.STAGE_DESIGN_VARIABLES[0]
rotor_blading = config.STAGE_BLADING_PARAMETERS[0]
rotor_blading.pop("rotational_rate")

rotor_vector = []
for dict in rotor_design:
    section_vector = [dict["b_0"],
                      dict["b_2"],
                      dict["b_8"] /  min(dict['y_t'], np.sqrt(-2 * dict['r_LE'] * dict['x_t'] / 3)),
                      dict["b_15"],
                      dict["b_17"],
                      dict["x_t"],
                      dict["y_t"],
                      dict["x_c"],
                      dict["y_c"],
                      dict["z_TE"],
                      dict["dz_TE"],
                      dict["r_LE"],
                      dict["trailing_wedge_angle"],
                      dict["trailing_camberline_angle"],
                      dict["leading_edge_direction"]]
    rotor_vector += section_vector

rotor_vector += [rotor_blading["root_LE_coordinate"],
                 rotor_blading['blade_count'],
                 rotor_blading['reference_section_blade_angle'],
                 max(rotor_blading["radial_stations"])]

station_data = [chord for chord in rotor_blading["chord_length"]]
rotor_vector += station_data
station_data = [sweep for sweep in rotor_blading["sweep_angle"]]
rotor_vector += station_data
station_data = [blade for blade in rotor_blading["blade_angle"]]
rotor_vector += station_data

rotor_dict = {f'x{i}': rotor_vector[i] for i in range(len(rotor_vector))}


config.OPTIMIZE_CENTERBODY = False
config.OPTIMIZE_DUCT = True
config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)
config.OPTIMIZE_STAGE[0] = [True]
duct_rotor_vector = duct_vector + rotor_vector
duct_rotor_dict = {f'x{i}': duct_rotor_vector[i] for i in range(len(duct_rotor_vector))}

print(duct)
_, decon_duct, blading, _, _, _ = DeconstructDesignVector(duct_rotor_dict)
print(duct)
print(decon_duct)
print(rotor_blading)
print(blading)
input()

class TestDeconstructFunction(unittest.TestCase):
    
    def test_centerbody_equality(self):
        self.maxDiff = None
        config.OPTIMIZE_CENTERBODY = True
        config.OPTIMIZE_DUCT = False
        config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)
        deconstructed_cb, _, _, _, _, _ = DeconstructDesignVector(centerbody_dict)
        self.assertDictEqual(CB, deconstructed_cb)

    
    def test_duct_equality(self):
        self.maxDiff = None
        config.OPTIMIZE_DUCT = True
        config.OPTIMIZE_CENTERBODY = False
        config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)
        _, deconstructed_duct, _, _, _, _ = DeconstructDesignVector(duct_dict)
        self.assertDictEqual(duct, deconstructed_duct)

    def test_rotor_design_equality(self): 
        self.maxDiff = None
        config.OPTIMIZE_DUCT = False
        config.OPTIMIZE_CENTERBODY = False
        config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)
        config.OPTIMIZE_STAGE[0] = True
        _, _, _, design, _, _ = DeconstructDesignVector(rotor_dict)
        self.assertListEqual(rotor_design, design[0])
    
    def test_rotor_blading_equality(self): 
        self.maxDiff = None
        config.OPTIMIZE_DUCT = False
        config.OPTIMIZE_CENTERBODY = False
        config.OPTIMIZE_STAGE = [False] * len(config.OPTIMIZE_STAGE)
        config.OPTIMIZE_STAGE[0] = True
        _, _, blading, _, _, _ = DeconstructDesignVector(rotor_dict)
        print(rotor_blading)
        print(blading[0])
        self.assertDictEqual(rotor_blading, blading[0])



if __name__ == "__main__":
    unittest.main()