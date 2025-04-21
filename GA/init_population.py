"""
init_population
===============


"""

import numpy as np
from types import ModuleType
from pymoo.core.mixed import MixedVariableSampling

from designvectorinit import DesignVector 


class InitPopulation():
    """
    
    """


    def __init__(self,
                 type: str,
                 cfg: ModuleType) -> None:
        """
        Initialisation for the InitPopulation class.

        Parameters
        ----------
        - type : str
            A string
        - cfg : ModuleType
            The config module containing the design vector configuration.
        """

        self.type = type
        self.cfg = cfg

        # Create a new design vector object
        self.design_vector = DesignVector()._construct_vector(cfg)
    

    def DeconstructDictFromReferenceDesign(self)->list:
        """
        Deconstruct the reference design vector dictionaries back into a pymoo design vector.
        This is used to create the initial population for the optimisation problem.

        Returns
        -------
        - vars : list
            A list of design variables for the optimisation problem.
        """

        vars = []
        if self.cfg.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, read in the reference design dictionary and extract the design vector x
            vars.append(self.cfg.CENTERBODY_VALUES["b_8"] / min(self.cfg.CENTERBODY_VALUES["y_t"], np.sqrt(-2 * self.cfg.CENTERBODY_VALUES["x_t"] * self.cfg.CENTERBODY_VALUES["r_LE"] / 3)) if min(self.cfg.CENTERBODY_VALUES["y_t"], np.sqrt(-2 * self.cfg.CENTERBODY_VALUES["x_t"] * self.cfg.CENTERBODY_VALUES["r_LE"] / 3)) > 0 else 0)
            vars.append(self.cfg.CENTERBODY_VALUES["b_15"])
            vars.append(self.cfg.CENTERBODY_VALUES["x_t"])
            vars.append(self.cfg.CENTERBODY_VALUES["y_t"])
            vars.append(self.cfg.CENTERBODY_VALUES["dz_TE"])
            vars.append(self.cfg.CENTERBODY_VALUES["r_LE"])
            vars.append(self.cfg.CENTERBODY_VALUES["trailing_wedge_angle"])
            vars.append(self.cfg.CENTERBODY_VALUES["Chord Length"])

        for i in range(self.cfg.NUM_STAGES):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if self.cfg.OPTIMIZE_STAGE[i]:
                # Read the reference values into the design vector
                for j in range(self.cfg.NUM_RADIALSECTIONS):
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["b_0"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["b_2"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["b_8"] / min(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["y_t"], np.sqrt(-2 * self.cfg.STAGE_DESIGN_VARIABLES[i][j]["x_t"] * self.cfg.STAGE_DESIGN_VARIABLES[i][j]["r_LE"] / 3)) if min(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["y_t"], np.sqrt(-2 * self.cfg.STAGE_DESIGN_VARIABLES[i][j]["x_t"] * self.cfg.STAGE_DESIGN_VARIABLES[i][j]["r_LE"] / 3)) > 0 else 0)
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["b_15"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["b_17"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["x_t"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["y_t"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["x_c"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["y_c"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["z_TE"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["dz_TE"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["r_LE"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["trailing_wedge_angle"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["trailing_camberline_angle"])
                    vars.append(self.cfg.STAGE_DESIGN_VARIABLES[i][j]["leading_edge_direction"])

        for i in range(self.cfg.NUM_STAGES):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if self.cfg.OPTIMIZE_STAGE[i]:
                # Read the reference values into the design vector
                vars.append(self.cfg.STAGE_BLADING_PARAMETERS[i]["root_LE_coordinate"])
                vars.append(int(self.cfg.STAGE_BLADING_PARAMETERS[i]["blade_count"]))
                vars.append(self.cfg.STAGE_BLADING_PARAMETERS[i]["ref_blade_angle"])
                vars.append(np.max(self.cfg.STAGE_BLADING_PARAMETERS[i]["radial_stations"]))  # The interfaces uses the radial locations, but the design varable is the blade radius!

                for j in range(self.cfg.NUM_RADIALSECTIONS):
                    vars.append(self.cfg.STAGE_BLADING_PARAMETERS[i]["chord_length"][j])
                for j in range(self.cfg.NUM_RADIALSECTIONS):
                    vars.append(self.cfg.STAGE_BLADING_PARAMETERS[i]["sweep_angle"][j])
                for j in range(self.cfg.NUM_RADIALSECTIONS):
                    vars.append(self.cfg.STAGE_BLADING_PARAMETERS[i]["blade_angle"][j])

        if self.cfg.OPTIMIZE_DUCT:
            # If the duct is to be optimised, read the reference values into the design vector
            vars.append(self.cfg.DUCT_VALUES["b_0"])
            vars.append(self.cfg.DUCT_VALUES["b_2"])
            vars.append(self.cfg.DUCT_VALUES["b_8"] / min(self.cfg.DUCT_VALUES["y_t"], np.sqrt(-2 * self.cfg.DUCT_VALUES["x_t"] * self.cfg.DUCT_VALUES["r_LE"] / 3)) if min(self.cfg.DUCT_VALUES["y_t"], np.sqrt(-2 * self.cfg.DUCT_VALUES["x_t"] * self.cfg.DUCT_VALUES["r_LE"] / 3)) > 0 else 0)
            vars.append(self.cfg.DUCT_VALUES["b_15"])
            vars.append(self.cfg.DUCT_VALUES["b_17"])
            vars.append(self.cfg.DUCT_VALUES["x_t"])
            vars.append(self.cfg.DUCT_VALUES["y_t"])
            vars.append(self.cfg.DUCT_VALUES["x_c"])
            vars.append(self.cfg.DUCT_VALUES["y_c"])
            vars.append(self.cfg.DUCT_VALUES["z_TE"])
            vars.append(self.cfg.DUCT_VALUES["dz_TE"])
            vars.append(self.cfg.DUCT_VALUES["r_LE"])
            vars.append(self.cfg.DUCT_VALUES["trailing_wedge_angle"])
            vars.append(self.cfg.DUCT_VALUES["trailing_camberline_angle"])
            vars.append(self.cfg.DUCT_VALUES["leading_edge_direction"])
            vars.append(self.cfg.DUCT_VALUES["Chord Length"])
            vars.append(self.cfg.DUCT_VALUES["Leading Edge Coordinates"][0])

        return vars
                               

    def GenerateBiasedPopulation(self) -> np.ndarray:
        """
        Generate the initial population for the optimisation problem based on an existing solution.

        Returns
        -------
        - pop
            The initial population for the optimisation problem as np.ndarray of shape (n, m), 
            where n is the number of individuals and m is the number of design variables.
        """

        # Get the invidivual corresponding to the reference design
        reference_individual = np.array(self.DeconstructDictFromReferenceDesign())

        # Define helper functions to introduce spread in initial population
        def apply_real_spread(value: float, 
                              bounds: tuple[float, float]) -> float:
            """ Apply perturbation while keeping values within bounds. """
            lower_spread = value * self.cfg.SPREAD_CONTINUOUS[0]
            upper_spread = value * self.cfg.SPREAD_CONTINUOUS[1]
            return np.clip(np.random.uniform(value - lower_spread, value + upper_spread), 
                           bounds[0], bounds[1])

        def apply_integer_spread(value: int,
                                 bounds: tuple[int, int]) -> int:
            """ Apply spread for the integer variable while keeping bounds. """
            return np.clip(value + np.random.choice([self.cfg.SPREAD_DISCRETE[0], self.cfg.SPREAD_DISCRETE[1]]), bounds[0], bounds[1])
        
        # Generate the initial population
        pop = np.zeros((self.cfg.POPULATION_SIZE, len(reference_individual)))
        for i in range(self.cfg.POPULATION_SIZE):
            if i == 0:
                # First individual is the reference individual
                pop[i] = reference_individual
                continue
            else:
                # Create a perturbed individual based on the reference individual
                individual = np.copy(reference_individual)
                for j, value in enumerate(individual):
                    # Apply spread based on the variable type
                    if isinstance(value, (float, np.float64)):
                        individual[j] = apply_real_spread(value, self.design_vector[f"x{j}"].bounds)
                    elif isinstance(value, (int)):
                        individual[j] = apply_integer_spread(value, self.design_vector[f"x{j}"].bounds)
                # Add the perturbed individual to the population
                pop[i] = individual

        # Return the generated population
        return pop
    

    def GeneratePopulation(self):
        """
        Generate the initial population for the optimisation problem.

        Use either: 
        - A biased population where we introduce some perturbations around a known initial design vector
        - A random population where we sample the design vector uniformly across the bounds
        """

        # Use either: 
        # - A biased population where we introduce some perturbations around a known design vector
        # - A random population where we sample the design vector uniformly across the bounds

        if self.type == "biased":
            # Generate a biased population based on an existing solution
            pop = self.GenerateBiasedPopulation()
        elif self.type == "random":   
            # Generate a random population
            pop = MixedVariableSampling()
        else:
            raise ValueError("Invalid population type. Choose 'biased' or 'random'.")
            
        return pop
    

if __name__ == "__main__":
    import config
    import time
    start_time = time.time()
    test = InitPopulation("biased", config)
    print(test.GeneratePopulation())
    print(f"Generation took: {time.time() - start_time} seconds")


