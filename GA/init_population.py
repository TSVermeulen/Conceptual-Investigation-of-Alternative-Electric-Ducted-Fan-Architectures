"""
init_population
===============

Description
-----------
This module provides functionality to initialize populations for optimization problems, 
either by generating a biased population based on a reference design or by sampling 
randomly within the bounds of the design variables.

Classes
-------
InitPopulation
    A class to initialize populations for optimization problems. It supports generating 
    biased populations with perturbations around a reference design or random populations 
    sampled uniformly across the bounds.

Examples
--------
>>> import config
>>> import time

>>> start_time = time.time()
>>> init_pop = InitPopulation("biased", config)
>>> population = init_pop.GeneratePopulation()
>>> print(population)
>>> print(f"Population generation took: {time.time() - start_time} seconds")

Notes
-----
This module is designed to work with the pymoo optimization framework and assumes 
the design vector configuration is provided in the `config` module.

References
----------
pymoo: https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial version with tested biased population generation and basic random population generation functionality.
"""

import numpy as np
import random
from types import ModuleType
from pymoo.core.mixed import MixedVariableSampling
from pymoo.core.population import Population
from pymoo.core.individual import Individual

from init_designvector import DesignVector 


class InitPopulation():
    """
    Population generation class to handle generation of the starting point for the genetic algorithm optimisation. 
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

        # Set the seed for the random number generator to ensure reproducibility
        random.seed(42)  # 42 is the answer to everything
        np.random.seed(42)


    def DeconstructDictFromReferenceDesign(self)->dict:
        """
        Deconstruct the reference design vector dictionaries back into a pymoo design vector.
        This is used to create the initial population for the optimisation problem.

        Returns
        -------
        - vars : dict
            A dict of design variables for the optimisation problem.
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


        # Change vars from a list to a dictionary to match the expected structure of pymoo
        keys = list(self.design_vector.keys())
        vars = {key: value for key, value in zip(keys, vars)}

        return vars
                               

    def GenerateBiasedPopulation(self) -> list[Individual]:
        """
        Generate the initial population for the optimisation problem based on an existing solution.

        Returns
        -------
        - pop : list[Individual]
            The initial population for the optimisation problem as a list of shape (n, m), 
            where n is the number of individuals and m is the number of design variables.
        """

        # Get the invidivual corresponding to the reference design
        reference_individual = self.DeconstructDictFromReferenceDesign()

        # Define helper functions to introduce spread in initial population
        def apply_real_spread(value: float, 
                              bounds: tuple[float, float]) -> float:
            """ Apply perturbation while keeping values within bounds. """
            lower_spread = value * self.cfg.SPREAD_CONTINUOUS[0]
            upper_spread = value * self.cfg.SPREAD_CONTINUOUS[1]        
            return max(bounds[0], min(bounds[1], random.uniform(value - lower_spread, value + upper_spread)))

        def apply_integer_spread(value: int,
                                 bounds: tuple[int, int]) -> int:
            """ Apply spread for the integer variable while keeping bounds. """
            return max(bounds[0], min(bounds[1], value + random.randint(self.cfg.SPREAD_DISCRETE[0], self.cfg.SPREAD_DISCRETE[1])))
        
        # Generate the initial population
        pop_dict = [None] * self.cfg.INIT_POPULATION_SIZE

        # Create the first individual as the reference design
        pop_dict[0] = reference_individual

        for i in range(1, self.cfg.INIT_POPULATION_SIZE):
            individual = reference_individual.copy()
            for key, value in reference_individual.items():
                bounds = self.design_vector[key].bounds
                # Apply spread based on the variable type
                if isinstance(value, (float, np.float64)):
                    individual[key] = apply_real_spread(value, bounds)
                elif isinstance(value, (int)):
                    individual[key] = apply_integer_spread(value, bounds)
            # Add the perturbed individual to the population
            pop_dict[i] = individual

        # Construct the population object
        individuals = []
        for pop in pop_dict:
            individuals.append(Individual(X=pop))
        pop = Population.create(*individuals)

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


