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
import numbers
import random
from pymoo.core.mixed import MixedVariableSampling
from pymoo.core.population import Population
from pymoo.core.individual import Individual

from init_designvector import DesignVector 
import config


class InitPopulation():
    """
    Population generation class to handle generation of the starting point for the genetic algorithm optimisation. 
    """


    def __init__(self,
                 population_type: str) -> None:
        """
        Initialisation for the InitPopulation class.

        Parameters
        ----------
        - population_type : str
            A string indicating the type of population to generate. Either 'biased' or 'random'.
            "biased" will generate a population with perturbed individuals based on a reference design, while "random" will generate a random population.
        - cfg : ModuleType
            The config module containing the design vector configuration.
        """

        self.type = population_type

        # Create a new design vector object
        self.design_vector = DesignVector()._construct_vector(config)

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
        if config.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, read in the reference design dictionary and extract the design vector x
            vars.append(config.CENTERBODY_VALUES["b_8"] / min(config.CENTERBODY_VALUES["y_t"], np.sqrt(-2 * config.CENTERBODY_VALUES["x_t"] * config.CENTERBODY_VALUES["r_LE"] / 3)) if min(config.CENTERBODY_VALUES["y_t"], np.sqrt(-2 * config.CENTERBODY_VALUES["x_t"] * config.CENTERBODY_VALUES["r_LE"] / 3)) > 0 else 0)
            vars.append(config.CENTERBODY_VALUES["b_15"])
            vars.append(config.CENTERBODY_VALUES["x_t"])
            vars.append(config.CENTERBODY_VALUES["y_t"])
            vars.append(config.CENTERBODY_VALUES["dz_TE"])
            vars.append(config.CENTERBODY_VALUES["r_LE"])
            vars.append(config.CENTERBODY_VALUES["trailing_wedge_angle"])
            vars.append(config.CENTERBODY_VALUES["Chord Length"])

        for i in range(config.NUM_STAGES):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if config.OPTIMIZE_STAGE[i]:
                # Read the reference values into the design vector
                for j in range(config.NUM_RADIALSECTIONS):
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_0"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_2"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_8"] / min(config.STAGE_DESIGN_VARIABLES[i][j]["y_t"], np.sqrt(-2 * config.STAGE_DESIGN_VARIABLES[i][j]["x_t"] * config.STAGE_DESIGN_VARIABLES[i][j]["r_LE"] / 3)) if min(config.STAGE_DESIGN_VARIABLES[i][j]["y_t"], np.sqrt(-2 * config.STAGE_DESIGN_VARIABLES[i][j]["x_t"] * config.STAGE_DESIGN_VARIABLES[i][j]["r_LE"] / 3)) > 0 else 0)
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_15"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_17"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["x_t"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["y_t"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["x_c"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["y_c"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["z_TE"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["dz_TE"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["r_LE"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["trailing_wedge_angle"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["trailing_camberline_angle"])
                    vars.append(config.STAGE_DESIGN_VARIABLES[i][j]["leading_edge_direction"])

        for i in range(config.NUM_STAGES):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if config.OPTIMIZE_STAGE[i]:
                # Read the reference values into the design vector
                vars.append(config.STAGE_BLADING_PARAMETERS[i]["root_LE_coordinate"])
                vars.append(int(config.STAGE_BLADING_PARAMETERS[i]["blade_count"]))
                vars.append(config.STAGE_BLADING_PARAMETERS[i]["ref_blade_angle"])
                vars.append(np.max(config.STAGE_BLADING_PARAMETERS[i]["radial_stations"]))  # The interfaces uses the radial locations, but the design varable is the blade radius!

                for j in range(config.NUM_RADIALSECTIONS):
                    vars.append(config.STAGE_BLADING_PARAMETERS[i]["chord_length"][j])
                for j in range(config.NUM_RADIALSECTIONS):
                    vars.append(config.STAGE_BLADING_PARAMETERS[i]["sweep_angle"][j])
                for j in range(config.NUM_RADIALSECTIONS):
                    vars.append(config.STAGE_BLADING_PARAMETERS[i]["blade_angle"][j])

        if config.OPTIMIZE_DUCT:
            # If the duct is to be optimised, read the reference values into the design vector
            vars.append(config.DUCT_VALUES["b_0"])
            vars.append(config.DUCT_VALUES["b_2"])
            vars.append(config.DUCT_VALUES["b_8"] / min(config.DUCT_VALUES["y_t"], np.sqrt(-2 * config.DUCT_VALUES["x_t"] * config.DUCT_VALUES["r_LE"] / 3)) if min(config.DUCT_VALUES["y_t"], np.sqrt(-2 * config.DUCT_VALUES["x_t"] * config.DUCT_VALUES["r_LE"] / 3)) > 0 else 0)
            vars.append(config.DUCT_VALUES["b_15"])
            vars.append(config.DUCT_VALUES["b_17"])
            vars.append(config.DUCT_VALUES["x_t"])
            vars.append(config.DUCT_VALUES["y_t"])
            vars.append(config.DUCT_VALUES["x_c"])
            vars.append(config.DUCT_VALUES["y_c"])
            vars.append(config.DUCT_VALUES["z_TE"])
            vars.append(config.DUCT_VALUES["dz_TE"])
            vars.append(config.DUCT_VALUES["r_LE"])
            vars.append(config.DUCT_VALUES["trailing_wedge_angle"])
            vars.append(config.DUCT_VALUES["trailing_camberline_angle"])
            vars.append(config.DUCT_VALUES["leading_edge_direction"])
            vars.append(config.DUCT_VALUES["Chord Length"])
            vars.append(config.DUCT_VALUES["Leading Edge Coordinates"][0])


        # Change vars from a list to a dictionary to match the expected structure of pymoo
        keys = list(self.design_vector.keys())
        vars = {key: value for key, value in zip(keys, vars)}

        return vars
                               

    def GenerateBiasedPopulation(self) -> Population:
        """
        Generate the initial population for the optimisation problem based on an existing solution.

        Returns
        -------
        - pop : Population
            The initial population for the optimisation problem as a pymoo Population object.
        """

        # Get the invidivual corresponding to the reference design
        reference_individual = self.DeconstructDictFromReferenceDesign()

        # Define helper functions to introduce spread in initial population
        def apply_real_spread(value: float, 
                              bounds: tuple[float, float]) -> float:
            """ Apply perturbation while keeping values within bounds. """
            lower_spread = value * config.SPREAD_CONTINUOUS[0]
            upper_spread = value * config.SPREAD_CONTINUOUS[1]        
            return max(bounds[0], min(bounds[1], random.uniform(value - lower_spread, value + upper_spread)))

        def apply_integer_spread(value: int,
                                 bounds: tuple[int, int]) -> int:
            """ Apply spread for the integer variable while keeping bounds. """
            return max(bounds[0], min(bounds[1], value + random.randint(config.SPREAD_DISCRETE[0], config.SPREAD_DISCRETE[1])))
        
        # Generate the initial population
        pop_dict = [None] * config.INIT_POPULATION_SIZE

        # Create the first individual as the reference design
        pop_dict[0] = reference_individual

        for i in range(1, config.INIT_POPULATION_SIZE):
            individual = reference_individual.copy()
            for key, value in reference_individual.items():
                bounds = self.design_vector[key].bounds
                # Apply spread based on the variable type
                if isinstance(value, (float, np.floating)):
                    individual[key] = apply_real_spread(value, bounds)
                elif isinstance(value, numbers.Integral):
                    individual[key] = apply_integer_spread(value, bounds)
            # Add the perturbed individual to the population
            pop_dict[i] = individual

        # Construct the population object
        individuals = []
        for pop in pop_dict:
            individuals.append(Individual(X=pop))
        pop = Population.create(*individuals)

        return pop
    

    def GeneratePopulation(self) -> Population|MixedVariableSampling:
        """
        Generate the initial population for the optimisation problem.

        Use either: 
        - A biased population where we introduce some perturbations around a known initial design vector
        - A random population where we sample the design vector uniformly across the bounds

        Returns
        -------
        - pop : Population|MixedVariableSampling
            The initial population for the optimisation problem as a pymoo Population object or a MixedVariableSampling object.
            The type of population is determined by the `type` parameter.
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
    import time
    start_time = time.time()
    test = InitPopulation("biased")
    print(test.GeneratePopulation())
    print(f"Generation took: {time.time() - start_time} seconds")


