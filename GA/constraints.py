"""
constraints
===========

Description
-----------
This module provides an interface to define and compute constraints for optimization problems. 
It includes methods to calculate equality and inequality constraints based on analysis outputs.

Classes
-------
Constraints
    A class to define and compute the constraints for the optimization problem.

Examples
--------
>>> constraints_class = Constraints()
>>> outputs = {"data": {"Total power CP": 0.5, "Total force CT": 0.3}}
>>> Lref = 1.0
>>> out = {}
>>> computed_constraints = constraints_class.ComputeConstraints(outputs, Lref, out)
>>> print(computed_constraints)

Notes
-----
This module is designed to work with optimization frameworks such as PyMoo. 
The constraints are structured to be compatible with PyMoo's constraint handling approach.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.1

Changelog:
- V1.0: Initial implementation with basic equality and inequality constraints.
- V1.1: Implemented inequality constraint for efficiency such that eta is always > 0. 
"""

import numpy as np
from types import ModuleType

class Constraints:
    """
    Class containing all the constraints for the genetic algorithm optimisation. 
    """


    def __init__(self) -> None:
        """
        Initialisation of the Constraints class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

    
    def ConstantPower(self, 
                      analysis_outputs: dict, 
                      Lref: float,
                      cfg: ModuleType) -> float:
        """
        Compute the equality constraint for the power coefficient.

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3)
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter. 
        - cfg: ModuleType
            The configuration module containing the reference values for Power.
            This module should contain the attributes `P_ref_constr`.
        """

        # Compute the equality constraint for the power coefficient. Assuming constant flight condition (i.e. density and speed), 
        power = analysis_outputs['data']['Total power CP'] * (0.5 * cfg.atmosphere.density[0] * cfg.oper["Vinl"] ** 3 * Lref ** 2)  # Power in Watts
        return power - cfg.P_ref_constr 

    
    def ConstantThrust(self,
                       analysis_outputs: dict,
                       Lref: float,
                       cfg: ModuleType) -> float:
        """
        Compute the equality constraint for the thrust coefficient.

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3)
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter. 
        - cfg: ModuleType
            The configuration module containing the reference values for Thrust.
            This module should contain the attributes `T_ref_constr`.
        """

        # Compute the equality constraint for the thrust coefficient. Assuming constant flight condition (i.e. density and speed), 
        thrust = analysis_outputs['data']['Total force CT'] * (0.5 * cfg.atmosphere.density[0] * cfg.oper["Vinl"] ** 2 * Lref ** 2)  # Thrust in Newtons
        return thrust - cfg.T_ref_constr
    

    def KeepEfficiencyFeasible(self,
                               analysis_outputs: dict,
                               Lref: float,
                               cfg: ModuleType) -> float:
        """
        Compute the inequality constraint for the efficiency. Enforces that eta>0. 

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3).
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter.
            Not used in this method, but required for a uniform constraint function signature.
        - cfg: ModuleType
            The configuration module containing the reference values for CP, CT, and Lref.
            Not used in this method, but required for a uniform constraint function signature.

        Returns
        -------
        - float
            The computed efficiency constraint. This is a scalar value representing the efficiency of the system.
        """

        # Compute the inequality constraint for the efficiency.
        return -analysis_outputs['data']['EtaP']


    def ComputeConstraints(self,
                           analysis_outputs: dict,
                           Lref: float,
                           out: dict,
                           cfg: ModuleType) -> dict:              
        """
        Compute the inequality and equality constraints based on the provided analysis outputs
        and configuration, and store the results in the output dictionary.

        Parameters
        ----------
        - analysis_outputs: dict 
            A dictionary containing the results of the analysis,
            which are used as inputs to the constraint functions.
        - Lref : float
            A reference length used in the computation of constraints.
        - out : dict
            A dictionary to store the computed constraints. The keys "G" and "H"
            will be populated with the inequality and equality constraints, respectively.
        - cfg: ModuleType
            The configuration module containing the reference values for CP and Lref.
            This module should contain the attributes `CP_ref_constr` and `L_ref_constr`.
        
        Returns
        -------
        - dict: 
            The updated output dictionary with the computed constraints:
                - "G": A 2D numpy array containing the computed inequality constraints.
                - "H": A 2D numpy array containing the computed equality constraints.

        Notes
        -----
            - The function uses `config.constraint_IDs` to determine which constraints to compute.
            - `config.constraint_IDs[0]` specifies the indices of inequality constraints.
            - `config.constraint_IDs[1]` specifies the indices of equality constraints.
            - If no constraints are specified, the corresponding output arrays ("G" or "H")
              will be empty 2D numpy arrays.                 
        """

        # Define lists of all inequality and equality constraints, and filter them based on the constraint IDs
        ineq_constraints_list = [self.KeepEfficiencyFeasible]
        eq_constraints_list = [self.ConstantPower, self.ConstantThrust]
        ineq_constraints = [ineq_constraints_list[i] for i in cfg.constraint_IDs[0]]
        eq_constraints = [eq_constraints_list[i] for i in cfg.constraint_IDs[1]]
        
        # Compute the inequality constraints and write them to out["G"]
        if ineq_constraints:
            computed_ineq_constraints = []
            for i in range(len(ineq_constraints)):
                computed_ineq_constraints.append(ineq_constraints[i](analysis_outputs,
                                                                     Lref,
                                                                     cfg))
            
            out["G"] = np.column_stack(computed_ineq_constraints)
        else:
            out["G"] = [[]]

        # Compute the equality constraints and write them to out["H"]
        if eq_constraints:
            computed_eq_constraints = []
            for i in range(len(eq_constraints)):
                computed_eq_constraints.append(eq_constraints[i](analysis_outputs,
                                                                 Lref,
                                                                 cfg))
        
            out["H"] = np.column_stack(computed_eq_constraints)
        else: 
            out["H"] = [[]]

        return out
    

if __name__ == "__main__":
    # Test execution of constraints using a test-case forces output. 

    # Add the parent and submodels paths to the system path
    import os
    import sys
    import config
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    submodels_path = os.path.join(parent_dir, "Submodels")
    sys.path.extend([parent_dir, submodels_path])

    # Import MTFLOW interface submodels and other dependencies
    from Submodels.output_handling import output_processing
    
    # Extract outputs from the forces output file
    outputs = output_processing(analysis_name='test_case').GetAllVariables(3)
    
    # Create an instance of the Constraints class
    test = Constraints()

    # Compute the constraints
    constraints = test.ComputeConstraints(outputs, 
                                          Lref=config.L_ref_constr,
                                          out={},
                                          cfg=config)
    
    print(constraints)