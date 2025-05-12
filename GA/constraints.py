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
Version: 1.2

Changelog:
- V1.0: Initial implementation with basic equality and inequality constraints.
- V1.1: Implemented inequality constraint for efficiency such that eta is always > 0. 
- V1.2: Normalised constraints, added 1<T/Tref<1.01 constraint, extracted common power and thrust calculations to separate helper methods
"""

import numpy as np
import config

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

    
    def _calculate_power(self,
                         analysis_outputs: dict,
                         Lref: float) -> float:
        """ 
        Helper method to calculate the power in Watts.
        
        Parameters
        ----------
        - analysis_outputs : dict
            Outputs from MTFLOW
        - Lref : float
            Reference length

        Returns
        -------
        - Power : float
            A float of the power in Watts
        """
        return analysis_outputs['data']['Total power CP'] * (0.5 * config.atmosphere.density[0] * self.oper["Vinl"] ** 3 * Lref ** 2)


    def _calculate_thrust(self,
                          analysis_outputs: dict,
                          Lref: float) -> float:
        """
        Helper method to calculate the thrust in Newtons.
        
        Parameters
        ----------
        - analysis_outputs : dict
            Outputs from MTFLOW
        - Lref : float
            Reference length

        Returns
        -------
        - Thrust : float
            A float of the thrust in Newtons
        """
        return analysis_outputs['data']['Total force CT'] * (0.5 * config.atmosphere.density[0] * self.oper["Vinl"] ** 2 * Lref ** 2)


    def ConstantPower(self, 
                      analysis_outputs: dict, 
                      Lref: float,
                      thrust: float,
                      power: float) -> float:
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
        - thrust : float
            The thrust in Newtons. Not used here but included to force constant signature.
        - power : float
            The power in Watts. 

        Returns
        -------
        - float
            The computed normalised power constraint. This is a scalar value representing the power of the system.
        """

        return (power - self.ref_power) / self.ref_power  # Normalized power constraint 
    

    def KeepEfficiencyFeasibleUpper(self,
                               analysis_outputs: dict,
                               Lref: float,
                               thrust: float,
                               power: float) -> float:
        """
        Compute the inequality constraint for the efficiency. Enforces that eta < 1. 

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3).
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter.
            Not used in this method, but required for a uniform constraint function signature.
        - thrust : float
            The thrust in Newtons. Not used here but included to force constant signature.
        - power : float
            The power in Watts. Not used here but included to force constant signature.

        Returns
        -------
        - float
            The computed efficiency constraint. This is a scalar value representing the efficiency of the system.
        """

        # Compute the inequality constraint for the efficiency.
        return analysis_outputs['data']['EtaP'] - 1
    

    def KeepEfficiencyFeasibleLower(self,
                               analysis_outputs: dict,
                               Lref: float,
                               thrust: float,
                               power: float) -> float:
        """
        Compute the inequality constraint for the efficiency. Enforces that eta > 0. 

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3).
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter.
            Not used in this method, but required for a uniform constraint function signature.
        - thrust : float
            The thrust in Newtons. Not used here but included to force constant signature.
        - power : float
            The power in Watts. Not used here but included to force constant signature.

        Returns
        -------
        - float
            The computed efficiency constraint. This is a scalar value representing the efficiency of the system.
        """

        # Compute the inequality constraint for the efficiency.
        return -analysis_outputs['data']['EtaP']
    
    

    def MinimumThrust(self,
                      analysis_outputs: dict,
                      Lref: float,
                      thrust: float,
                      power: float) -> float:
        """
        Compute the inequality constraint for the thrust. Enforces that T > (1 - delta) * T_ref.

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3). 
            Not used here but included to force constant signature.
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter. 
            Not used here but included to force constant signature.
        - thrust : float
            The thrust in Newtons. 
        - power : float
            The power in Watts. Not used here but included to force constant signature.

        Returns
        -------
        - float 
            The computed normalised thrust constraint. 
        """
        return (thrust - (1 - config.deviation_range) * self.ref_thrust) / self.ref_thrust  # Normalized thrust constraint
    

    def MaximumThrust(self,
                      analysis_outputs: dict,
                      Lref: float,
                      thrust: float,
                      power: float) -> float:
        """
        Compute the upper bound for the thrust. Enforces that T < T_ref + delta.

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs from the MTFLOW forces output file. 
            Must contain all entries corresponding to an execution of 
            output_handling.output_processing().GetAllVariables(3)
            Not used here but included to force constant signature.
        - Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter. 
            Not used here but included to force constant signature.
        - thrust : float
            The thrust in Newtons. 
        - power : float
            The power in Watts. Not used here but included to force constant signature.

        Returns
        -------
        - float
            The computed normalised thrust constraint.
        """
        return (config.deviation_range * self.ref_thrust - thrust) / self.ref_thrust  # Normalized thrust constraint


    def ComputeConstraints(self,
                           analysis_outputs: dict,
                           Lref: float,
                           oper: dict,
                           out: dict) -> None:              
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
        - oper : dict
            The operating conditions dictionary.
        - out : dict
            A dictionary to store the computed constraints. The keys "G" and "H"
            will be populated with the inequality and equality constraints, respectively.
        
        Returns
        -------
        None, the out dictionary is updated in place with the computed constraints.

        Notes
        -----
            - The function uses `config.constraint_IDs` to determine which constraints to compute.
            - `config.constraint_IDs[0]` specifies the indices of inequality constraints.
            - `config.constraint_IDs[1]` specifies the indices of equality constraints.
            - If no constraints are specified, the corresponding output arrays ("G" or "H")
              will be empty 2D numpy arrays.                 
        """

        # Copy the operating conditions
        self.oper = oper.copy()

        # Define lists of all inequality and equality constraints, and filter them based on the constraint IDs
        ineq_constraints_list = [self.KeepEfficiencyFeasibleLower, self.KeepEfficiencyFeasibleUpper, self.MinimumThrust, self.MaximumThrust]
        eq_constraints_list = [self.ConstantPower]
        ineq_constraints = [ineq_constraints_list[i] for i in config.constraint_IDs[0]]
        eq_constraints = [eq_constraints_list[i] for i in config.constraint_IDs[1]]

        # Compute thrust and power
        thrust = self._calculate_thrust(analysis_outputs, Lref)
        power = self._calculate_power(analysis_outputs, Lref)

        # Set the reference values to self
        self.ref_power = config.P_ref_constr[0]
        self.ref_thrust = config.T_ref_constr[0]
        
        # Compute the inequality constraints and write them to out["G"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if ineq_constraints:
            computed_ineq_constraints = []
            for i in range(len(ineq_constraints)):
                computed_ineq_constraints.append(round(ineq_constraints[i](analysis_outputs,
                                                                           Lref,
                                                                           thrust,
                                                                           power), 5))
            
            out["G"] = np.column_stack(computed_ineq_constraints)
        else:
            out["G"] = [[]]

        # Compute the equality constraints and write them to out["H"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if eq_constraints:
            computed_eq_constraints = []
            for i in range(len(eq_constraints)):
                computed_eq_constraints.append(round(eq_constraints[i](analysis_outputs,
                                                                       Lref,
                                                                       thrust,
                                                                       power), 5))
        
            out["H"] = np.column_stack(computed_eq_constraints)
        else: 
            out["H"] = [[]]


    def ComputeMultiPointConstraints(self,
                                     analysis_outputs: list[dict],
                                     Lref: float,
                                     oper: list[dict],
                                     out: dict) -> None:              
        """
        Compute the inequality and equality constraints for a multi-point analysis based on the provided analysis outputs
        and configuration, and store the results in the output dictionary.

        Parameters
        ----------
        - analysis_outputs: list[dict] 
            A list of the output dictionaries containing the results of the analyses,
            which are used as inputs to the constraint functions.
        - Lref : float
            A reference length used in the computation of constraints.
        - out : dict
            A dictionary to store the computed constraints. The keys "G" and "H"
            will be populated with the inequality and equality constraints, respectively.
        
        Returns
        -------
        None, the out dictionary is updated in place with the computed constraints.

        Notes
        -----
            - The function uses `config.constraint_IDs` to determine which constraints to compute.
            - `config.constraint_IDs[0]` specifies the indices of inequality constraints.
            - `config.constraint_IDs[1]` specifies the indices of equality constraints.
            - If no constraints are specified, the corresponding output arrays ("G" or "H")
              will be empty 2D numpy arrays.                 
        """

        # Copy the operating conditions
        self.multi_oper = oper.copy()

        # Define lists of all inequality and equality constraints, and filter them based on the constraint IDs
        ineq_constraints_list = [self.KeepEfficiencyFeasibleLower, self.KeepEfficiencyFeasibleUpper, self.MinimumThrust, self.MaximumThrust]
        eq_constraints_list = [self.ConstantPower]
        ineq_constraints = [ineq_constraints_list[i] for i in config.constraint_IDs[0]]
        eq_constraints = [eq_constraints_list[i] for i in config.constraint_IDs[1]]

        num_outputs = len(analysis_outputs)

        # Compute thrust and power
        thrust = []
        power = []
        for i in range(num_outputs):
            self.oper = self.multi_oper[i]
            thrust.append(self._calculate_thrust(analysis_outputs[i], Lref))
            power.append(self._calculate_power(analysis_outputs[i], Lref))
        

        # Compute the inequality constraints and write them to out["G"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if ineq_constraints:
            num_ineq = len(ineq_constraints)
            computed_ineq_constraints = np.empty(num_outputs * num_ineq)
            for i, outputs in enumerate(analysis_outputs):
                self.ref_thrust = config.T_ref_constr[i]
                self.ref_power = config.P_ref_constr[i]
                self.oper = self.multi_oper[i]
                for j, constraint in enumerate(ineq_constraints):
                    computed_ineq_constraints[i * num_ineq + j] = round(constraint(outputs,
                                                                                   Lref,
                                                                                   thrust[i],
                                                                                   power[i]), 5)
            
            out["G"] = np.column_stack(computed_ineq_constraints)
        else:
            out["G"] = [[]]

        # Compute the equality constraints and write them to out["H"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if eq_constraints:
            num_eq = len(eq_constraints)
            computed_eq_constraints = np.empty(num_outputs * num_eq)

            for i, outputs in enumerate(analysis_outputs):
                self.oper = self.multi_oper[i]
                self.ref_thrust = config.T_ref_constr[i]
                self.ref_power = config.P_ref_constr[i]
                for j, constraint in enumerate(eq_constraints):
                    computed_eq_constraints[i * num_eq + j] = round(constraint(outputs,
                                                                               Lref,
                                                                               thrust[i],
                                                                               power[i]), 5)
        
            out["H"] = np.column_stack(computed_eq_constraints)
        else: 
            out["H"] = [[]]

    
if __name__ == "__main__":
    # Test execution of constraints using a test-case forces output. 

    # Add the parent and submodels paths to the system path
    import sys
    from pathlib import Path
    # Add the parent and submodels paths to the system path if they are not already in the path
    parent_path = str(Path(__file__).resolve().parent.parent)
    submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

    if parent_path not in sys.path:
        sys.path.append(parent_path)

    if submodels_path not in sys.path:
        sys.path.append(submodels_path)

    # Import MTFLOW interface submodels and other dependencies
    from Submodels.output_handling import output_processing
    
    # Extract outputs from the forces output file
    outputs = output_processing(analysis_name='initial_analysis').GetAllVariables(3)
    
    # Create an instance of the Constraints class
    test = Constraints()

    # Compute the constraints
    output = {}
    test.ComputeConstraints(outputs, 
                            Lref=config.BLADE_DIAMETERS[0],
                            oper=config.multi_oper[0],
                            out=output)
    
    print(output)

    output = {}
    test.ComputeMultiPointConstraints([outputs, outputs], 
                                      Lref=config.BLADE_DIAMETERS[0],
                                      oper=config.multi_oper,
                                      out=output)
    
    print(output)