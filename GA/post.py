"""
post
====

Description
----------
This module provides post-processing functionality for analyzing and visualizing the results of a Pymoo optimization. 
It includes methods for loading optimization results, extracting population data, and generating comparative plots 
for axisymmetric geometries, blading data, and blade design profiles.

Classes
--------
PostProcessing
    A class to handle post-processing of optimization results, including data extraction and visualization.

Examples
--------    
>>> output = Path('res_pop20_gen20_250506220150593712.dill')
>>> processing_class = PostProcessing(fname=output)
>>> res = processing_class.load_res()
>>> processing_class.ExtractPopulationData(res)
>>> processing_class.main()

Notes
-----
This module assumes the presence of a Pymoo optimization results file in `.dill` format. It integrates with 
various utility modules for handling design vectors, parameterizations, and file operations. Ensure that the 
required dependencies and configuration files are correctly set up before using this module.

References
----------
For more details on the Pymoo framework, refer to the official documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version 1.1

Changelog:
- V1.0: Initial implementation of plotting capabilities of outputs.
- V1.1: Added convergence property plotter.
"""

# Import standard libraries
import dill
import copy
from pathlib import Path
from cycler import cycler
from typing import Any, Optional

# Import 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP

# Ensure all paths are correctly setup
from utils import ensure_repo_paths
ensure_repo_paths()
 

# Import interfacing modules
import config # type: ignore 
from Submodels.Parameterizations import AirfoilParameterization # type: ignore 
from design_vector_interface import DesignVectorInterface # type: ignore 
from Submodels.file_handling import fileHandlingMTFLO #type: ignore

# Adjust open figure warning
plt.rcParams['figure.max_open_warning'] = 50

class PostProcessing:
    """
    Class to analyse all output data from the Pymoo optimisation.
    """

    _airfoil_param = AirfoilParameterization()  # shared, read-only


    def __init__(self,
                 fname: Path,
                 base_dir: Optional[Path] = None) -> None:
        """
        Initialization of the PostProcessing class.

        Parameters
        ----------
        - fname : Path
            The filename or path of the .dill file to be loaded.
            If not an absolute path it will be relative to base_dir.
        - base_dir : Path, optional
            The base directory to use if fname is not an absolute path.
            Defaults to the directory containing this script. 

        Returns
        -------
        None
        """
        
        # If base_dir is not provided, use the script's directory
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = base_dir

        # Coerce fname to Path and resolve it if it's not already absolute
        fname = Path(fname)
        self.results_path = fname if fname.is_absolute() else (self.base_dir / fname).resolve()
        
        # Validate file extension
        if self.results_path.suffix.lower() != '.dill':
            raise ValueError(f"File must have .dill extension. Got: {self.results_path.suffix}")


    def load_res(self) -> object:
        """
        Load and return the optimization results from the specified .dill file.
        
        Returns
        - res : object
            The reconstructed pymoo optimisation results object
        """

        try:
            # Open and load the results file
            with self.results_path.open('rb') as f:
                # ignore=False ensures we get an error if the object cannot be reconstructed. 
                res = dill.load(f, ignore=False)
                
            return res
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Results file not found. Ensure {self.results_path} exists") from e

        except dill.UnpicklingError as e:
            raise RuntimeError(f"Error loading results: {e}") from e


    def _extract_data(self, population):
        """
        Helper method to extract and deconstruct vectors from a population.
        """
        vec_interface = DesignVectorInterface()

        decomposed_data = [vec_interface.DeconstructDesignVector(individual.X) for individual in population]
        CB_data = [data[0] for data in decomposed_data]
        duct_data = [data[1] for data in decomposed_data]
        design_data = [data[2] for data in decomposed_data]
        blading_data = [data[3] for data in decomposed_data]
        
        return CB_data, duct_data, blading_data, design_data
    

    def ExtractPopulationData(self,
                              res: object) -> None:
        """
        Extract all population data from the results object, 
        deconstruct the design vectors, and write all data to lists in self.

        Parameters
        ----------
        - res : object
            The reconstructed pymoo results object.
        """
            
        # Write data from the full population to self
        self.CB_data, self.duct_data, self.blading_data, self.design_data = self._extract_data(res.pop)

        # Write data from the optimum individuals to self
        self.CB_data_opt, self.duct_data_opt, self.blading_data_opt, self.design_data_opt = self._extract_data(res.opt)
    

    def CompareAxisymmetricGeometry(self,
                                    reference: dict[str, Any],
                                    optimised: list[dict[str, Any]],
                                    individual: bool = False) -> None:
        """
        Generate plots of the original and optimised (normalised) axisymmetric profiles.

        Parameters
        ----------
        - reference : dict[str, Any]
            The reference geometry.
        - optimised : list[dict[str, Any]]
            A list of the optimised geometry.
        - individual : bool, optional
            An optional bool to determine if individual comparison plots between each optimised individual and 
            the reference design should be generated. Default value is False.

        Returns
        -------
        None
        """

        # Initiate a local instance of the AirfoilParameterization class
        parameterization = AirfoilParameterization()

        # Compute the original geometry (x,y) coordinates
        (original_upper_x, 
        original_upper_y, 
        original_lower_x, 
        original_lower_y) = parameterization.ComputeProfileCoordinates(reference)
        
        # Precompute the concatenated original geometry coordinates to avoid repeated operatings while plotting
        original_x = np.concatenate((original_upper_x, np.flip(original_lower_x)), axis=0)
        original_y = np.concatenate((original_upper_y, np.flip(original_lower_y)), axis=0)

        # Create grouped figure to compare the geometry between the reference and the optimised designs
        grouped_fig, ax1 = plt.subplots()
        
        # First plot the original geometry
        ax1.plot(original_x,
                 original_y, 
                 "k-.", 
                 label="Original Geometry",
                 )
        
        # Loop over all individuals in the final population and plot their geometries
        for i, geom in enumerate(optimised):                
            # Compute the optimised geometry
            (opt_upper_x, 
            opt_upper_y, 
            opt_lower_x, 
            opt_lower_y) = parameterization.ComputeProfileCoordinates(geom)
            
            # Compute the concatenated optimised x and y coordinates
            opt_x = np.concatenate((opt_upper_x, np.flip(opt_lower_x)), axis=0)
            opt_y = np.concatenate((opt_upper_y, np.flip(opt_lower_y)), axis=0)

            # Plot the optimised geometry
            ax1.plot(opt_x,
                     opt_y, 
                     label=f"Individual {i}",
                     )
            
            if individual:     
                # Create figure for the individual comparison plot
                plt.figure(f"Comparison for individual {i}")
                # plot the original geometry
                plt.plot(original_x,
                         original_y, 
                         "k-.", 
                         label="Original Geometry",
                         )
                plt.plot(opt_x,
                         opt_y, 
                         label=f"Individual {i}",
                         )
                plt.legend(bbox_to_anchor=(1,1))
                plt.grid(which='both')
                plt.minorticks_on()
                plt.tight_layout()
                plt.xlabel("Axial Coordinate [m]")
                plt.ylabel("Radial Coordinate [m]")
                plt.show()
                plt.close()
        
        ax1.grid(which='both')
        ax1.minorticks_on()
        ax1.set_xlabel("Axial Coordinate [m]")
        ax1.set_ylabel("Radial Coordinate [m]")
        ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
        grouped_fig.tight_layout()


    def ConstructBladeProfile(self,
                              design:list[dict[str, Any]],
                              section_idx: int) -> tuple:
        """
        Function to compute the rotated upper and lower profile coordinates for a blade section.

        Parameters
        ----------
        - design : list[dict[str, Any]]
            The list of the design dictionaries for each blade profile in the stage. 
        - section_idx : int
            The index of the radial section being constructed. 

        Returns
        -------
        - tuple [np.ndarray]
            - upper_x
            - upper_y
            - lower_x
            - lower_y
        """
   
        # Create complete airfoil representation
        (upper_x, 
         upper_y, 
         lower_x,
         lower_y) = self._airfoil_param.ComputeProfileCoordinates(design[section_idx])

        return upper_x, upper_y, lower_x, lower_y
    

    def _plot_scalar_blading_parameter(self, x, k, key, reference_value, optimised_blading, stage_idx, base_bar_width):
        """Helper method to plot scalar blading parameters."""
        ref_val = reference_value
        if key == "ref_blade_angle":
            ref_val = np.rad2deg(ref_val)
        
        # Plot the reference data
        plt.bar(x[k], ref_val, width=base_bar_width, label="Reference",
                color='black', hatch='//', edgecolor='white')
        
        # Plot the optimised blading parameters
        for j, opt_vals in enumerate(optimised_blading):
            opt_val = opt_vals[stage_idx][key]
            if key == "ref_blade_angle":
                opt_val = np.rad2deg(opt_val)
            plt.bar(x[k] + (j + 1) * base_bar_width, opt_val,
                    width=base_bar_width, label=f"Individual {j}")


    def _plot_rps_blading_parameter(self, x, k, reference_rps, optimised_blading, stage_idx, base_bar_width):
        """Helper method to plot RPS blading parameters."""
        num_rps = len(reference_rps)
        sub_bar_width = base_bar_width / num_rps
        
        # Plot reference RPS values
        for r, r_val in enumerate(reference_rps):
            offset = (r - (num_rps - 1) / 2) * sub_bar_width
            plt.bar(x[k] + offset, r_val, width=sub_bar_width,
                    label="Reference" if r == 0 else "", 
                    color='black', hatch='//', edgecolor='white')

        # Plot optimised RPS values
        for j, opt_vals in enumerate(optimised_blading):
            opt_rps = opt_vals[stage_idx]["RPS_lst"]
            for r, opt_r_val in enumerate(opt_rps):
                offset = (r - (num_rps - 1) / 2) * sub_bar_width
                plt.bar(x[k] + offset + (j + 1) * base_bar_width, opt_r_val,
                        width=sub_bar_width,
                        label=f"Individual {j}" if r == 0 else "")


    def _plot_radial_stations_parameter(self, x, k, reference_value, optimised_blading, stage_idx, base_bar_width):
        """Helper method to plot radial stations parameters (blade diameter)."""
        plt.bar(x[k], max(reference_value) * 2, width=base_bar_width, 
                color='black', hatch="//", edgecolor="white")
        
        for j, opt_vals in enumerate(optimised_blading):
            opt_val = opt_vals[stage_idx]["radial_stations"]
            opt_val = max(opt_val) * 2  # Time 2 since the radial stations array is defined over the blade radius. 
            plt.bar(x[k] + (j + 1) * base_bar_width, opt_val,
                    width=base_bar_width, label=f"Individual {j}")


    def _plot_blading_bar_chart(self, stage_idx, reference_blading, optimised_blading):
        """Helper method to create bar chart comparing blading parameters."""
        variables = [
            "Root LE coordinate [m]",
            "Reference Blade Angle [deg]",
            "Blade Count [-]",
            "RPS [-]",
            "Blade Diameter [m]"
        ]
        
        keys = [
            "root_LE_coordinate",
            "ref_blade_angle", 
            "blade_count",
            "RPS_lst",
            "radial_stations"
        ]

        num_indiv = len(optimised_blading)
        x = np.arange(len(keys))
        base_bar_width = 0.8 / num_indiv

        plt.figure(f"Bar Chart with blading parameters for stage {stage_idx}")

        for k, key in enumerate(keys):
            if key == "radial_stations":
                self._plot_radial_stations_parameter(x, k, reference_blading[stage_idx][key], 
                                                    optimised_blading, stage_idx, base_bar_width)
            elif key == "RPS_lst":
                self._plot_rps_blading_parameter(x, k, reference_blading[stage_idx][key],
                                            optimised_blading, stage_idx, base_bar_width)
            else:
                self._plot_scalar_blading_parameter(x, k, key, reference_blading[stage_idx][key],
                                                optimised_blading, stage_idx, base_bar_width)

        # Format the plot
        plt.xticks(x + (base_bar_width * num_indiv) / 2, variables, rotation=90)
        plt.title("Comparison of Reference vs Optimized Design Variables")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(axis='y', which='both')
        plt.yscale('log')
        plt.minorticks_on()
        plt.tight_layout()


    def _plot_sectional_blading_data(self, stage_idx, reference_blading, optimised_blading):
        """Helper method to plot sectional blading data (chord length, sweep angle, blade angle)."""
        marker_cycle = cycler(marker=['o', 's', '^', '<', 'v', '>', '*', '+'])
        color_cycle = cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
        
        fig, ax = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        fig.suptitle(f"Sectional blading data for stage {stage_idx}")

        # Set cyclers and grids
        for row in range(2):
            for col in range(2):
                if row == 1 and col == 1:  # Skip the bottom-right subplot (used for legend)
                    continue
                ax[row, col].set_prop_cycle(marker_cycle * color_cycle)
                ax[row, col].minorticks_on()
                ax[row, col].grid(which='both')
                ax[row, col].set_xlabel("Radial coordinate [m]")

        # Plot reference data
        ax[0,0].plot(reference_blading[stage_idx]["radial_stations"],
                    reference_blading[stage_idx]["chord_length"],
                    label="Reference", color='black', marker="x", ms=3)
        ax[0,0].set_title("Chord length distribution [m]")

        ax[0,1].plot(reference_blading[stage_idx]["radial_stations"],
                    np.rad2deg(reference_blading[stage_idx]["sweep_angle"]),
                    label="Reference", color='black', marker="x", ms=3)
        ax[0,1].set_title("Sweep angle distribution [deg]")

        ax[1,0].plot(reference_blading[stage_idx]["radial_stations"],
                    np.rad2deg(reference_blading[stage_idx]["blade_angle"]),
                    label="Reference", color='black', marker="x", ms=3)
        ax[1,0].set_title("Blade angle distribution [deg]")

        # Plot optimised data
        for j, opt_vals in enumerate(optimised_blading):
            ax[0,0].plot(opt_vals[stage_idx]["radial_stations"],
                        opt_vals[stage_idx]["chord_length"],
                        label=f"Individual {j}", ms=3)
            
            ax[0,1].plot(opt_vals[stage_idx]["radial_stations"],
                        np.rad2deg(opt_vals[stage_idx]["sweep_angle"]),
                        label=f"Individual {j}", ms=3)
            
            ax[1,0].plot(opt_vals[stage_idx]["radial_stations"],
                        np.rad2deg(opt_vals[stage_idx]["blade_angle"]),
                        label=f"Individual {j}", ms=3)

        # Use bottom-right subplot for legend
        ax[1,1].axis('off')
        handles, labels = ax[0,0].get_legend_handles_labels()
        ax[1,1].legend(handles, labels, loc='center', ncol=2)


    def CompareBladingData(self,
                        reference_blading: list[dict[str, Any]],
                        optimised_blading: list[list[dict[str, Any]]]) -> None:
        """
        Generate plots of the blading data for the final population members and the initial reference design.

        Parameters
        ----------
        - reference_blading : list[dict[str, Any]]
            The reference blading data. Each dictionary in the list corresponds to a stage.
        - optimised_blading : list[dict[str, Any]]
            The optimised blading data. Each nested list corresponds to an individual in the final optimised population.
        """
        # Generate bar charts and sectional plots for each optimized stage
        for stage_idx, opt_stage in enumerate(config.OPTIMIZE_STAGE):
            if opt_stage:
                # Create bar chart comparison
                self._plot_blading_bar_chart(stage_idx, reference_blading, optimised_blading)
                
                # Create sectional data plots
                self._plot_sectional_blading_data(stage_idx, reference_blading, optimised_blading)
    
    
    def _plot_single_blade_profile(self, 
                                   design: list[dict[str, Any]], 
                                   section_idx: int, 
                                   label: str, 
                                   color: str, 
                                   linestyle: str = '-') -> None:
        """
        Helper method to plot a single blade profile with camber line.
        
        Parameters
        ----------
        - design : list[dict[str, Any]]
            The design data for the blade stage
        - section_idx : int
            The radial section index
        - label : str
            Label for the plot legend
        - color : str
            Color for the plot
        - linestyle : str, optional
            Line style for the plot. Defaults to '-'
        """
        upper_x, upper_y, lower_x, lower_y = self.ConstructBladeProfile(design, section_idx)
        
        # Plot the blade profile
        plt.plot(np.concatenate((upper_x, np.flip(lower_x))),
                np.concatenate((upper_y, np.flip(lower_y))),
                label=label, color=color, linestyle=linestyle)
        
        # Plot the camber line
        plt.plot((upper_x + lower_x) / 2,
                (upper_y + lower_y) / 2,
                color=color, linestyle=linestyle)


    def _plot_reference_blade_profile(self, 
                                      reference_design: list[list[dict[str, Any]]], 
                                      stage_idx: int, 
                                      section_idx: int) -> None:
        """
        Helper method to plot the reference blade profile.
        
        Parameters
        ----------
        - reference_design : list[list[dict[str, Any]]]
            The reference design data
        - stage_idx : int
            The stage index
        - section_idx : int
            The radial section index
        """
        self._plot_single_blade_profile(reference_design[stage_idx], section_idx, 
                                    "Reference", "tab:orange", "-.")


    def _format_blade_plot(self, 
                           radial_coordinate: float, 
                           stage_idx: int, 
                           plot_type: str = "") -> None:
        """
        Helper method to format blade profile plots.
        
        Parameters
        ----------
        - radial_coordinate : float
            The radial coordinate for the plot title
        - stage_idx : int
            The stage index for the plot title
        - plot_type : str, optional
            Additional text for the plot title
        """
        title_suffix = f" ({plot_type})" if plot_type else ""
        plt.legend()
        plt.title(f"Blade profile at r={round(radial_coordinate, 3)}R for Stage {stage_idx}{title_suffix}")
        plt.minorticks_on()
        plt.grid(which='both')
        plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
        plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
        plt.tight_layout()


    def _plot_multiple_optimum_designs(self, 
                                       multi_optimum_designs: list, 
                                       reference_design: list[list[dict[str, Any]]], 
                                       stage_idx: int) -> None:
        """
        Helper method to plot multiple optimum designs for a given stage.
        
        Parameters
        ----------
        - multi_optimum_designs : list
            List of optimum design data
        - reference_design : list[list[dict[str, Any]]]
            The reference design data
        - stage_idx : int
            The stage index to plot
        """
        radial_coordinates = np.linspace(0, 1, config.NUM_RADIALSECTIONS[stage_idx])
        colors = plt.cm.tab10(np.linspace(0, 1, len(multi_optimum_designs)))
        
        for j, radial_coordinate in enumerate(radial_coordinates):
            plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}_Stage{stage_idx}")
            
            # Plot each optimum design
            for opt_idx, current_opt_design in enumerate(multi_optimum_designs):
                self._plot_single_blade_profile(current_opt_design[stage_idx], j, 
                                            f"Optimised (Ind {opt_idx})", colors[opt_idx])
            
            # Plot reference design
            self._plot_reference_blade_profile(reference_design, stage_idx, j)
            
            # Format the plot
            self._format_blade_plot(radial_coordinate, stage_idx, "Multiple Optima")


    def _plot_single_optimum_design(self, 
                                    optimised_design: list[list[dict[str, Any]]], 
                                    reference_design: list[list[dict[str, Any]]], 
                                    stage_idx: int, 
                                    individual: int | str) -> None:
        """
        Helper method to plot a single optimum design for a given stage.
        
        Parameters
        ----------
        - optimised_design : list[list[dict[str, Any]]]
            The optimised design data
        - reference_design : list[list[dict[str, Any]]]
            The reference design data
        - stage_idx : int
            The stage index to plot
        - individual : int | str
            The individual identifier for labeling
        """
        radial_coordinates = np.linspace(0, 1, config.NUM_RADIALSECTIONS[stage_idx])
        
        for j, radial_coordinate in enumerate(radial_coordinates):
            plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}_Stage{stage_idx}")
            
            # Plot optimised design
            self._plot_single_blade_profile(optimised_design[stage_idx], j, "Optimised", 'tab:blue')
            
            # Plot reference design
            self._plot_reference_blade_profile(reference_design, stage_idx, j)
            
            # Format the plot
            self._format_blade_plot(radial_coordinate, stage_idx, f"individual: {individual}")


    def CompareBladeDesignData(self,
                            reference_design: list[list[dict[str, Any]]],
                            res: object,
                            individual: int | str = "opt",
                            optimised_design: Optional[list[list[dict[str, Any]]]] = None) -> None:
        """
        Compares the blade design data of a reference design with an optimized design 
        and generates plots for visual comparison at various radial sections.

        Parameters
        ----------
        - reference_design : list[list[dict[str, Any]]] 
            The reference blade design data, structured as a list of stages, 
            where each stage contains a list of dictionaries representing radial sections.
        - res : object 
            The optimization result object containing the design vector of the optimized design.
        - individual : int | str, optional 
            Specifies which individual design to compare against. If "opt", the optimum design 
            from the optimization result is used. If an integer, the corresponding individual 
            from the `optimised_design` list is used. Defaults to "opt".
        - optimised_design list[list[dict[str, Any]]], optional 
            The optimized blade design data, structured similarly to `reference_design`. 
            Required if `individual` is an integer. Defaults to None.
                            
        Returns
        -------
        None
        """
        
        if individual == "opt":
            # Handle optimum design case
            if len(res.X) == 1:
                # Single optimum design
                optimum_dict = res.X[0]
                (_, _, optimised_design, _, _) = DesignVectorInterface().DeconstructDesignVector(optimum_dict)
                
                # Plot single optimum for each optimized stage
                for i in range(len(config.OPTIMIZE_STAGE)):
                    if config.OPTIMIZE_STAGE[i]:
                        self._plot_single_optimum_design(optimised_design, reference_design, i, individual)
            else:
                # Multiple optimum designs
                multi_optimum_designs = []
                for design_dict in res.X:
                    (_, _, design_opt, _, _) = DesignVectorInterface().DeconstructDesignVector(design_dict)
                    multi_optimum_designs.append(design_opt)
                
                # Plot multiple optima for each optimized stage
                for i in range(len(config.OPTIMIZE_STAGE)):
                    if config.OPTIMIZE_STAGE[i]:
                        self._plot_multiple_optimum_designs(multi_optimum_designs, reference_design, i)
        else:
            # Handle individual index case
            if optimised_design is None:
                raise ValueError("'optimised_design' must be supplied when 'individual' is an int.")
            
            optimised_design = copy.deepcopy(optimised_design[individual])
            
            # Plot individual design for each optimized stage
            for i in range(len(config.OPTIMIZE_STAGE)):
                if config.OPTIMIZE_STAGE[i]:
                    self._plot_single_optimum_design(optimised_design, reference_design, i, individual)

    
    def GenerateConvergenceStatistics(self,
                                      res : object) -> None:
        """
        Generate some graphs to analyse the convergence behaviour of the optimisation. 
        Analyses:
            - The convergence of the best and average objective values.
            - Diversity of the design vectors
            - Maximum successive change in design vectors
            - Constraint violation
        """ 

        # First visualise the convergence of the objective values
        n_evals = [e.evaluator.n_eval for e in res.history]
        generational_optimum = np.array([e.opt[0].F for e in res.history])       
        avg_objectives = np.array([np.mean(e.pop.get("F"), axis=0) for e in res.history])

        plt.figure()
        plt.title("Optimum and average objective values over generations")

        # For multi-objective probems, we plot each objective separately.
        if avg_objectives.ndim > 1 and avg_objectives.shape[1] >1:
            n_obj = avg_objectives.shape[1]
            for i in range(n_obj):
                plt.plot(n_evals, generational_optimum[:, i], "-.x", label=f'Generational optimum for objective {i + 1}')
                plt.plot(n_evals, avg_objectives[:,i], "-*", label=f"Generational average for objective {i + 1}")      
        else:
            avg_objectives = avg_objectives.squeeze()
            plt.plot(n_evals, generational_optimum, "-.x", label='Generational optimum')
            plt.plot(n_evals, avg_objectives, "-*", label="Generational average")

        plt.grid(which='both')
        plt.yscale('log')
        plt.xlabel("Total number of function evaluations [-]")
        plt.ylabel("Objective value [-]")
        plt.legend()
        plt.minorticks_on()
        plt.tight_layout()

        # Visualise diversity of the design vectors, measured through the averaged standard deviation of all variables of the generation
        diversity = []

        # Extract the key ordering of the first optimum individual and use it to ensure all individuals are ordered the same
        x_keys = list(res.opt.get("X")[0].keys())

        # Compute the mean standard deviation of each population
        for e in res.history:
            X_dicts = e.pop.get("X")
            X = np.array([[d[k] for k in x_keys] for d in X_dicts])
            std_dev = np.mean(np.std(X, axis=0))
            diversity.append(std_dev)

        plt.figure()
        plt.title("Population diversity (average std dev of design vectors)")
        plt.plot(n_evals, diversity, "-x")
        plt.grid(which='both')
        plt.minorticks_on()
        plt.xlabel("Total number of function evaluations [-]")
        plt.ylabel("Average std deviation of the design variables [-]")
        plt.yscale('log')
        plt.tight_layout()

        # Visualise the maximum change in design vectors from one generation to the next
        max_change = [0]  # First generation has no predecessor so change is zero. 
        for i in range(1, len(res.history)):
            # Get current and previous populations' design vectors
            # Sorts X_current and X_prev based on the x_keys list derived earlier
            X_current = res.history[i].pop.get("X")
            X_current = np.array([[design_dict[k] for k in x_keys] for design_dict in X_current])
            X_prev = res.history[i - 1].pop.get("X")
            X_prev = np.array([[design_dict[k] for k in x_keys] for design_dict in X_prev])
            
            # For each design vector in the current generation, find the minimum Euclidean distance to any design vector in the previous generation.
            # This enables us to compute the maximum change even if the population size changes with generations. 
            # Process the design vectors in chunks to improve memory efficiency.
            max_change_value = 0
            for x in X_current:
                min_d = np.inf
                for chunk in np.array_split(X_prev, 8):
                    diff = np.linalg.norm(chunk - x, axis=1)
                    min_d = min(min_d, diff.min())
                max_change_value = max(max_change_value, min_d)
            max_change.append(max_change_value)

        plt.figure()
        plt.title("Maximum change in design vectors between generations")
        plt.plot(n_evals, max_change, "-x")
        plt.grid(which='both')
        plt.minorticks_on()
        plt.xlabel("Total number of function evaluations [-]")
        plt.ylabel("Maximum design change (Euclidean norm) [-]")
        plt.yscale("log")
        plt.tight_layout()

        # Visualise the constraint violation 
        CV = []
        for e in res.history:
            constraints = e.pop.get("CV")
            max_violation = np.max(constraints)
            CV.append(max_violation)
        
        plt.figure()
        plt.title("Maximum constraint violation between generations")
        plt.plot(n_evals, CV, "-x")
        plt.grid(which='both')
        plt.minorticks_on()
        plt.xlabel("Total number of function evaluations [-]")
        plt.ylabel("Maximum normalised constraint violation [-]")
        plt.tight_layout()


    def PlotObjectiveSpace(self,
                           res: object) -> None:
        """
        Visualise the objective space for all feasible solutions.

        Parameters
        ----------
        - res : object
            The optimization result object containing the design vector of the optimized design.
        """

        # Collect the objective values of the complete evaluated solution set
        F_all = np.vstack([gen.pop.get("F") for gen in res.history])

        # Collect the constraint violations for all evaluated solutions
        CV_all = np.vstack([gen.pop.get("CV") for gen in res.history])
        
        # Select only the feasible designs
        feasible_mask = np.all(CV_all <= 0, axis=1)
        F_feasible = F_all[feasible_mask]

        # Create scatter plot of the objective space
        plot = Scatter(title="Objective space for the feasible evaluated solution set")	
        plot.add(res.history[0].pop[0].get("F"), marker="x", facecolor="blue", s=35, label="Reference Design")
        plot.add(F_feasible, facecolor='none', edgecolor='black', s=10, label="Evaluated solutions")
        plot.add(res.F, facecolor='red', s=20, label="Optimum solutions")
        plot.legend = True
        plot.show()


    def AnalyseDesignSpace(self,
                           res: object,
                           idx_list: list[int]
                           ) -> None:
        """
        Visualise the feasible design space for the design variables whose indices are given in idx_list. 

        Parameters
        ----------
        - res : object
            The optimization result object containing the design vector of the optimized design.
        - idx_list : list[int]
            A list of integers which correspond to the indices of the design variables which need to be plotted. 
            To determine which integer value correspond to which design variable, inspect the init_designvector class.            
        """

        # Collect all evaluated design vectors
        X_all_dicts = [design for gen in res.history for design in gen.pop.get("X")]

        # Collect the constraint violations for all evaluated solutions
        CV_all = np.array([design for gen in res.history for design in gen.pop.get("CV")])

        # Convert the feasible solution set to arrays for plotting
        # Selects only the design variables whose indices are given in idx_list. 
        keys = [f"x{i}" for i in idx_list]
        X_all_arr = np.array([[d[k] for k in keys] for d in X_all_dicts])
        
        # Select only the feasible design vectors
        feasible_mask = np.all(CV_all <=0, axis=1)
        X_feasible_arr = X_all_arr[feasible_mask]

        # Create parallel coordinate plot for the design variables
        pcp = PCP(labels=keys)
        pcp.add(X_feasible_arr)
        pcp.tight_layout = True
        pcp.show()

        # Create scatter plot for the design variables
        scatter = Scatter(labels=keys)
        scatter.add(X_feasible_arr).show()


    def CreateBladeGeometryPlots(self,
                                blading: list[list[dict[str, any]]],
                                design: list[list[list[dict[str, any]]]]) -> None:
        """
        Generate 2D and 3D plots of blade geometries for each stage the ducted fan design.
        This method visualizes the blade sections by constructing their geometry using provided blading and design data.

        For each stage marked for optimization, it creates:
          - A 2D plot showing the airfoil profiles at multiple radial stations.
          - A 3D plot displaying the full blade surface by stacking the airfoil sections along the span.

        The method utilizes geometry construction and transformation utilities from the fileHandlingMTFLO class and
        airfoil coordinate conversion from the _airfoil_param attribute.

        Parameters
        ----------
        - blading : list[list[dict[str, any]]]
            A list containing blading data for each stage. Each stage is represented as a list of dictionaries
            with geometric and aerodynamic properties at various radial stations.
        - design : list[list[list[dict[str, any]]]]
            A list containing design data for each stage. Each stage is represented as a list of lists of dictionaries
            with design parameters for the blade sections.

        Notes
        -----
        - Only stages specified in the global config.OPTIMIZE_STAGE list are plotted.
        - The method assumes the existence of external classes and configuration, such as fileHandlingMTFLO and config.
        - Plots are displayed using matplotlib.
        """

        # Define the number of radial sections to generate a plot for, the number of chordwise data points to use 
        # and their distribution. 
        n_points_radial = 16
        n_data = 120
        axial_points = (1 - np.cos(np.linspace(0, np.pi, n_data))) / 2

        # Initialize fileHandlingMTFLO class - we use the methods developed there to construct the blade shape
        # Initialized with random inputs since we will not be generating any outputs from the class. 
        # We just use the methods from the class to generate the geometry for plotting. 
        fh = fileHandlingMTFLO("*", 1)  

        # Create plot for each stage:
        for i in range(len(blading)):
            # Only compute the plots if the stage is to be optimized
            if config.OPTIMIZE_STAGE[i]:
                # Set up figures for 2D and 3D plotting
                fig2d, ax2d = plt.subplots(figsize=(12, 8))
                fig3d = plt.figure(figsize=(12, 8))
                ax3d = fig3d.add_subplot(111, projection='3d')

                # Construct the radial points at which we obtain the data. 
                radial_points = np.linspace(blading[i]["radial_stations"][0], 
                                            blading[i]["radial_stations"][-1], 
                                            n_points_radial,
                                            ) 
                
                # Define lists to store the section geometries
                x_data = []
                y_data = []
                r_data = []
                    
                # Compute the blade geometry interpolations
                blade_geometry: dict = fh.ConstructBlades(blading[i],
                                                          design[i])
                    
                # Loop over the blade span
                for r in radial_points:
                     # All parameters are normalised using the local chord length, so we need to obtain the local chord in order to obtain the dimensional parameters
                    local_chord = blade_geometry["chord_distribution"](r)
                    axial_coordinates = axial_points * local_chord
                
                    # Create complete airfoil representation from the camber and thickness distributions
                    camber_distribution = blade_geometry["camber_distribution"]((r, axial_points)) * local_chord
                    thickness_distribution = blade_geometry["thickness_distribution"]((r, axial_points)) * local_chord
                    upper_x, upper_y, lower_x, lower_y = self._airfoil_param.ConvertBezier2AirfoilCoordinates(axial_coordinates, 
                                                                                                              thickness_distribution, 
                                                                                                              axial_coordinates, 
                                                                                                              camber_distribution)

                    # Rotate the airfoil profile to the correct angle
                    # The blade pitch is defined with respect to the blade pitch angle at the reference radial station, and thus is corrected accordingly. 
                    blade_pitch = (blade_geometry["pitch_distribution"](r) + blading[i]["ref_blade_angle"] - blading[i]["reference_section_blade_angle"])
                    rotated_upper_x, rotated_upper_y, rotated_lower_x, rotated_lower_y  = fh.RotateProfile(blade_pitch,
                                                                                                            upper_x,
                                                                                                            lower_x,
                                                                                                            upper_y,
                                                                                                            lower_y)
                    
                    # Compute the local leading edge offset at the radial station from the provided interpolant
                    # Use it to offset the x-coordinates of the upper and lower surfaces to the correct position
                    LE_coordinate = blade_geometry["leading_edge_distribution"](r)
                    rotated_upper_x += LE_coordinate - rotated_upper_x[0]
                    rotated_lower_x += LE_coordinate - rotated_lower_x[0]

                    # Concatenate the upper and lower data sets
                    rotated_x = np.concatenate((rotated_upper_x, np.flip(rotated_lower_x)), axis=0)
                    rotated_y = np.concatenate((rotated_upper_y, np.flip(rotated_lower_y)), axis=0)
                    
                    # Plot the 2D profile on 2D axes
                    ax2d.plot(rotated_x, rotated_y)

                    # Append the section to the list for 3d plotting
                    x_data.append(rotated_x)
                    y_data.append(rotated_y)
                    r_data.append(np.full_like(rotated_x, r))

                    # Plot the blade section in the 3D plot
                    ax3d.plot(rotated_x, 
                              rotated_y,
                              np.full_like(rotated_x, r),  # Each section is defined at constant r
                              color='black')
                
                # Convert all data to arrays - this is needed to use the plot_surface method. 
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                r_data = np.array(r_data)

                # Plot the blade surface in 3D
                ax3d.plot_surface(x_data, 
                                  y_data, 
                                  r_data, 
                                  alpha=0.75)        

                # Format 2D plot
                ax2d.set_title("2D Projection of Blade Geometry at Each Radial Section")
                ax2d.set_xlabel("Axial Coordinate [m]")
                ax2d.set_ylabel("Thickness/Height Coordinate [m]")
                ax2d.minorticks_on()
                ax2d.grid(which='both')

                # Format 3D plot
                ax3d.set_title("3D Blade Geometry")
                ax3d.set_xlabel("Axial Coordinate [m]")
                ax3d.set_ylabel("Thickness/Height Coordinate [m]")
                ax3d.set_zlabel("Radial Coordinate [m]")
                ax3d.minorticks_on()
                ax3d.grid(which='both')

            plt.show()


    def main(self) -> None:
        """
        Main post-processing method. 
        """
        
        # Load in the results object and extract the population data to self
        res = self.load_res()
        self.ExtractPopulationData(res)

        # Visualise the convergence behavior of the solution
        self.GenerateConvergenceStatistics(res)
        plt.show()
        plt.close('all')
       
        # Visualise the objective space
        self.PlotObjectiveSpace(res)

        # Visualise the design space
        self.AnalyseDesignSpace(res,
                                [0, 2, 4, 5, 6, 7, 8, 9])

        # Plot the centerbody designs
        if config.OPTIMIZE_CENTERBODY:
            # First plot the complete final population
            self.CompareAxisymmetricGeometry(config.CENTERBODY_VALUES,
                                             self.CB_data)
            plt.show()
            plt.close('all')
            # Plot the optimum solution set
            self.CompareAxisymmetricGeometry(config.CENTERBODY_VALUES,
                                             self.CB_data_opt)
            plt.show()
            plt.close('all')

        # Plot the duct designs
        if config.OPTIMIZE_DUCT:
            # First plot the complete final population
            self.CompareAxisymmetricGeometry(config.DUCT_VALUES,
                                             self.duct_data)
            plt.show()
            plt.close('all')
            # Plot the optimum solution set
            self.CompareAxisymmetricGeometry(config.DUCT_VALUES,
                                             self.duct_data_opt)
            plt.show()
            plt.close('all')
        
        # Plot the optimised stage designs
        for i in range(len(config.OPTIMIZE_STAGE)):

            if config.OPTIMIZE_STAGE[i]:
                # First plot the complete final population
                self.CompareBladingData(config.STAGE_BLADING_PARAMETERS,
                                        self.blading_data)
                plt.show()
                plt.close('all')

                # Plot the optimum solution set
                self.CompareBladingData(config.STAGE_BLADING_PARAMETERS,
                                        self.blading_data_opt)
                plt.show()
                plt.close('all')

                # Plot the optimum solution set
                self.CompareBladeDesignData(reference_design=config.STAGE_DESIGN_VARIABLES,
                                            res=res,
                                            individual="opt")
                plt.show()
                plt.close('all')

                # Plot the optimum solution set
                for j in range(len(self.blading_data_opt)):
                    self.CreateBladeGeometryPlots(self.blading_data_opt[j],
                                                  self.design_data_opt[j])
                    plt.show()
                    plt.close('all')

if __name__ == "__main__":
    output = Path('Results/res_pop30_eval4000_250523111905788625.dill')

    processing_class = PostProcessing(fname=output)
    processing_class.main()