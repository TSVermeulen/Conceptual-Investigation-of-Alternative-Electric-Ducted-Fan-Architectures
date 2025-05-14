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
- V1.1: Added convergence property plotter. Untested for multi-objective data.  
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

# Ensure all paths are correctly setup
from utils import ensure_repo_paths
ensure_repo_paths()
 

# Import interfacing modules
import config
from Submodels.Parameterizations import AirfoilParameterization
from design_vector_interface import DesignVectorInterface

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
        
        # Initialize empty lists
        opt_CB_data = []
        opt_duct_data = []
        opt_blading_data = []
        opt_design_data = []

        # Create local instance of the vector interfacing class
        vec_interface = DesignVectorInterface()

        # Loop over the population members and deconstruct their design vectors
        for individual in res.pop:
            (centerbody_variables, 
             duct_variables, 
             blade_design_parameters, 
             blade_blading_parameters, 
             _) = vec_interface.DeconstructDesignVector(individual.X)
            opt_CB_data.append(centerbody_variables)
            opt_duct_data.append(duct_variables)
            opt_blading_data.append(blade_blading_parameters)
            opt_design_data.append(blade_design_parameters)
        
        # Write all data to self
        self.CB_data = opt_CB_data
        self.duct_data = opt_duct_data
        self.blading_data = opt_blading_data
        self.design_data = opt_design_data

        # Deconstruct the optimum set of solutions too 
        # Loop over the population members and deconstruct their design vectors
        opt_CB_data = []
        opt_duct_data = []
        opt_blading_data = []
        opt_design_data = []
        for individual in res.opt:
            (centerbody_variables, 
             duct_variables, 
             blade_design_parameters, 
             blade_blading_parameters, 
             _) = vec_interface.DeconstructDesignVector(individual.X)
            opt_CB_data.append(centerbody_variables)
            opt_duct_data.append(duct_variables)
            opt_blading_data.append(blade_blading_parameters)
            opt_design_data.append(blade_design_parameters)
        self.CB_data_opt = opt_CB_data
        self.duct_data_opt = opt_duct_data
        self.blading_data_opt = opt_blading_data
        self.design_data_opt = opt_design_data
    

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
        original_lower_y) = parameterization.ComputeProfileCoordinates([reference["b_0"],
                                                                        reference["b_2"],
                                                                        reference["b_8"],
                                                                        reference["b_15"],
                                                                        reference["b_17"]],
                                                                        reference)
        
        # Create grouped figure to compare the geometry between the reference and the optimised designs
        grouped_fig, ax1 = plt.subplots()
        
        # First plot the original geometry
        ax1.plot(np.concatenate((original_upper_x, np.flip(original_lower_x)), axis=0),
                np.concatenate((original_upper_y, np.flip(original_lower_y)), axis=0), 
                "k-.", 
                label="Original Geometry",
                )
        
        # Loop over all individuals in the final population and plot their geometries
        for i, geom in enumerate(optimised):                
            # Compute the optimised geometry
            (opt_upper_x, 
            opt_upper_y, 
            opt_lower_x, 
            opt_lower_y) = parameterization.ComputeProfileCoordinates([geom["b_0"],
                                                                       geom["b_2"],
                                                                       geom["b_8"],
                                                                       geom["b_15"],
                                                                       geom["b_17"]],
                                                                       geom)

            # Plot the optimised geometry
            ax1.plot(np.concatenate((opt_upper_x, np.flip(opt_lower_x)), axis=0),
                    np.concatenate((opt_upper_y, np.flip(opt_lower_y)), axis=0), 
                    label=f"Individual {i}",
                    )
            
            if individual:      
                # Create figure for the individual comparison plot
                plt.figure(f"Comparison for individual {i}")
                # plot the original geometry
                plt.plot(np.concatenate((original_upper_x, np.flip(original_lower_x)), axis=0),
                        np.concatenate((original_upper_y, np.flip(original_lower_y)), axis=0), 
                        "k-.", 
                        label="Original Geometry",
                        )
                plt.plot(np.concatenate((opt_upper_x, np.flip(opt_lower_x)), axis=0),
                        np.concatenate((opt_upper_y, np.flip(opt_lower_y)), axis=0), 
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
        
        Returns
        -------
        None
        """

        # First we compare the blading data
        # We use a bar chart to compare the "singular" values, and line charts to compare the sectional distributions
        # Construct the variables used to annotate the plot. These are "polished" versions of the keys
        variables = ["Root LE coordinate [m]",
                     "Reference Blade Angle [deg]",
                     "Blade Count [-]",
                     "RPS [-]",
                     "Blade Diameter [m]"]
        
        # Keys in the blading dictionaries of interest
        keys = ["root_LE_coordinate",
                "ref_blade_angle",
                "blade_count",
                "RPS_lst",
                "radial_stations"]
        
        # Construct figure for bar chart
        # Inside your CompareBladingData() method – updated bar chart section:
        for i in range(len(config.OPTIMIZE_STAGE)):
            if config.OPTIMIZE_STAGE[i]:
                plt.figure(f"Bar Chart with blading parameters for stage {i}")

                # Update your variables and keys lists accordingly:
                variables = [
                    "Root LE coordinate [m]",
                    "Reference Blade Angle [deg]",
                    "Blade Count [-]",
                    "RPS [-]",          # now a list in the data
                    "Blade Diameter [m]"
                ]
                # Note: for radial stations (or blade diameter in your case) you might still use the same logic.
                keys = [
                    "root_LE_coordinate",
                    "ref_blade_angle",
                    "blade_count",
                    "RPS_lst",          # key now holds a list of values
                    "radial_stations"
                ]

                num_indiv = len(optimised_blading)
                x = np.arange(len(keys))  
                # For scalar (single value) fields we use a base bar width:
                base_bar_width = 0.8 / num_indiv

                # Now loop over each key/variable slot
                for k, key in enumerate(keys):
                    if key == "radial_stations":
                        plt.bar(x[k], 
                                max(reference_blading[i][key]) * 2, 
                                width=base_bar_width, 
                                color='black',
                                hatch="//",
                                edgecolor="white")
                        for j, opt_vals in enumerate(optimised_blading):
                            opt_val = opt_vals[i][key]
                            opt_val = max(opt_val) * 2
                            plt.bar(x[k] + (j + 1) * base_bar_width,
                                    opt_val,
                                    width=base_bar_width,
                                    label=f"Individual {j}")
                            
                    elif key == "RPS_lst":
                        # For the reference data:
                        ref_rps = reference_blading[i][key]  # a list, e.g. [rps_val1, rps_val2, …]
                        num_rps = len(ref_rps)
                        # Divide the base bar width among each RPS sub-bar
                        sub_bar_width = base_bar_width / num_rps
                        # Plot each sub-bar (centered around x[k])
                        for r, r_val in enumerate(ref_rps):
                            # Compute an offset so that the group is centered:
                            offset = (r - (num_rps - 1) / 2) * sub_bar_width
                            plt.bar(x[k] + offset,
                                    r_val,
                                    width=sub_bar_width,
                                    label="Reference" if r == 0 else "", 
                                    color='black',
                                    hatch='//',
                                    edgecolor='white')

                        # Now do the same for each optimised individual:
                        for j, opt_vals in enumerate(optimised_blading):
                            opt_rps = opt_vals[i][key]
                            for r, opt_r_val in enumerate(opt_rps):
                                offset = (r - (num_rps - 1) / 2) * sub_bar_width
                                plt.bar(
                                    x[k] + offset + (j + 1) * base_bar_width,
                                    opt_r_val,
                                    width=sub_bar_width,
                                    label=f"Individual {j}" if r == 0 else ""
                                )
                    else:
                        # For the scalar values (or values that remain unchanged)
                        ref_val = reference_blading[i][key]
                        if key == "ref_blade_angle":
                            ref_val = np.rad2deg(ref_val)
                        plt.bar(x[k],
                                ref_val,
                                width=base_bar_width,
                                label="Reference",
                                color='black',
                                hatch='//',
                                edgecolor='white')
                        for j, opt_vals in enumerate(optimised_blading):
                            opt_val = opt_vals[i][key]
                            if key == "ref_blade_angle":
                                opt_val = np.rad2deg(opt_val)
                            plt.bar(x[k] + (j + 1) * base_bar_width,
                                    opt_val,
                                    width=base_bar_width,
                                    label=f"Individual {j}")
                            
                # Format the x-axis using the provided variables
                plt.xticks(x + (base_bar_width * num_indiv) / 2, variables, rotation=90)
                plt.title("Comparison of Reference vs Optimized Design Variables")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
                plt.grid(axis='y', which='both')
                plt.yscale('log')
                plt.minorticks_on()
                plt.tight_layout()

        # Next generate a few graphs to compare the sectional blading data
        marker_cycle = cycler(marker=['o', 's', '^', '<', 'v', '>', '*', '+'])
        color_cycle = cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for i in range(len(config.OPTIMIZE_STAGE)):
            if config.OPTIMIZE_STAGE[i]:
                fig, ax = plt.subplots(nrows=2,
                                       ncols=2,
                                       constrained_layout=True)
                fig.suptitle(f"Sectional blading data for stage {i}")

                # Set cyclers for marker and color
                ax[0,0].set_prop_cycle(marker_cycle * color_cycle)
                ax[0,1].set_prop_cycle(marker_cycle * color_cycle)
                ax[1,0].set_prop_cycle(marker_cycle * color_cycle)

                # Set grids on
                ax[0,0].minorticks_on()
                ax[0,0].grid(which='both')
                ax[1,0].minorticks_on()
                ax[1,0].grid(which='both')
                ax[0,1].minorticks_on()
                ax[0,1].grid(which='both')

                # Set x-axis label
                ax[0,0].set_xlabel("Radial coordinate [m]")
                ax[0,1].set_xlabel("Radial coordinate [m]")
                ax[1,0].set_xlabel("Radial coordinate [m]")
                    
                # First plot the reference data
                ax[0,0].plot(reference_blading[i]["radial_stations"],
                             reference_blading[i]["chord_length"],
                             label="Reference",
                             color='black',
                             marker="x",
                             ms=3)
                ax[0,0].set_title("Chord length distribution [m]")

                ax[0,1].plot(reference_blading[i]["radial_stations"],
                             np.rad2deg(reference_blading[i]["sweep_angle"]),
                             label="Reference",
                             color='black',
                             marker="x",
                             ms=3)
                ax[0,1].set_title("Sweep angle distribution [deg]")

                ax[1,0].plot(reference_blading[i]["radial_stations"],
                             np.rad2deg(reference_blading[i]["blade_angle"]),
                             label="Reference",
                             color='black',
                             marker="x",
                             ms=3)
                ax[1,0].set_title("Blade angle distribution [deg]")

                # Loop over the optimised individuals
                for j, opt_vals in enumerate(optimised_blading):
                    ax[0,0].plot(opt_vals[i]["radial_stations"],
                                 opt_vals[i]["chord_length"],
                                 label=f"Individual {j}",
                                 ms=3)
                    
                    ax[0,1].plot(opt_vals[i]["radial_stations"],
                                 np.rad2deg(opt_vals[i]["sweep_angle"]),
                                 label=f"Individual {j}",
                                 ms=3)
                    
                    ax[1,0].plot(opt_vals[i]["radial_stations"],
                                 np.rad2deg(opt_vals[i]["blade_angle"]),
                                 label=f"Individual {j}",
                                 ms=3)
                
                # Disable the 4th plot and use its place for the legend
                ax[1,1].axis('off')
                handles, labels = ax[0,0].get_legend_handles_labels()
                ax[1,1].legend(handles, labels, loc='center', ncol=2)


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
         lower_y) = self._airfoil_param.ComputeProfileCoordinates([design[section_idx]["b_0"],
                                                                   design[section_idx]["b_2"],
                                                                   design[section_idx]["b_8"],
                                                                   design[section_idx]["b_15"],
                                                                   design[section_idx]["b_17"]],
                                                                   design[section_idx])

        return upper_x, upper_y, lower_x, lower_y
    
    
    def CompareBladeDesignData(self,
                               reference_design: list[list[dict[str, Any]]],
                               res: object,
                               individual: int | str = "opt",
                               optimised_design: list[list[dict[str, Any]]] = None) -> None:
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
            # Check if res.X is a list of dictionaries
            if len(res.X) == 1:
                optimum_dict = res.X[0]
                (_, _, optimised_design, _, _) = DesignVectorInterface().DeconstructDesignVector(optimum_dict)
            else:
                # Otherwise, loop through each dictionary in the list and deconstruct it
                multi_optimum_designs = []
                for design_dict in res.X:
                    (_, _, design_opt, _, _) = DesignVectorInterface().DeconstructDesignVector(design_dict)
                    multi_optimum_designs.append(design_opt)
                    
                # For each optimum design, generate its own set of plots
                # Plot multiple optimum designs on the same axes per stage and radial slice.
                for i in range(len(config.OPTIMIZE_STAGE)):
                    if config.OPTIMIZE_STAGE[i]:
                        radial_coordinates = np.linspace(0, 1, config.NUM_RADIALSECTIONS[i])
                        for j, radial_coordinate in enumerate(radial_coordinates):
                            plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}_Stage{i}")
                                
                            # Define a color palette for the optimum designs:
                            colors = plt.cm.tab10(np.linspace(0, 1, len(multi_optimum_designs)))
                                
                            # Loop over each optimum design and plot on the same axes.
                            for opt_idx, current_opt_design in enumerate(multi_optimum_designs):
                                (upper_x_opt, upper_y_opt, lower_x_opt, lower_y_opt) = \
                                    self.ConstructBladeProfile(current_opt_design[i], j)
                                plt.plot(np.concatenate((upper_x_opt, np.flip(lower_x_opt))),
                                        np.concatenate((upper_y_opt, np.flip(lower_y_opt))),
                                        label=f"Optimised (Ind {opt_idx})",
                                        color=colors[opt_idx])
                                plt.plot((upper_x_opt + lower_x_opt) / 2,
                                        (upper_y_opt + lower_y_opt) / 2,
                                        color=colors[opt_idx])
                                
                            # Construct and plot the reference airfoil representation.
                            (upper_x_ref, upper_y_ref, lower_x_ref, lower_y_ref) = \
                                self.ConstructBladeProfile(reference_design[i], j)
                            plt.plot(np.concatenate((upper_x_ref, np.flip(lower_x_ref))),
                                    np.concatenate((upper_y_ref, np.flip(lower_y_ref))),
                                    "-.",
                                    color="tab:orange",
                                    label="Reference")
                            plt.plot((upper_x_ref + lower_x_ref) / 2,
                                    (upper_y_ref + lower_y_ref) / 2,
                                    color="tab:orange")
                                
                            plt.legend()
                            plt.title(f"Blade profile at r={round(radial_coordinate, 3)}R for Stage {i} (Multiple Optima)")
                            plt.minorticks_on()
                            plt.grid(which='both')
                            plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
                            plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
                            plt.tight_layout()
                # Once the multiple optimum designs have been plotted, we return early.
                return
        else:
            # When an individual index is provided, we expect the optimised_design to be supplied.
            if optimised_design is None:
                raise ValueError("'optimised_design' must be supplied when 'individual' is an int.")
            optimised_design = copy.deepcopy(optimised_design[individual])

        # Process a single optimum design (or an optimised design specified by an integer)
        for i in range(len(config.OPTIMIZE_STAGE)):
            if config.OPTIMIZE_STAGE[i]:
                radial_coordinates = np.linspace(0, 1, config.NUM_RADIALSECTIONS[i])
                for j, radial_coordinate in enumerate(radial_coordinates):
                    plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}_Stage{i}")
                    
                    # Construct and plot the optimised design profile and its camber line
                    (upper_x_opt, upper_y_opt, lower_x_opt, lower_y_opt) = self.ConstructBladeProfile(optimised_design[i], j)
                    plt.plot(np.concatenate((upper_x_opt, np.flip(lower_x_opt))),
                            np.concatenate((upper_y_opt, np.flip(lower_y_opt))),
                            label="Optimised",
                            color='tab:blue')
                    plt.plot((upper_x_opt + lower_x_opt) / 2,
                            (upper_y_opt + lower_y_opt) / 2,
                            color='tab:blue')
                    
                    # Construct and plot the reference design profile and its camber line
                    (upper_x_ref, upper_y_ref, lower_x_ref, lower_y_ref) = self.ConstructBladeProfile(reference_design[i], j)
                    plt.plot(np.concatenate((upper_x_ref, np.flip(lower_x_ref))),
                            np.concatenate((upper_y_ref, np.flip(lower_y_ref))),
                            "-.",
                            color="tab:orange",
                            label="Reference")
                    plt.plot((upper_x_ref + lower_x_ref) / 2,
                            (upper_y_ref + lower_y_ref) / 2,
                            color='tab:orange')
                    
                    # Format the plot
                    plt.legend()
                    plt.title(f"Blade profile at r={round(radial_coordinate, 3)}R for stage {i}, individual: {individual}")
                    plt.minorticks_on()
                    plt.grid(which='both')
                    plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
                    plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
                    plt.tight_layout()


        # # Switching logic if we should compare against the specified individual by integer or against the optimum design
        # if individual == "opt":
        #     optimum_vector = res.X
        #     (_, 
        #      _, 
        #      optimised_design, 
        #      _, 
        #      _) = DesignVectorInterface().DeconstructDesignVector(optimum_vector)
        # else:
        #     if optimised_design is None:
        #         raise ValueError("'optimised_design' must be supplied when 'individual' is an int.")
        #     optimised_design = copy.deepcopy(optimised_design[individual])

        # # Loop over all stages and compare against the reference design if the stage is optimised:
        # for i in range(len(config.OPTIMIZE_STAGE)):
        #     if config.OPTIMIZE_STAGE[i]:
        #         radial_coordinates = np.linspace(0, 1, config.NUM_RADIALSECTIONS[i])
                
        #         # Loop over the radial slices
        #         for j, radial_coordinate in enumerate(radial_coordinates):
        #             # Create plot figure
        #             plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}")
   
        #             # Create complete optimised airfoil representation
        #             (upper_x, 
        #              upper_y, 
        #              lower_x,
        #              lower_y) = self.ConstructBladeProfile(optimised_design[i],
        #                                                    j)

        #             # Plot the optimised profile
        #             plt.plot(np.concatenate((upper_x, np.flip(lower_x)), axis=0),
        #                      np.concatenate((upper_y, np.flip(lower_y)), axis=0),
        #                      label="Optimised",
        #                      color='tab:blue')
                    
        #             # Plot the optimised camber line
        #             plt.plot((upper_x + lower_x) / 2,
        #                      (upper_y + lower_y) / 2,
        #                      color='tab:blue')
   
        #             # Create complete reference airfoil representation
        #             (upper_x, 
        #              upper_y, 
        #              lower_x,
        #              lower_y) = self.ConstructBladeProfile(reference_design[i],
        #                                                    j)

        #             # Plot the reference profile
        #             plt.plot(np.concatenate((upper_x, np.flip(lower_x)), axis=0),
        #                      np.concatenate((upper_y, np.flip(lower_y)), axis=0),
        #                      "-.",
        #                      color="tab:orange",
        #                      label="Reference")
                    
        #             # Plot the reference camber line
        #             plt.plot((upper_x + lower_x) / 2,
        #                      (upper_y + lower_y) / 2,
        #                      color='tab:orange')
                    
        #             # Format plot and add legend
        #             plt.legend()
        #             plt.title(f"Blade profile comparison at r={round(radial_coordinate, 3)}R for stage {i}, individual: {individual}")
        #             plt.minorticks_on()
        #             plt.grid(which='both')
        #             plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
        #             plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
        #             plt.tight_layout()

    
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
        
        avg_objectives = []
        for e in res.history:
            F_data = e.pop.get("F")
            avg_objectives.append(np.mean(F_data, axis=0))

        avg_objectives = np.array(avg_objectives)

        plt.figure()
        plt.title("Optimum and average objective values over generations")

        # For multi-objective probems, we plot each objective separately.
        if avg_objectives.ndim > 1 and avg_objectives.shape[1] >1:
            n_obj = avg_objectives.shape[1]
            for i in range(n_obj):
                plt.plot(n_evals, generational_optimum[:, i], "-.x", label=f'Generational optimum for objective {i}')
                plt.plot(n_evals, avg_objectives[:,i], "-*", label=f"Generational average for objective {i}")      
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


    def ComputeHyperVolume(self)->float:
        """
        
        """
        raise NotImplementedError


    def PlotObjectiveSpace(self,
                           res: object) -> None:
        """
        
        """

        raise NotImplementedError
    

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

                self.CompareBladeDesignData(reference_design=config.STAGE_DESIGN_VARIABLES,
                                            res=res,
                                            individual="opt")
                plt.show()
                plt.close('all')
        

if __name__ == "__main__":
    output = Path('Results/res_pop20_gen20_unsga3_moo_250514065954949106.dill')

    processing_class = PostProcessing(fname=output)
    processing_class.main()