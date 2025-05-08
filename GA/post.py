"""
post
====

Description
----------
This module provides post-processing functionality for analyzing and visualizing the results of a Pymoo optimization. 
It includes methods for loading optimization results, extracting population data, and generating comparative plots 
for axisymmetric geometries, blading data, and blade design profiles.

Classses
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
Version 1.0

Changelog:
- V1.0: Initial implementation of plotting capabilities of outputs. 
"""

# Import standard libraries
import dill
from pathlib import Path
from cycler import cycler

# Import 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter

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


    def __init__(self,
                 fname: Path,
                 base_dir: Path = None) -> None:
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

        # Convert fname to Path and resolve it if it's not alredy absolute
        self.results_path = self.base_dir / fname if not fname.is_absolute() else fname
        
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

    
    def CompareAxisymmetricGeometry(self,
                                    reference: dict[str, any],
                                    optimised: list[dict[str, any]],
                                    individual: bool = False) -> None:
        """
        Generate plots of the original and optimised (normalised) axisymmetric profiles.

        Parameters
        ----------
        - reference : dict[str, any]
            The reference geometry.
        - optimised : list[dict[str, any]]
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
        for i in range(len(optimised)):                  
            # Compute the optimised geometry
            (opt_upper_x, 
            opt_upper_y, 
            opt_lower_x, 
            opt_lower_y) = parameterization.ComputeProfileCoordinates([optimised[i]["b_0"],
                                                                       optimised[i]["b_2"],
                                                                       optimised[i]["b_8"],
                                                                       optimised[i]["b_15"],
                                                                       optimised[i]["b_17"]],
                                                                       optimised[i])

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
        
        ax1.grid(which='both')
        ax1.minorticks_on()
        ax1.set_xlabel("Axial Coordinate [m]")
        ax1.set_ylabel("Radial Coordinate [m]")
        ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
        grouped_fig.tight_layout()


    def CompareBladingData(self,
                         reference_blading: list[dict[str, any]],
                         optimised_blading: list[list[dict[str, any]]]) -> None:
        """
        Generate plots of the blading data for the final population members and the initial reference design. 

        Parameters
        ----------
        - reference_blading : list[dict[str, any]]
            The reference blading data. Each dictionary in the list corresponds to a stage. 
        - optimised_blading : list[dict[str, any]]
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
                "RPS",
                "radial_stations"]
        
        # Construct figure for bar chart
        for i in range(len(config.OPTIMIZE_STAGE)):
            if config.OPTIMIZE_STAGE[i]:
                plt.figure("Bar Chart with blading parameters for stage {i}")
                num_vars = len(keys)
                num_indiv = len(optimised_blading)
                x = np.arange(num_vars)
                bar_width = 1 / (2 * num_indiv)

                # Plot the reference bars
                reference_bar_data = [reference_blading[i][key] if key != "radial_stations" 
                                    else (max(reference_blading[i][key]) * 2) for key in keys]
                reference_bar_data[1] = np.rad2deg(reference_bar_data[1])
                plt.bar(x,
                        reference_bar_data,
                        width=bar_width,
                        label='Reference',
                        color='black',
                        hatch='//',
                        edgecolor='white')
                
                # Plot the optimised bars
                for j, opt_vals in enumerate(optimised_blading):
                    optimised_bar_data = [opt_vals[i][key] if key != "radial_stations" 
                                          else (max(opt_vals[i][key]) * 2) for key in keys]
                    optimised_bar_data[1] = np.rad2deg(optimised_bar_data[1])
                    plt.bar(x + (j + 1) * bar_width, 
                            optimised_bar_data,
                            width=bar_width,
                            label=f"Individual {j}")
        
                # Labels and formatting
                plt.xticks(x + bar_width, variables, rotation=90)
                plt.title("Comparison of Reference vs Optimized Design Variables")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
                plt.grid(axis='y',
                        which='both')
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
                              design:list[dict[str, any]],
                              section_idx: int) -> tuple:
        """
        Function to compute the rotated upper and lower profile coordinates for a blade section.

        Parameters
        ----------
        - design : list[dict[str, any]]
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
         lower_y) = AirfoilParameterization().ComputeProfileCoordinates([design[section_idx]["b_0"],
                                                                         design[section_idx]["b_2"],
                                                                         design[section_idx]["b_8"],
                                                                         design[section_idx]["b_15"],
                                                                         design[section_idx]["b_17"]],
                                                                         design[section_idx])

        return upper_x, upper_y, lower_x, lower_y
    
    
    def CompareBladeDesignData(self,
                               reference_design: list[list[dict[str, any]]],
                               res: object,
                               individual: int | str = "opt",
                               optimised_design: list[list[dict[str, any]]] = None) -> None:
        """
        Compares the blade design data of a reference design with an optimized design 
        and generates plots for visual comparison at various radial sections.

        Parameters
        ----------
        - reference_design : list[list[dict[str, any]]] 
            The reference blade design data, structured as a list of stages, 
            where each stage contains a list of dictionaries representing radial sections.
        - res : object 
            The optimization result object containing the design vector of the optimized design.
        - individual : int | str, optional 
            Specifies which individual design to compare against. If "opt", the optimum design 
            from the optimization result is used. If an integer, the corresponding individual 
            from the `optimised_design` list is used. Defaults to "opt".
        - optimised_design list[list[dict[str, any]]], optional 
            The optimized blade design data, structured similarly to `reference_design`. 
            Required if `individual` is an integer. Defaults to None.
                            
        Returns
        -------
        None
        """

        # Switching logic if we should compare against the specified individual by integer or against the optimum design
        if isinstance(individual, str):
            optimum_vector = res.X
            (_, 
             _, 
             optimised_design, 
             _, 
             _) = DesignVectorInterface().DeconstructDesignVector(optimum_vector)
        else:
            optimised_design = optimised_design[individual]

        # Loop over all stages and compare against the reference design if the stage is optimised:
        for i in range(len(config.OPTIMIZE_STAGE)):
            if config.OPTIMIZE_STAGE[i]:
                radial_coordinates = np.linspace(0, 1, config.NUM_RADIALSECTIONS[i])
                
                # Loop over the radial slices
                for j, radial_coordinate in enumerate(radial_coordinates):
                    # Create plot figure
                    plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}")
   
                    # Create complete optimised airfoil representation
                    (upper_x, 
                     upper_y, 
                     lower_x,
                     lower_y) = self.ConstructBladeProfile(optimised_design[i],
                                                           j)

                    # Plot the optimised profile
                    plt.plot(np.concatenate((upper_x, np.flip(lower_x)), axis=0),
                             np.concatenate((upper_y, np.flip(lower_y)), axis=0),
                             label="Optimised",
                             color='tab:blue')
                    
                    # Plot the optimised camber line
                    plt.plot((upper_x + lower_x) / 2,
                             (upper_y + lower_y) / 2,
                             color='tab:blue')
   
                    # Create complete reference airfoil representation
                    (upper_x, 
                     upper_y, 
                     lower_x,
                     lower_y) = self.ConstructBladeProfile(reference_design[i],
                                                           j)

                    # Plot the reference profile
                    plt.plot(np.concatenate((upper_x, np.flip(lower_x)), axis=0),
                             np.concatenate((upper_y, np.flip(lower_y)), axis=0),
                             "-.",
                             color="tab:orange",
                             label="Reference")
                    
                    # Plot the reference camber line
                    plt.plot((upper_x + lower_x) / 2,
                             (upper_y + lower_y) / 2,
                             color='tab:orange')
                    
                    # Format plot and add legend
                    plt.legend()
                    plt.title(f"Blade profile comparison at r={round(radial_coordinate, 3)}R for stage {i}, individual: {individual}")
                    plt.minorticks_on()
                    plt.grid(which='both')
                    plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
                    plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
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

        # Plot the centerbody designs
        if config.OPTIMIZE_CENTERBODY:
            self.CompareAxisymmetricGeometry(config.CENTERBODY_VALUES,
                                             self.CB_data)
            plt.show()
            plt.close('all')

        # Plot the duct designs
        if config.OPTIMIZE_DUCT:
            self.CompareAxisymmetricGeometry(config.DUCT_VALUES,
                                             self.duct_data)
            plt.show()
            plt.close('all')
        
        # Plot the optimised stage designs
        for i in range(len(config.OPTIMIZE_STAGE)):
            if config.OPTIMIZE_STAGE[i]:
                self.CompareBladingData(config.STAGE_BLADING_PARAMETERS,
                                        self.blading_data)
                plt.show()
                plt.close('all')

                self.CompareBladeDesignData(config.STAGE_DESIGN_VARIABLES,
                                            config.STAGE_BLADING_PARAMETERS,
                                            res,
                                            'opt')
                plt.show()
                plt.close('all')
        

if __name__ == "__main__":
    output = Path('res_pop20_gen20_250506220150593712.dill')

    processing_class = PostProcessing(fname=output)
    processing_class.main()
