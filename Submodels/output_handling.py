"""
output_handling
=============

Description
-----------
This module provides classes and methods to process and visualise the output of MTFLOW in terms of the flowfield and boundary layer data. 

Classes
-------
output_visualisation()
    A class to plot the streamline parameters and boundary layer data for the converged MTSOL case. 
output_processing()
    A class responsible for the post-processing of the MTFLOW output data.

Examples
--------
>>> test = output_visualisation(analysis_name='test_case')
>>> create_individual_plots = False
>>> test.PlotOutputs(plot_individual=create_individual_plots)

Notes
-----
The CreateBoundaryLayerPLots() method is only executed if the boundary_layer.analysis_name file exists in the local working directory. 

References
----------
None

Versioning
------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.0

Changelog:
- V1.0: Initial working version, containing only the plotting capabilities based on the flowfield.analysis_name and boundary_layer.analysis_name files. The output_processing() class is still a placeholder. 
"""

import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class output_visualisation:
    """
    This class handles the visualization of flowfield and boundary layer data from MTFLOW analysis.

    Methods
    -------
    - __init__(self, analysis_name: str = None) -> None
        Initializes the output_visualisation class with the given analysis name.
    
    - GetFlowfield(self) -> tuple[list[pd.DataFrame], pd.DataFrame]
        Loads the flowfield data from the flowfield.analysis_name file and returns it as a list of DataFrames for each streamline and a combined DataFrame.
    
    - GetBoundaryLayer(self) -> list[pd.DataFrame]
        Loads the boundary layer data from the boundary_layer.analysis_name file and returns it as a list of DataFrames for each surface.
    
    - ReadGeometry(self) -> list[np.ndarray]
        Reads the geometry data from the walls.analysis_name file and returns it as a list of numpy arrays for each axisymmetric body.
    
    - CreateContours(self, df: pd.DataFrame, shapes: list) -> None
        Creates contour plots for each parameter in the flowfield data and overlays the axisymmetric body shapes.
    
    - CreateStreamlinePlots(self, blocks: list[pd.DataFrame], plot_individual_streamlines: bool = False) -> None
        Creates streamline plots for each parameter in the flowfield data, with options to plot individual streamlines.
    
    - CreateBoundaryLayerPlots(self, blocks: list[pd.DataFrame]) -> None
        Creates plots for each boundary layer quantity for the axisymmetric surfaces.
    """

    # Define the columns from the flowfield file
    FLOWFIELD_COLUMNS = ['x', 'y', 'rho/rhoinf', 'p/pinf', 'u/Uinf', 'v/Uinf', 'Vtheta/Uinf', 
                         'q/Uinf', 'm/rhoinf Uinf', 'M', 'Cp', 'Cp0', '(q/Uinf)^2']
        
    # Define the columns from the boundary layer file
    BOUNDARY_LAYER_COLUMNS = ['x', 'r', 's', 'b0', 'Cp', 'Ue/Uinf', 'rhoe/rhoinf', 'Me', 'Hk', 'R_theta',
                              'delta*', 'theta', 'theta*', 'delta**', 'Cf/2', 'CD', 'ctau', 'm', 'P', 'K',
                              'Delta*', 'Theta', 'Theta*', 'Delta**', 'Gl', 'Gt']


    def __init__(self, 
                 analysis_name: str = None) -> None:
        """
        Initialize the output_visualisation class.
        
        This method sets up the initial state of the class.

        Parameters
        ----------
        - analysis_name : str
            A string of the analysis name. Must equal the filename extension used for walls.xxx, tflow.xxx, tdat.xxx, boundary_layer.xxx, and flowfield.xxx. 

        Returns
        -------
        None
        """ 

        

        # Simple input validation
        if analysis_name is None:
            raise IOError("The variable 'analysis_name' cannot be none in output_visualisation!")

        self.analysis_name = analysis_name

        # Write the local directory to self
        self.local_dir = Path(__file__).parent.resolve()

        # Validate if the required files exist
        self.flowfield_path = self.local_dir / f"flowfield.{self.analysis_name}"
        self.walls_path = self.local_dir / f"walls.{self.analysis_name}"

        if not os.path.exists(self.flowfield_path) or not os.path.exists(self.walls_path):
            raise FileNotFoundError(f"One of the required files flowfield.{self.analysis_name} or walls.{self.analysis_name} was not found.")

        # Check if the boundary layer file exists, and if so, set viscous_exists to True
        boundary_layer_path = self.local_dir / f"boundary_layer.{self.analysis_name}"
        if os.path.exists(boundary_layer_path):
            self.viscous_exists = True
        else:
            self.viscous_exists = False
        
        # Set the maximum number of figures that can be opened before raising a warning
        plt.rcParams['figure.max_open_warning'] = 100


    def GetFlowfield(self) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """
        Load in the flowfield.analysis_name file and write it to a Pandas dataframe. 

        Returns
        -------
        - tuple[list, pd.DataFrame] :
            - block_dfs : list[pd.DataFrame]
                List of nested DataFrames of the flow variables for each streamline.
            - df : pd.DataFrame
                A Pandas DataFrame containing the flowfield values across all streamlines.
        """

        # Get the path for the flowfield.analysis_name file and read data
        flowfield_path = self.local_dir / f"flowfield.{self.analysis_name}"
        try:
            with open(flowfield_path, 'r') as file:
                data = file.read()
        except IOError as e:
            raise IOError(f"Failred to read the flowfield data: {e}") from e
        
        # Split the data into blocks for each streamline 
        blocks = data.strip().split('\n\n')
        all_data = []
        block_dfs = []

        # Load in the numbers but not text or comments        
        for block in blocks:
            block_data = []
            lines = block.strip().split('\n')
            for line in lines:
                if not line.startswith('#'):
                    all_data.append([float(x) for x in line.split()])
                    block_data.append([float(x) for x in line.split()])
            
            # Convert block data to DataFrame and add it to the list of block DataFrames
            block_df = pd.DataFrame(block_data, columns=self.FLOWFIELD_COLUMNS)
            block_dfs.append(block_df)

        #Construct the dataframe
        df = pd.DataFrame(all_data, columns=self.FLOWFIELD_COLUMNS)

        return block_dfs, df
    
    
    def GetBoundaryLayer(self) -> list[pd.DataFrame]:
        """
        Load in the boundary_layer.analysis_name file and write the data for each element to a Pandas dataframe. 

        Returns
        - list[pd.DataFrame] :
            A list of nested DataFrames with the viscous variables for each boundary layer. 
        """

        # Get the path for the boundary_layer.analysis_name file and read data
        flowfield_path = self.local_dir / f"boundary_layer.{self.analysis_name}"
        try:
            with open(flowfield_path, 'r') as file:
                data = file.read()
        except IOError as e:
            raise IOError(f"Failed to read the boundary layer data: {e}") from e
            
        # Split the data into blocks for each streamline 
        blocks = data.strip().split('\n\n')
        element_dfs = []

        # Load in the numbers but not text or comments        
        for block in blocks:
            element_data = []
            lines = block.strip().split('\n')
            for line in lines:
                if not line.startswith('#'):
                    element_data.append([float(x) for x in line.split()])
            
            # Convert block data to DataFrame and add it to the list of block DataFrames
            element_df = pd.DataFrame(element_data, columns=self.BOUNDARY_LAYER_COLUMNS)
            element_dfs.append(element_df)

        return element_dfs
    

    def ReadGeometry(self,
                     ) -> list:
        """
        Read in the centrebody and duct geometry from the walls.analysis_name file

        Returns
        -------
        - shapes : list[np.ndarray]
            A list of nested arrays, where each array contains the geometry of one of the axisymmetric bodies. 
        """

        # Get the path for the walls.analysis_name file and read the data
        walls_path = self.local_dir / f"walls.{self.analysis_name}"
        try:
            with open(walls_path, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            raise IOError(f"Failed to read the geometry data: {e}") from e

        # Initialize an empty shapes and current_shape list
        shapes = []
        current_shape = []

        # Start reading in data from the 3rd line onwards, as the first 2 lines do not contain geometry points
        for line in lines[2:]:
            if "999.0    999.0" in line:
                shapes.append(np.array(current_shape))
                current_shape = []
            else:
                current_shape.append([float(x) for x in line.split()])
        if current_shape:
            shapes.append(np.array(current_shape))

        return shapes
    

    def ReadBlades(self,
                   ) -> list:
        """
        Read the blade geometries from the tflow.analysis_name file.

        Returns
        -------
        - 
        """

        tflow_fpath = self.local_dir / f"tflow.{self.analysis_name}"
        try:
            with open(tflow_fpath, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            raise IOError(f"Failed to read the tflow file: {e}") from e

        # Create blade outlines for each blade row (i.e. each stage)
        stages_outlines = []

        current_outline = []
        in_section = False

        for line in lines:
            if line.strip() == "STAGE":
                if current_outline:
                    stages_outlines.append(np.array(current_outline))
                current_outline = []
                in_section = False
            elif line.strip() == "SECTION":
                in_section = True
            elif line.strip() == "END":
                in_section = False
                if current_outline:
                    stages_outlines.append(np.array(current_outline))
                    current_outline = []
            elif in_section:
                points = [float(x) for x in line.split()]
                # Extract only the leading (first) and trailing (last) points
                if not current_outline:
                    current_outline.append(points[:2])  # Leading point
                current_outline.append(points[:2])  # Trailing point

        # Append the last outline if any
        if current_outline:
            stages_outlines.append(np.array(current_outline))

        return stages_outlines
    

    def CreateContours(self,
                       df: pd.DataFrame,
                       shapes: list[np.ndarray[float]],
                       blades = list[np.ndarray[float]],
                       figsize: tuple[float, float] = (6.4, 4.8),
                       cmap: str = 'viridis',
                       ) -> None:
        """
        Create contour plots for every parameter in the flowfield.analysis_name file. 
        Plots the axisymmetric bodies in dimgrey to generate the complete flowfield. 

        Parameters
        ----------
        - df : pd.DataFrame
            The dataframe of the complete flowfield.
        - shapes : list[np.ndarray[float]]
            A nested list with the coordinates of all the axisymmetric bodies.
        - blades : list
            A nested list with the coordinates of the outlines of the rotor/stator blades in the domain. 
        - figsize : tuple[float, float], optional
            A tuple with the figure size. Default value corresponds to the internal default of matplotlib.pyplot. 
        - cmap : str, optional
            A string with the colourmap to be used for the contourplots. Default value is the viridis colourmap. 

        Returns
        -------
        None
        """

        # Close any existing figures to free memory
        plt.close('all')
        
        # Create a contour plot for every variable
        for var in self.FLOWFIELD_COLUMNS[2:]:
            plt.figure(figsize=figsize)
            plt.tricontourf(df['x'], 
                            df['y'], 
                            df[var], 
                            levels=100, 
                            cmap=cmap,
                            )
            plt.colorbar(label=var + ' [-]')

            for shape in shapes:
                plt.fill(shape[:,0], shape[:,1], 'dimgrey')
            
            for blade in blades:
                plt.plot(blade[:,0], blade[:,1], 'k-.')

            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel('Radial coordinate $r/L_{ref}$ [-]')
            plt.ylim(bottom=0)
            plt.minorticks_on()
            plt.grid()
            plt.title(f'Contour Plot of {var}')
        
        plt.show()

    
    def CreateStreamlinePlots(self,
                              blocks: list[pd.DataFrame],
                              plot_individual_streamlines: bool = False,
                              ) -> None:
        """
        Plot the total, interior, exterior, and optional individual streamlines for all logged parameters. 

        Parameters
        ----------
        - plot_individual_streamlines : bool, optional
            A control boolean to determine if plots for each individual streamline should be generated. This is useful for debugging, but generates a very large amount of plots (11 plots times the number of streamlines, 45). 
            Default is False. 

        Returns
        -------
        None
        """

        # Close any existing figures to free memory
        plt.close('all')
        
        # Create streamline plots for all streamlines and all variables in self.FLOWFIELD_COLUMNS
        for param in self.FLOWFIELD_COLUMNS[2:]:  # Skipping x and y
            # Create plot window, define plot title and axis labels
            plt.figure()  
            plt.title(f"{param} streamline distribution")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot all streamlines
            for i, df in enumerate(blocks):
                plt.plot(df['x'], df[param], label=f'Streamline {i + 1}')
            
            # Set grid and minor ticks 
            plt.minorticks_on()
            plt.grid(which='both')
        
            # Create plot window for interior streamlines, define plot title and axis labels
            plt.figure()  
            plt.title(f"{param} interior streamline distribution")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot interior streamlines
            for i, df in enumerate(blocks):
                if (df["Vtheta/Uinf"].abs() > 0).any():
                    plt.plot(df['x'], df[param], label=f'Streamline {i + 1}')
            
            # Set grid and minor ticks 
            plt.minorticks_on()
            plt.grid(which='both')

            # Create plot window for exterior streamlines, define plot title and axis labels
            plt.figure()  
            plt.title(f"{param} exterior streamline distribution")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot exterior streamlines
            for i, df in enumerate(blocks):
                if not (df["Vtheta/Uinf"].abs() > 0).any():
                    plt.plot(df['x'], df[param], label=f'Streamline {i + 1}')
            
            # Set grid and minor ticks 
            plt.minorticks_on()
            plt.grid(which='both')
        
            #Show all streamline plots
            plt.show()
        
        if plot_individual_streamlines:
            # Create individual streamline plots for all variables in self.FLOWFIELD_COLUMNS
            for i,df in enumerate(blocks):
                if i != 0:
                    for param in self.FLOWFIELD_COLUMNS[2:]:  # Skipping x and y
                        # Create plot window, define plot tile and axis labels
                        plt.figure()
                        plt.title(f"{param} distribution for streamline {i + 1}")
                        plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
                        plt.ylabel(f'{param} [-]')

                        # Plot the streamline distribution
                        plt.plot(df['x'], df[param], ms=3, marker="x")

                        # Set grid and minor ticks
                        plt.minorticks_on()
                        plt.grid(which='both')
                    
                    #Show all plots for the streamline
                    plt.show()
    

    def CreateBoundaryLayerPlots(self,
                                 blocks : list[pd.DataFrame]) -> None:
        """
        Plot the boundary layer quantities for each of the axi-symmetric surfaces

        Parameters
        ----------
        - blocks : list[pd.DataFrame]
            A nested list of dataframes containing the boundary layer quantities for each surface. 
        
        Returns
        -------
        None
        """

        # Close any existing figures to free memory
        plt.close('all')

        # Create a plot for each boundary layer quantity, except the x and r coordinates. 
        for param in self.BOUNDARY_LAYER_COLUMNS[2:]:  # skip x and r 
            plt.figure()  
            plt.title(f"{param} boundary layer distributions")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot all streamlines
            for i, df in enumerate(blocks):
                plt.plot(df['x'], df[param], label=f'Surface {i + 1}', ms=3, marker="x")
                
            # Set grid and minor ticks 
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='both')
            
        plt.show()
        

    def PlotOutputs(self,
                    plot_individual: bool = False,
                    ) -> None:
        """ 
        Generate all output plots for the analysis.

        Parameters
        ----------
        - plot_individual : bool, optional
            A controlling boolean to determine if plots for each individual streamline should be generated. 
            Default value is False. 
        """
    
        # Load in the flowfield into blocks for each streamline and an overall dataframe
        blocks, df = self.GetFlowfield()

        # Read in the axi-symmetric geometry
        bodies = self.ReadGeometry()

        # Read in the blade outlines
        blades = self.ReadBlades()

        # Create contour plots from the flowfield
        self.CreateContours(df, bodies, blades)

        # Create the streamline plots
        self.CreateStreamlinePlots(blocks,
                                   plot_individual_streamlines=plot_individual)
        
        # Load in the boundary layer data and create the boundary layer plots if a boundary layer data file exists
        if self.viscous_exists:
            boundary_layer_blocks = self.GetBoundaryLayer()
            self.CreateBoundaryLayerPlots(boundary_layer_blocks)

    
class output_processing:
    """
    A class responsible for post-processing MTFLOW output data.

    TODO
    ----
    - Implement methods for data analysis
    - Add functionality for:
        - Statistical analysis
        - Performance metrics calculations/processing
        - Data export capabilitites
    
    Notes
    -----
    - This class is currently a placeholder and will be implemented in future versions.
    """

    def __init__(self,
                 analysis_name: str = None):
        # TODO: Implement initialization

        # Simple input validation
        if analysis_name is None:
            raise IOError("The variable 'analysis_name' cannot be none in output_visualisation!")

        self.analysis_name = analysis_name

        # Write the local directory to self
        self.local_dir = Path(__file__).parent.resolve()

        # Validate if the required forces file exist
        self.forces_path = self.local_dir / f"forces.{self.analysis_name}"

        if not os.path.exists(self.forces_path):
            raise FileNotFoundError(f"The required file forces.{self.analysis_name} was not found.")

        pass


    def GetCTCPEtaP(self,
                ) -> tuple[float, float]:
        """
        Read the forces.analysis_name file and return the thrust and power coefficients with the propulsive efficiency.

        Parameters
        ----------
        None

        Returns
        -------
        - tuple[float, float, float]
            A tuple of the form (CT, CP, EtaP) containing the thrust and power coefficients, together with the propulsive efficiency for the analysed case
        """

        try:
            with open(self.forces_path, 'r') as file:
                forces_file_contents = file.readlines()
                forces_file_contents = ''.join(forces_file_contents)
        except OSError as e:
            raise OSError(f"An error occurred opening the forces.{self.analysis_name} file: {e}") from e
        
        # Define regex pattern to extract CP, CT, and EtaP
        # pattern accepts both scientific notation and regular float notation. 
        pattern = r"Total power\s+CP\s+=\s+([-\d.]+(?:E[-+]?\d+)?)\s+EtaP\s+=\s+([-\d.]+(?:E[-+]?\d+)?)\s+Total force\s+CT\s+=\s+([-\d.]+(?:E[-+]?\d+)?)"

        # Search for the pattern and extract the data
        match = re.search(pattern, forces_file_contents)
        if match is None:
            raise ValueError(f"Failed to extract the CP, CT, and EtaP values from the forces.{self.analysis_name} file.")
        
        total_CP = float(match.group(1))
        EtaP = float(match.group(2))
        total_CT = float(match.group(3))

        return total_CT, total_CP, EtaP
       

if __name__ == "__main__":
    # Example usage for the output_visualisation class
    test = output_visualisation(analysis_name='x22a_validation')

    create_individual_plots = True
    test.PlotOutputs(plot_individual=create_individual_plots)

    # Example usage for the output_processing class 
    test = output_processing(analysis_name='test_case')
    test.GetCTCPEtaP()