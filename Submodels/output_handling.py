"""


"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class output_handling:
    """
    
    """


    def __init__(self, 
                 streamline_number: int = None,
                 plotted_variable: str = None,
                 analysis_name: str = None) -> None:
        """
        
        """ 

        self.analysis_name = analysis_name

        # Write the local directory to self
        self.local_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Write the columns from the flowfield file to self
        self.flowfield_columns = ['x', 'y', 'rho/rhoinf', 'p/pinf', 'u/Uinf', 'v/Uinf', 'Vtheta/Uinf', 
                                  'q/Uinf', 'm/rhoinf Uinf', 'M', 'Cp', 'Cp0', '(q/Uinf)^2']
        
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
        with open(flowfield_path, 'r') as file:
            data = file.read()
        
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
            block_df = pd.DataFrame(block_data, columns=self.flowfield_columns)
            block_dfs.append(block_df)

        #Construct the dataframe
        df = pd.DataFrame(all_data, columns=self.flowfield_columns)

        return block_dfs, df
    

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
        with open(walls_path, 'r') as file:
            lines = file.readlines()

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
    

    def CreateContours(self,
                       df,
                       shapes,
                       ) -> None:
        """
        Create contour plots for every parameter in the flowfield.analysis_name file. Plots the axisymmetric bodies in dimgrey to generate the complete flowfield. 

        Parameters
        ----------
        - df : pd.DataFrame
            The dataframe of the complete flowfield.
        - shapes : list
            A nested list with the coordinates of all the axisymmetric bodies.

        Returns
        -------
        None
        """

        variables = df.columns[2:]  # Select variables excluding 'x' and 'y'
        
        # Create a contour plot for every variable
        for var in variables:
            plt.figure()
            plt.tricontourf(df['x'], 
                            df['y'], 
                            df[var], 
                            levels=100, 
                            cmap='viridis',
                            )
            plt.colorbar(label=var + ' [-]')

            for shape in shapes:
                plt.fill(shape[:,0], shape[:,1], 'dimgrey')

            plt.xlabel('Axial coordinate $x$ [m]')
            plt.ylabel('Radial coordinate $r$ [m]')
            plt.ylim(bottom=0)
            plt.minorticks_on()
            plt.grid()
            plt.title(f'Contour Plot of {var}')
        
        plt.show()

    
    def CreateStreamlinePlots(self,
                              blocks: list[pd.DataFrame],
                              ) -> None:
        """
        
        """
        
        # Create streamline plots for all streamlines and all variables in self.flowfield_columns
        for param in self.flowfield_columns[2:]:  # Skipping x and y
            # Create plot window, define plot title and axis labels
            plt.figure()  
            plt.title(f"{param} streamline distribution")
            plt.xlabel('Axial coordinate $x$ [m]')
            plt.ylabel(f'{param} [-]')

            # Plot all streamlines
            for i, df in enumerate(blocks):
                plt.plot(df['x'], df[param], label=f'Streamline {i + 1}', ms=1, marker="x")
            
            # Set grid and minor ticks 
            plt.minorticks_on()
            plt.grid(which='both')
        
            # Create plot window for interior streamlines, define plot title and axis labels
            plt.figure()  
            plt.title(f"{param} interior streamline distribution")
            plt.xlabel('Axial coordinate $x$ [m]')
            plt.ylabel(f'{param} [-]')

            # Plot interior streamlines
            for i, df in enumerate(blocks):
                if i < (len(blocks) / 2 - 3):
                    plt.plot(df['x'], df[param], label=f'Streamline {i + 1}', ms=1, marker="x")
            
            # Set grid and minor ticks 
            plt.minorticks_on()
            plt.grid(which='both')

            # Create plot window for exterior streamlines, define plot title and axis labels
            plt.figure()  
            plt.title(f"{param} exterior streamline distribution")
            plt.xlabel('Axial coordinate $x$ [m]')
            plt.ylabel(f'{param} [-]')

            # Plot interior streamlines
            for i, df in enumerate(blocks):
                if i > (len(blocks) / 2 - 3):
                    plt.plot(df['x'], df[param], label=f'Streamline {i + 1}', ms=1, marker="x")
            
            # Set grid and minor ticks 
            plt.minorticks_on()
            plt.grid(which='both')
        
        #Show all streamline plots
        plt.show()
        
        # Create individual streamline plots for all variables in self.flowfield_columns
        for i,df in enumerate(blocks):
            if i != 0:
                for param in self.flowfield_columns[2:]:  # Skipping x and y
                    # Create plot window, define plot tile and axis labels
                    plt.figure()
                    plt.title(f"{param} distribution for streamline {i + 1}")
                    plt.xlabel('Axial coordinate $x$ [m]')
                    plt.ylabel(f'{param} [-]')

                    # Plot the streamline distribution
                    plt.plot(df['x'], df[param], ms=1, marker="x")

                    # Set grid and minor ticks
                    plt.minorticks_on()
                    plt.grid(which='both')
                
                #Show all plots for the streamline
                plt.show()


if __name__ == "__main__":
    # Example usage
    file_path = 'flowfield.test_case'  # Replace with your actual file path
    test = output_handling(analysis_name='test_case')

    blocks, df = test.GetFlowfield()
    shapes = test.ReadGeometry()
    test.CreateContours(df, shapes)
    test.CreateStreamlinePlots(blocks)