# Conceptual Investigation of Alternative Electric Ducted Fan Architectures
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/TSVermeulen/Conceptual-Investigation-of-Alternative-Electric-Ducted-Fan-Architectures)

This GitHub repository contains the codebase developed for the Unified Ducted Fan Design and Analysis Code (UDFDAC) and its implementation into a Unified Non-dominated Sorting Genetic Algorithm III (U-NSGA-III) generalised mixed variable problem, as part of the MSc  thesis titled "INSERT TITLE HERE" at Delft University of Technology, Faculty of Aerospace Engineering. 

This thesis builds on the MTFLOW software developed by M. Drela to create a fast, robust, and accurate ducted fan analysis code. This code is implemented in the U-NSGA-III algorithm to enable design explorations for different operating conditions, objectives, and constraints. This repository also contains the validation data used to validate the implementation against experimental wind tunnel data of the X-22A ducted propulsor. This wind tunnel data is reported in NASA-TN-D-4142. The validation data is contained in the validation folder. This folder also contains a partial implementation of validation against the TU Delft XPROP. However, this was never completed, as the adopted parameterisation cannot model negative camber. 

For a detailed description of the developed methods and results, the reader is referred to the thesis, which is publicly available: **INSERT LINK HERE**

As per the License for MTFLOW, the MTFLOW codes cannot be freely distributed.
Should the reader wish to use the developed methods in this thesis, they need to request a license for MTFLOW directly from the MIT Technology Licensing Office. 
The code in this repository is designed to work on Windows. For a Linux/Unix-like system, the MTFLOW executable filenames and filepaths need to be adjusted accordingly.

For best performance, it is recommended to run the genetic algorithm on a computer with as many CPU cores/threads as possible, since each core can be used to run 1 or 2 analyses simultaneously, depending on the computational power of the machine. Testing of the developer shows 16 analyses can be conducted simultaneously on an AMD Ryzen 5xxx 8-core/16-thread CPU. 
