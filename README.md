   GArNet can optimize mutation combination by adopting the following calculation steps: 
 running short generation of virtual evolution by GAOptimizer and selection of mutations 
 by summation of the results based on network theory.The mutation candidates were selected
 from the libraries containing homologs of sequence of target protein. 
 Currently, three selection pressure (HiSol, REU and both of the scores) can be used for the optimization (20250210). 
 The program was developed by H. Ozawa.

 Required Python Libraries and Dependencies
•	Python 3.11.7
•	Pyrosetta4: Rosetta’s Python interface (requires a Rosetta license and installation).
•	BioPython (version 1.83) (Bio): For parsing sequence alignments. 
•	Numpy (version 1.26.4): For numerical computations. 
•	Networkx (version 3.1): For network analysis. 
•	Matplotlib (version 3.8): For network visualization.
•	MAFFT (version 7.525) (external tool, must be installed on the system for sequence alignment).
•	Scipy (version 1.11.4), if not included with numpy/matplotlib distributions, may be required for polyfit (linear regression in network analysis steps).


 
