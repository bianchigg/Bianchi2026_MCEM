This is the code and data associated with the paper Modelling the Geomorphological Changes of a 
Supraglacial Meltwater Channel, G Bianchi et al. (2026).

Note that to run many of these scripts you will need to change the path.
There are 3 folders, which are as follows:

1. MCEM_files. This folder contains all the code required to run the model.
The main meltwater channel evolution model in is MCEM_1_1.py. This function relies on each of the other 3 scripts in the folder. Cfunctions.py is a script of helper functions used in MCEM_1_1 and elsewhere.
IceWallMeltFunctionGhost.py contains the functions which evolve the temperature of the ice column and output melt rates for a given point.
ObjectBoundary.py is a script which helps with defining the channels initially.

2. Plotting routines. This folder contains all the plotting scripts to reproduce the figures in the paper.
PlotScriptPosition.py is to show how to plot any given run from the pre-run data, and isn't used to create a plot in the paper.
Commented out at the start of this script is an example of how you can run the script yourself.
PlotSideBySideHPCruns.py is a script that produces the channel plots in the paper and a few others.
SolarRadiationPlots.py produces the variation of solar radiation based on the surface tilt plot.

3. StreamVelocities. Contains a script and data. The data is from the Greenland field season in 2024 on Isunguta Sermia and has velocity
profiles across a supraglacial channel. This data was augmented by a no-slip (0 velocity) condition on the boundary.
The python script plots this data and retrieves relevant properties which are presented in the table in the paper.

The prerun data to create the scripts is too large to upload to GitHub. When the paper is published, it will be made available on Zenodo.
