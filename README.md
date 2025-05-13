# OpenMM MD Simulation of a Lipid Bilayer System 

This assembly was built from CHARMM-GUI (www.charmm-gui.org) using the Membrane Builder module. With the job script file `run_job.sh`, we can run simulations with OpenMM program. 

# First convert mol2 to required mol2 and str using this website

https://app.cgenff.com/login

# conver downloaed mol2 to pdb using below code

obabel -imol2 Cholestrol.mol2 -opdb -O Cholestrol.pdb

# change the input file names in run_... for input of interest

# run run_merged_simulation.py

# run analyze_ligand_position.py