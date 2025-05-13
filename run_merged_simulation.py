#!/usr/bin/env python
"""
Merges a ligand into a lipid bilayer system and runs an OpenMM simulation.

This script automates the following steps:
1. Checks and updates the toppar.str file to include the ligand parameters.
2. Merges the bilayer PSF/CRD and ligand PDB files using ParmEd.
3. Runs an OpenMM simulation using the merged system and existing CHARMM-GUI helper scripts.
"""

import os
import sys
import parmed as pmd
from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

# --- Import CHARMM-GUI Helper Modules ---
# Assume they are in the same directory or Python path
try:
    from omm_readinputs import read_inputs
    from omm_readparams import read_psf, read_crd, read_params, gen_box # Use gen_box if sysinfo isn't provided
    from omm_vfswitch import vfswitch
    from omm_barostat import barostat
    from omm_restraints import restraints
    from omm_hmr import HydrogenMassRepartition
    from omm_rewrap import rewrap
except ImportError as e:
    print(f"Error importing helper modules: {e}")
    print("Ensure omm_*.py files are in the current directory or Python path.")
    sys.exit(1)

# ---------- USER-CONFIGURABLE SETTINGS ----------
# Input Files
bilayer_psf_file = "step5_charmm2omm.psf"
bilayer_crd_file = "step5_charmm2omm.crd"
ligand_pdb_file  = "Cholestrol.pdb"
ligand_str_file  = "Cholestrol.str" # Parameter file for the ligand
toppar_file      = "toppar.str"          # Master parameter stream file
input_recipe_file= "step6.1_equilibration.inp" # Simulation settings (like nsteps, temp)

# Merged Files (Intermediate)
merged_psf_file = "merged.psf"
merged_pdb_file = "merged.pdb" # Using PDB for coordinates for simplicity with OpenMM

# Output Files Prefix
output_prefix   = "merged_equil" # e.g., merged_equil.rst, merged_equil.dcd

# Ligand Placement (Example: Translate 30 Angstroms along Z-axis)
# Set to None or [0,0,0] if no translation is needed relative to ligand PDB origin
ligand_translation_A = [10.0, 0.0, 55.0] * angstroms

# Simulation Run Options
apply_hmr = False # Set to True to apply Hydrogen Mass Repartitioning
rewrap_coords = False # Set to True to re-wrap coordinates periodically/finally

# Platform Selection
platform_name = 'CUDA' # Options: 'CUDA', 'OpenCL', 'CPU'
platform_properties = {'CudaPrecision': 'single'} # Or {'Precision': 'single'} for OpenCL
# --------------------------------------------------


# --- Main Script Logic ---

def update_toppar(toppar_path, ligand_stream_file):
    """Appends the ligand stream file to toppar.str if not already present."""
    ligand_stream_entry = f"stream {ligand_stream_file}"
    found = False
    if os.path.exists(toppar_path):
        with open(toppar_path, 'r') as f:
            for line in f:
                if ligand_stream_entry in line.strip():
                    found = True
                    break
    if not found:
        print(f"Adding '{ligand_stream_entry}' to {toppar_path}")
        with open(toppar_path, 'a') as f:
            f.write(f"""
* --- Added by run_merged_simulation.py ---
""")
            f.write(f"{ligand_stream_entry}\n")
    else:
        print(f"Ligand stream '{ligand_stream_file}' already found in {toppar_path}.")

def merge_structures(bilayer_psf, bilayer_crd, ligand_pdb, translation, out_psf, out_pdb):
    """Merges bilayer and ligand structures using ParmEd."""
    print(f"Loading bilayer topology: {bilayer_psf}")
    bilayer = pmd.load_file(bilayer_psf)
    print(f"Loading bilayer coordinates: {bilayer_crd}")
    bilayer_coords = pmd.load_file(bilayer_crd)
    bilayer.coordinates = bilayer_coords.coordinates
    # Optional: Assign box if the CRD contains it and PSF doesn't (usually PSF gets box later)
    if bilayer_coords.box is not None:
        bilayer.box = bilayer_coords.box

    print(f"Loading ligand: {ligand_pdb}")
    ligand = pmd.load_file(ligand_pdb)

    # --- Manually assign types to ligand atoms from its STR file ---
    print(f"Reading ligand STR file ({ligand_str_file}) for atom type definitions...")
    name_to_type_map = {}
    try:
        with open(ligand_str_file, 'r') as f_str:
            in_atom_section = False
            for line in f_str:
                # Basic parsing, handle comments and empty lines
                if '!' in line: line = line.split('!')[0]
                line = line.strip()
                if not line:
                    if in_atom_section: continue # Allow empty lines within section
                    else: pass # Ignore empty lines outside
                if line.startswith( ('ATOM', 'HETATM') ): # Check for ATOM or HETATM start
                    in_atom_section = True
                    parts = line.split()
                    if len(parts) >= 3:
                        atom_name = parts[1]
                        atom_type = parts[2]
                        name_to_type_map[atom_name] = atom_type
                # Stop if we hit end of atom section or another section keyword
                elif in_atom_section and line.startswith( ('BOND', 'IMPR', 'CMAP', 'END', 'PATCH', 'PRES') ):
                    in_atom_section = False
                    break
    except FileNotFoundError:
        print(f"Error: Ligand STR file {ligand_str_file} not found for typing.")
        # Decide whether to proceed or exit
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing ligand STR file {ligand_str_file}: {e}")
        sys.exit(1)

    print(f"Assigning {len(name_to_type_map)} types from STR file to ligand structure...")
    assigned_count = 0
    unassigned_names = set()
    corrected_count = 0
    for atom in ligand.atoms:
        if atom.name in name_to_type_map:
            correct_type = name_to_type_map[atom.name]
            original_type = atom.type # Store original type for comparison
            if atom.type != correct_type:
                # print(f"Correcting type for atom {atom.name}: {atom.type} -> {correct_type}")
                atom.type = correct_type # Set the type string directly
                corrected_count += 1
            # --- Add Debug Print Here ---
            print(f"Debug: Ligand Atom: Name='{atom.name}', OrigType='{original_type}', AssignedType='{atom.type}', TypeFromMap='{correct_type}'")
            # --------------------------
            assigned_count += 1
        else:
            unassigned_names.add(atom.name)

    if unassigned_names:
        print(f"Warning: Could not find type mapping for {len(unassigned_names)} ligand atom names: {sorted(list(unassigned_names))}. Keeping original types.")
    print(f"Assigned types for {assigned_count} ligand atoms. Corrected {corrected_count} types.")
    # ------------------------------------------------------------

    if translation and any(t != 0*angstroms for t in translation):
         print(f"Translating ligand by {translation}...")
         # ParmEd stores coordinates as a numpy array (N, 3)
         # Get translation vector in Angstroms
         trans_vec = translation.value_in_unit(angstroms)
         ligand.coordinates += trans_vec # Add the vector to all coordinates

    print("Combining structures...")
    merged = bilayer + ligand

    print(f"Saving merged PSF: {out_psf}")
    merged.save(out_psf, overwrite=True)
    print(f"Saving merged PDB: {out_pdb}")
    # Use PDB format for coordinates as it's simpler for OpenMM PDBFile class
    merged.save(out_pdb, overwrite=True, format='pdb')
    print("Merging complete.")
    return out_psf, out_pdb


def run_simulation():
    """Sets up and runs the OpenMM simulation."""
    # 1. Update Toppar
    update_toppar(toppar_file, ligand_str_file)

    # 2. Merge Structures
    merge_structures(bilayer_psf_file, bilayer_crd_file, ligand_pdb_file,
                     ligand_translation_A, merged_psf_file, merged_pdb_file)

    # 3. Load Inputs and Parameters for OpenMM
    print(f"Loading simulation inputs from: {input_recipe_file}")
    inputs = read_inputs(input_recipe_file)
    print(f"Loading parameters from: {toppar_file}")
    params = read_params(toppar_file) # Reads the updated toppar

    print(f"Loading merged PSF: {merged_psf_file}")
    psf = read_psf(merged_psf_file)
    print(f"Loading merged PDB coordinates: {merged_pdb_file}")
    pdb = PDBFile(merged_pdb_file) # Use OpenMM's PDBFile reader

    # Set Box Vectors - gen_box recalculates from PDB coordinates
    # This is safer after merging than relying on old sysinfo
    print("Generating box vectors from merged PDB coordinates...")
    psf = gen_box(psf, pdb)
    print(f"Box Vectors (Angstroms): {psf.boxVectors}")

    # 4. Build OpenMM System
    print("Building OpenMM System...")
    system = psf.createSystem(params, nonbondedMethod=inputs.coulomb,
                              nonbondedCutoff=inputs.r_off*nanometers,
                              constraints=inputs.cons,
                              ewaldErrorTolerance=inputs.ewald_Tol)

    print("Applying modifications (VFSwitch, Barostat, Restraints, HMR)...")
    if inputs.vdw == 'Force-switch': system = vfswitch(system, psf, inputs)
    if inputs.pcouple == 'yes':      system = barostat(system, inputs)
    # Restraints need careful checking after merging. Using original crd for ref might be wrong.
    # For simplicity, let's assume restraints are either not used or adapted separately.
    # If restraints are needed, the restraint generation logic might need modification
    # to use the new merged PDB as the reference structure.
    if inputs.rest == 'yes':
        print("WARNING: Restraints enabled. Ensure restraint files (`restraints/*.txt`) are compatible with the *merged* system topology and indices.")
        # Original restraint code uses the initial CRD. We need to use the merged PDB.
        # We pass the PDB object's positions instead of the original crd object
        system = restraints(system, pdb, inputs) # Pass PDB object
    if apply_hmr:
        print("Applying Hydrogen Mass Repartitioning...")
        system = HydrogenMassRepartition(system, psf)

    # 5. Setup Integrator and Simulation
    print(f"Setting up Langevin Integrator: Temp={inputs.temp}K, Friction={inputs.fric_coeff}/ps, Step={inputs.dt}ps")
    integrator = LangevinIntegrator(inputs.temp*kelvin, inputs.fric_coeff/picosecond, inputs.dt*picoseconds)

    print(f"Setting up Platform: {platform_name}")
    try:
        platform = Platform.getPlatformByName(platform_name)
        prop = platform_properties if platform_properties else {}
        print(f"Successfully obtained platform: {platform.getName()}")
    except Exception as e:
        print(f"Error setting up platform {platform_name}: {e}")
        print("Available platforms:", [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])
        sys.exit(1)

    print("Creating Simulation object...")
    simulation = Simulation(psf.topology, system, integrator, platform, prop)
    simulation.context.setPositions(pdb.positions) # Set positions from merged PDB

    # Handle initial state (Checkpoints/Restarts) - Optional, typically for continuations
    # For a fresh merged run, we usually start from minimized/equilibrated coords
    # if args.icrst: ... (read charmm rst)
    # if args.irst: ... (read xml restart)
    # if args.ichk: ... (read checkpoint)

    # 6. Rewrap (Optional, before minimization/dynamics)
    if rewrap_coords:
        print("Applying initial re-wrapping...")
        simulation = rewrap(simulation) # Ensure rewrap works correctly with Simulation object

    # 7. Initial Energy and Minimization
    print("\nCalculating initial system energy...")
    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Initial Potential Energy: {initial_energy}")

    if inputs.mini_nstep > 0:
        print(f"\nEnergy Minimization: {inputs.mini_nstep} steps, Tol={inputs.mini_Tol} kJ/mol/nm")
        simulation.minimizeEnergy(tolerance=inputs.mini_Tol*kilojoule/mole/nanometer, maxIterations=inputs.mini_nstep)
        min_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"Energy after minimization: {min_energy}")

    # 8. Generate Initial Velocities
    if inputs.gen_vel == 'yes':
        print("\nGenerating velocities at {inputs.gen_temp} K (Seed: {inputs.gen_seed})")
        if inputs.gen_seed:
            simulation.context.setVelocitiesToTemperature(inputs.gen_temp*kelvin, inputs.gen_seed)
        else:
            simulation.context.setVelocitiesToTemperature(inputs.gen_temp*kelvin)

    # 9. Production Run
    if inputs.nstep > 0:
        print("\nStarting MD run: {inputs.nstep} steps")

        # Setup Reporters
        output_dcd = f"{output_prefix}.dcd"
        output_rst = f"{output_prefix}.rst"
        output_chk = f"{output_prefix}.chk"
        output_pdb_final = f"{output_prefix}_final.pdb"

        if inputs.nstdcd > 0:
            print(f"Will write trajectory to {output_dcd} every {inputs.nstdcd} steps.")
            simulation.reporters.append(DCDReporter(output_dcd, inputs.nstdcd))

        print(f"Will write state data to stdout every {inputs.nstout} steps.")
        simulation.reporters.append(
            StateDataReporter(sys.stdout, inputs.nstout, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, progress=True,
                              remainingTime=True, speed=True, totalSteps=inputs.nstep, separator='	')
        )

        # Checkpoint reporter
        if inputs.nstout > 0: # Use nstout for checkpoint frequency too? Or add new input?
             print(f"Will write checkpoint file to {output_chk} every {inputs.nstout} steps.")
             simulation.reporters.append(CheckpointReporter(output_chk, inputs.nstout))


        # Simulated Annealing (if specified in input)
        if inputs.annealing == 'yes':
            print(f"Running simulated annealing from {inputs.temp_init} K with interval {inputs.interval} K/step.")
            interval = inputs.interval
            current_temp = inputs.temp_init
            for i in range(inputs.nstep):
                integrator.setTemperature(current_temp*kelvin)
                simulation.step(1)
                current_temp += interval
        else:
            # Normal MD run
            simulation.step(inputs.nstep)

        print("\nMD run finished.")

        # 10. Write Final Files
        print(f"Saving final state to {output_rst}")
        state = simulation.context.getState( getPositions=True, getVelocities=True )
        with open(output_rst, 'w') as f:
            f.write(XmlSerializer.serialize(state))

        # Checkpoint is handled by reporter now, but save final explicitly if needed
        # print(f"Saving final checkpoint to {output_chk}")
        # with open(output_chk, 'wb') as f:
        #    f.write(simulation.context.createCheckpoint())

        print(f"Saving final PDB structure to {output_pdb_final}")
        final_positions = simulation.context.getState(getPositions=True).getPositions()
        # Apply rewrap before saving final PDB if requested
        if rewrap_coords:
            print("Applying final re-wrapping before saving PDB...")
            # Need to update context positions *before* getting them again if rewrap modifies inplace
            simulation = rewrap(simulation) # Re-assign simulation potentially
            final_positions = simulation.context.getState(getPositions=True).getPositions()

        PDBFile.writeFile(psf.topology, final_positions, open(output_pdb_final, 'w'))

    else:
        print("No production steps (nstep=0) requested in input file.")

    print("\nSimulation script completed.")

if __name__ == "__main__":
    # Basic check for required files
    required_files = [bilayer_psf_file, bilayer_crd_file, ligand_pdb_file, ligand_str_file, toppar_file, input_recipe_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: The following required files are missing:")
        for f in missing_files:
            print(f"- {f}")
        sys.exit(1)

    # Check for ParmEd
    try:
        import parmed
    except ImportError:
        print("Error: ParmEd library is required but not found.")
        print("Please install it, e.g., using 'pip install parmed'")
        sys.exit(1)

    run_simulation() 