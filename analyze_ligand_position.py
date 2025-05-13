#!/usr/bin/env python
"""
Analyzes the Z-distance between a ligand and the membrane center over time,
visualizes membrane boundaries, and calculates the minimum distance between
the ligand and the lipids.

Reads a simulation trajectory (DCD) and topology (PSF), calculates relevant
metrics, and generates plots.
"""

import sys
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import os # For checking file existence

# --- Configuration ---
topology_file = "merged.psf"
trajectory_file = "merged_equil.dcd"

ligand_resname = "UNK"  # Residue name assigned to the ligand
lipid_resname = "POPC" # Main lipid residue name
membrane_center_atom_element = "P" # Element used to define membrane center/leaflets

output_plot_z_distance = "ligand_membrane_z_distance_boundaries.png"
output_plot_min_distance = "ligand_lipid_min_distance.png"
# -------------------


if __name__ == "__main__":
    print(f"Analyzing trajectory: {trajectory_file} with topology: {topology_file}")

    # --- Dependency Checks ---
    try:
        import mdtraj
    except ImportError:
        print("Error: MDTraj library is required but not found.")
        print("Please install it, e.g., using 'pip install mdtraj matplotlib'")
        sys.exit(1)
    try:
        import matplotlib
    except ImportError:
        print("Error: Matplotlib library is required but not found.")
        print("Please install it, e.g., using 'pip install mdtraj matplotlib'")
        sys.exit(1)

    # --- File Existence Checks ---
    if not os.path.exists(topology_file):
        print(f"Error: Topology file not found: {topology_file}")
        sys.exit(1)
    if not os.path.exists(trajectory_file):
        print(f"Error: Trajectory file not found: {trajectory_file}")
        sys.exit(1)

    # --- Load Trajectory ---
    try:
        print("Loading trajectory... (This might take a while for large files)")
        # Load only coordinates to potentially save memory if state info isn't needed
        traj = md.load(trajectory_file, top=topology_file)
        print(f"Trajectory loaded: {traj.n_frames} frames, {traj.n_atoms} atoms.")
    except Exception as e:
        print(f"Error loading trajectory or topology files: {e}")
        sys.exit(1)

    # --- Atom Selections ---
    try:
        ligand_atom_indices = traj.topology.select(f"resname '{ligand_resname}'")
        # Select all atoms of the lipid for minimum distance calculation
        lipid_atom_indices = traj.topology.select(f"resname '{lipid_resname}'")
        # Select phosphorus atoms for membrane center/leaflet calculations
        p_atom_indices = traj.topology.select(f"resname '{lipid_resname}' and element {membrane_center_atom_element}")
    except Exception as e:
        print(f"Error selecting atoms: {e}")
        print("Check if the residue names ('{ligand_resname}', '{lipid_resname}') and element symbol ('{membrane_center_atom_element}') are correct.")
        sys.exit(1)

    # --- Selection Validation ---
    if ligand_atom_indices.size == 0:
        print(f"Error: No atoms found for ligand residue name '{ligand_resname}'.")
        sys.exit(1)
    else:
        print(f"Found {ligand_atom_indices.size} atoms for ligand '{ligand_resname}'.")

    if lipid_atom_indices.size == 0:
        print(f"Error: No atoms found for lipid residue name '{lipid_resname}'.")
        sys.exit(1)
    # No need to print lipid atom count, usually large

    if p_atom_indices.size == 0:
        print(f"Error: No atoms found for membrane center definition (Resname '{lipid_resname}', Element '{membrane_center_atom_element}').")
        sys.exit(1)
    else:
        print(f"Found {p_atom_indices.size} atoms for membrane center ('{lipid_resname}' {membrane_center_atom_element} atoms).")

    # --- Analysis 1: Z-Distance relative to Membrane Center & Leaflets ---
    print("Calculating centers of mass and Z-distance...")

    # Ligand COM
    ligand_com = md.compute_center_of_mass(traj.atom_slice(ligand_atom_indices))
    ligand_z = ligand_com[:, 2] # Z-coordinate of ligand COM

    # Membrane COM (using P atoms)
    membrane_p_com = md.compute_center_of_mass(traj.atom_slice(p_atom_indices))
    membrane_center_z = membrane_p_com[:, 2] # Z-coordinate of membrane center

    # Calculate relative Z-distance
    relative_z_distance = ligand_z - membrane_center_z

    # Identify upper and lower leaflets based on first frame P positions
    first_frame_p_z = traj.xyz[0, p_atom_indices, 2]
    z_midpoint = np.mean(first_frame_p_z)
    upper_leaflet_indices = p_atom_indices[first_frame_p_z > z_midpoint]
    lower_leaflet_indices = p_atom_indices[first_frame_p_z <= z_midpoint]

    if upper_leaflet_indices.size == 0 or lower_leaflet_indices.size == 0:
        print("Warning: Could not reliably separate upper and lower leaflets based on P atoms.")
        avg_upper_z = np.full_like(membrane_center_z, np.nan)
        avg_lower_z = np.full_like(membrane_center_z, np.nan)
    else:
        print("Calculating average Z of upper and lower leaflets...")
        # Calculate average Z for each leaflet across trajectory
        avg_upper_z = np.mean(traj.xyz[:, upper_leaflet_indices, 2], axis=1)
        avg_lower_z = np.mean(traj.xyz[:, lower_leaflet_indices, 2], axis=1)

        # Make leaflet positions relative to the membrane center for plotting
        avg_upper_z_relative = avg_upper_z - membrane_center_z
        avg_lower_z_relative = avg_lower_z - membrane_center_z

    # Time axis
    time_ns = traj.time / 1000.0

    # --- Analysis 2: Minimum Distance between Ligand and Lipids ---
    print("Calculating minimum distance between ligand and lipids...")
    # Define the pairs of indices between which to compute distances
    # (all ligand atoms vs all lipid atoms)
    atom_pairs = np.array([(l, p) for l in ligand_atom_indices for p in lipid_atom_indices])
    if atom_pairs.shape[0] == 0:
        print("Warning: No ligand-lipid atom pairs found. Cannot compute minimum distance.")
        min_distance_per_frame = np.full(traj.n_frames, np.nan)
    else:
        # Compute the distances for these pairs for all frames
        # This can be memory intensive if atom_pairs is huge
        distances = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)
        # Find the minimum distance across all pairs for each frame
        min_distance_per_frame = np.min(distances, axis=1)

    # --- Plotting ---

    # Plot 1: Z-Distance with Membrane Boundaries
    print(f"Generating Z-distance plot: {output_plot_z_distance}")
    plt.figure(figsize=(12, 7))
    plt.plot(time_ns, relative_z_distance, label=f"Ligand '{ligand_resname}' COM Z - Membrane COM Z", color='blue')

    # Plot leaflet boundaries if calculated
    if not np.isnan(avg_upper_z_relative).all():
        plt.plot(time_ns, avg_upper_z_relative, label="Upper Leaflet P atoms (avg Z)", color='red', linestyle='--')
        plt.plot(time_ns, avg_lower_z_relative, label="Lower Leaflet P atoms (avg Z)", color='green', linestyle='--')

    plt.xlabel("Time (ns)")
    plt.ylabel(f"Relative Z-Distance (nm)")
    plt.title(f"Ligand ('{ligand_resname}') Z-Position Relative to Membrane ('{lipid_resname}') Center")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    try:
        plt.savefig(output_plot_z_distance, dpi=300)
        print(f"Plot saved to: {output_plot_z_distance}")
    except Exception as e:
        print(f"Error saving Z-distance plot: {e}")
    plt.close() # Close figure to free memory

    # Plot 2: Minimum Distance
    print(f"Generating minimum distance plot: {output_plot_min_distance}")
    plt.figure(figsize=(12, 7))
    plt.plot(time_ns, min_distance_per_frame, label=f"Min dist Ligand('{ligand_resname}')-Lipid('{lipid_resname}')", color='purple')
    plt.xlabel("Time (ns)")
    plt.ylabel("Minimum Distance (nm)")
    plt.title(f"Minimum Distance between Ligand ('{ligand_resname}') and Lipids ('{lipid_resname}')")
    plt.ylim(bottom=0) # Distance cannot be negative
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    try:
        plt.savefig(output_plot_min_distance, dpi=300)
        print(f"Plot saved to: {output_plot_min_distance}")
    except Exception as e:
        print(f"Error saving minimum distance plot: {e}")
    plt.close()

    print("Analysis complete.") 