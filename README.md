# OpenMM MD Simulation of a Ligand-Membrane System

## Overview

This project performs molecular dynamics (MD) simulations of small molecules interacting with a lipid bilayer using OpenMM. It automates:

* Integration of ligand parameters from CGenFF
* Merging ligand into a CHARMM-GUI prepared membrane system
* MD simulation using OpenMM
* Post-analysis for ligand penetration assessment

---

## Requirements

* OpenMM
* ParmEd
* MDTraj
* Matplotlib
* RDKit (optional)
* Open Babel (optional)
* CHARMM-GUI files: `step5_charmm2omm.psf`, `step5_charmm2omm.crd`, and `step6.1_equilibration.inp`
* Ligand `.mol2` and `.str` from CGenFF

---

## File Structure

```
project/
├── run_merged_simulation.py           # Main merger & MD driver
├── analyze_ligand_position.py         # Ligand position & distance analysis
├── Cholesterol.mol2 / .pdb / .str     # Ligand input files
├── toppar.str                         # CHARMM stream file
├── step5_charmm2omm.psf/.crd          # Membrane topology and coordinates
├── step6.1_equilibration.inp          # CHARMM-GUI simulation settings
├── omm_*.py                           # CHARMM-GUI helper scripts
```

---

## Preparing Ligand Parameters

1. Visit [CGenFF Server](https://app.cgenff.com/login) to upload your ligand and get:

   * `ligand.mol2`
   * `ligand.str`

2. Convert `.mol2` to `.pdb`:

```bash
obabel -imol2 Cholesterol.mol2 -opdb -O Cholesterol.pdb
```

---

## Configuring the Simulation

Open `run_merged_simulation.py` and update:

```python
bilayer_psf_file    = "step5_charmm2omm.psf"
bilayer_crd_file    = "step5_charmm2omm.crd"
ligand_pdb_file     = "Cholesterol.pdb"
ligand_str_file     = "Cholesterol.str"
toppar_file         = "toppar.str"
input_recipe_file   = "step6.1_equilibration.inp"
ligand_translation_A = [10.0, 0.0, 55.0] * angstroms
```

---

## Running the MD Workflow

```bash
python run_merged_simulation.py
```

Generates:

* `merged.psf` / `merged.pdb`: Merged system
* `merged_equil.dcd`: Trajectory
* `merged_equil.rst`: Restart file
* `merged_equil_final.pdb`: Final structure

---

## Analyzing Ligand Penetration

```bash
python analyze_ligand_position.py
```

### What it does:

* Computes Z-distance of ligand COM from membrane COM (defined by lipid P atoms)
* Separates upper/lower membrane leaflets
* Calculates minimum ligand–lipid distances per frame
* Outputs:

  * `ligand_membrane_z_distance_boundaries.png`
  * `ligand_lipid_min_distance.png`

---

## Plot Interpretation

* **Z-distance plot** shows how the ligand traverses the membrane (crossing center or residing in leaflets).
* **Min-distance plot** reveals how close the ligand gets to membrane atoms across time.

---

## Troubleshooting

* Atom selection errors? Double-check `resname` (e.g., `UNK`, `POPC`) and element labels (e.g., `P`).
* Long load times? Use trajectory slicing or smaller timeframes.
* Memory errors? Compute distances on fewer atom pairs or use frame batching.

---

## License

MIT License

---

## Author

Arvin Haj-Mirzaian
