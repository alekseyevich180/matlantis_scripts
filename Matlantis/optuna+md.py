import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import perf_counter

from ase import Atoms, units
from ase.io import read, write
from ase.io.jsonio import write_json, read_json
from ase.optimize import LBFGS
from ase.build import add_adsorbate
from ase.constraints import FixAtoms
from ase.visualize import view
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md import MDLogger
from ase.units import fs, GPa

import optuna
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

# Setup calculator
estimator = Estimator(model_version="v3.0.0", calc_mode=EstimatorCalcMode.CRYSTAL_U0)
calculator = ASECalculator(estimator)

surface_file = "surface.cif"
mol_files = ["2-ketone.cif", "3-ketone.cif", "4-ketone.cif", "5-ketone.cif", "2-6ketone.cif", "3-6ketone.cif"]

time_step = 1.0
temperature = 300
warmup_steps = 10000
prod_steps = 50000
num_interval = 10
save_interval = 10
tau_berendsen = 10.0
ttime = 100.0
z_threshold = 9.0
height_init = 2.0

def atoms_to_json(atoms):
    f = io.StringIO()
    write(f, atoms, format="json")
    return f.getvalue()

def json_to_atoms(atoms_str):
    return read(io.StringIO(atoms_str), format="json")

def get_opt_energy(atoms, fmax=0.001):
    atoms.set_calculator(calculator)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax)
    return atoms.get_total_energy()

def get_adsorbate_com(atoms, z_thr):
    pos = atoms.get_positions()
    mask = pos[:, 2] > z_thr
    if not mask.any():
        raise ValueError(f"No atoms found with z > {z_thr} Å")
    return atoms[mask].get_center_of_mass()

def run_md(atoms, output_dir, base_name):
    atoms.pbc = True
    atoms.calc = calculator

    slab_positions = atoms.get_positions()
    fixed_indices = [i for i, pos in enumerate(slab_positions) if pos[2] < z_threshold]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)

    com_output_file = os.path.join(output_dir, "com.data")
    structure_dir = os.path.join(output_dir, "structure")
    os.makedirs(structure_dir, exist_ok=True)

    def print_com():
        com = get_adsorbate_com(atoms, z_threshold)
        print(f"Adsorbate COM: {com}")
        with open(com_output_file, "a") as f:
            f.write(f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}\n")

    def write_poscar(step):
        fname = os.path.join(structure_dir, f"POSCAR_step{step:06d}.vasp")
        write(fname, atoms, format="vasp")

    def print_dyn(dyn, atoms, start_time):
        imd = dyn.get_number_of_steps()
        etot = atoms.get_total_energy()
        temp_K = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True) / GPa
        stress_ave = np.mean(stress[:3])
        elapsed_time = perf_counter() - start_time
        print(f"  {imd: >6}   {etot:.3f}    {temp_K:.2f}    {stress_ave:.2f}  "
              f"{stress[0]:.2f}  {stress[1]:.2f}  {stress[2]:.2f}  "
              f"{stress[3]:.2f}  {stress[4]:.2f}  {stress[5]:.2f}    {elapsed_time:.3f}")

    traj_file1 = os.path.join(output_dir, "warmup.traj")
    log_file = os.path.join(output_dir, "md.log")

    dyn1 = NVTBerendsen(atoms, timestep=time_step * fs,
                        temperature_K=temperature, taut=tau_berendsen,
                        trajectory=traj_file1, loginterval=num_interval)
    start_time = perf_counter()
    dyn1.attach(lambda: print_dyn(dyn1, atoms, start_time), interval=num_interval)
    dyn1.attach(MDLogger(dyn1, atoms, log_file, header=True, stress=True, peratom=True, mode="w"), interval=num_interval)
    dyn1.attach(print_com, interval=500)
    dyn1.attach(lambda: write_poscar(dyn1.get_number_of_steps()), interval=save_interval)

    print("--- Warmup phase ---")
    dyn1.run(warmup_steps)

    traj_file2 = os.path.join(output_dir, "prod.traj")
    dyn2 = NPT(atoms, timestep=time_step * fs, temperature_K=temperature,
               externalstress=0.1e-6 * GPa, ttime=ttime, pfactor=None,
               loginterval=num_interval, trajectory=traj_file2)

    start_time = perf_counter()
    dyn2.attach(lambda: print_dyn(dyn2, atoms, start_time), interval=num_interval)
    dyn2.attach(MDLogger(dyn2, atoms, log_file, header=False, stress=True, peratom=True, mode="a"), interval=num_interval)
    dyn2.attach(print_com, interval=500)
    dyn2.attach(lambda: write_poscar(dyn2.get_number_of_steps() + warmup_steps), interval=save_interval)

    print("--- Production phase ---")
    dyn2.run(prod_steps)

    write(os.path.join(output_dir, "final.cif"), atoms)
    print(f"✅ Finished MD for {base_name}\n")

# Optuna search
def already_slab():
    slab = read(surface_file)
    slab.calc = calculator
    E_slab = get_opt_energy(slab)
    return slab, E_slab

def already_mol(filename):
    mol = read(filename)
    mol.calc = calculator
    E_mol = get_opt_energy(mol)
    return mol, E_mol

for mol_file in mol_files:
    mol, E_mol = already_mol(mol_file)
    slab, E_slab = already_slab()

    def objective(trial):
        slab_c = json_to_atoms(atoms_to_json(slab))
        mol_c = json_to_atoms(atoms_to_json(mol))

        phi = 180 * trial.suggest_float("phi", -1, 1)
        theta = np.arccos(trial.suggest_float("theta", -1, 1)) * 180. / np.pi
        psi = 180 * trial.suggest_float("psi", -1, 1)
        x_pos = trial.suggest_float("x_pos", 0, 0.5)
        y_pos = trial.suggest_float("y_pos", 0, 0.5)
        z_hig = trial.suggest_float("z_hig", 1, 5)

        mol_c.euler_rotate(phi=phi, theta=theta, psi=psi)
        xy_position = np.matmul([x_pos, y_pos, 0], slab_c.cell)[:2]
        add_adsorbate(slab_c, mol_c, height=z_hig, position=xy_position)

        E_total = get_opt_energy(slab_c, fmax=1e-3)
        trial.set_user_attr("structure", atoms_to_json(slab_c))
        return E_total - E_slab - E_mol

    study = optuna.create_study()
    study.set_user_attr("slab", atoms_to_json(slab))
    study.set_user_attr("E_slab", E_slab)
    study.set_user_attr("mol", atoms_to_json(mol))
    study.set_user_attr("E_mol", E_mol)
    study.optimize(objective, n_trials=50)

    best_structure = json_to_atoms(study.best_trial.user_attrs["structure"])
    base_name = os.path.splitext(os.path.basename(mol_file))[0]
    output_dir = f"optuna_md_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    run_md(best_structure, output_dir, base_name)
