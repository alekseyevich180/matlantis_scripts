
import os
import numpy as np
from time import perf_counter
from ase import units
from ase.io import read, write
from ase.build import add_adsorbate
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs, GPa
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator

# Setup estimator and calculator
estimator = Estimator(model_version="v2.0.0", calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3)
calculator = ASECalculator(estimator)

# Parameters
mol_files = [
    "output/2-ketone/best_trial.cif",
    "output/3-ketone/best_trial.cif",
    "output/4-ketone/best_trial.cif",
    "output/5-ketone/best_trial.cif",
    "output/2-6ketone/best_trial.cif",
    "output/3-6ketone/best_trial.cif"
]

time_step = 1.0
temperature = 300
warmup_steps = 10000
prod_steps = 50000
num_interval = 10
save_interval = 10
tau_berendsen = 10.0
ttime = 100.0
z_threshold = 9.0

def get_adsorbate_com(atoms, z_thr):
    pos = atoms.get_positions()
    mask = pos[:, 2] > z_thr
    if not mask.any():
        raise ValueError(f"No atoms found with z > {z_thr} Å")
    return atoms[mask].get_center_of_mass()

def run_md_for_molecule(mol_file):
    print(f"\n🔬 Starting MD for {mol_file}...\n")
    base_name = os.path.basename(os.path.dirname(mol_file))
    output_dir = f"MD_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    atoms = read(mol_file)
    atoms.calc = calculator
    atoms.pbc = True

    fixed_indices = [i for i, pos in enumerate(atoms.get_positions()) if pos[2] < z_threshold]
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
    print(f"✅ Finished MD for {mol_file}\n")

# Run for each optimized best_trial.cif
for mol_file in mol_files:
    run_md_for_molecule(mol_file)
