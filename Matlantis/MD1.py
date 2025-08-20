import os
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
estimator = Estimator(model_version="v2.0.0",calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3)
calculator = ASECalculator(estimator)

from ase.io import read,write
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from ase.md.npt import NPT
from ase.md import MDLogger
from ase import units
from time import perf_counter
from ase.md.nvtberendsen import NVTBerendsen
from ase import Atoms
#from ase.md.nvt import NVT


# Set up a crystal
atoms = read("surface-2.cif")
atoms.pbc = True
atoms.calc = calculator
print("atoms = ", atoms)

# Input parameters
time_step = 1.0      # fs
temperature = 300    # K
warmup_steps = 500
prod_steps = 500
num_interval = 100
save_interval = 1000
tau_berendsen = 10.0  # fs
tau_nose = 100.0      # fs
sigma   = 1.0     # External pressure in bar
ttime   = 100.0    # Time constant in fs
#pfactor = 2e6     # Barostat parameter in GPa
step_offset = 0
z_threshold = 9.0

output_filename = "surface_NVT_300K"
log_filename = output_filename + ".log"
traj_filename = output_filename + ".traj"
print("output_filename = ", output_filename)
structure_dir   = "structure"
com_output_file = os.path.join(structure_dir, "com.data")

os.makedirs(structure_dir, exist_ok=True)
print("output_filename = ", output_filename)

# Initialize momenta
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
Stationary(atoms)

# Define print function
def print_dyn(dyn, atoms):
    imd = dyn.get_number_of_steps()
    etot = atoms.get_total_energy()
    temp_K = atoms.get_temperature()
    stress = atoms.get_stress(include_ideal_gas=True)/units.GPa
    stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
    elapsed_time = perf_counter() - start_time
    print(f"  {imd: >6}   {etot:.3f}    {temp_K:.2f}    {stress_ave:.2f}  {stress[0]:.2f}  {stress[1]:.2f}  {stress[2]:.2f}  {stress[3]:.2f}  {stress[4]:.2f}  {stress[5]:.2f}    {elapsed_time:.3f}")

#def write_poscar(atoms=atoms):
    #step = dyn1.get_number_of_steps() if dyn1 is not None else dyn2.get_number_of_steps()
    #filename = f"POSCAR_{step:06d}"
    #write(filename, atoms, format='vasp')

def write_poscar():
    step = dyn1.get_number_of_steps() if dyn1 and dyn1.get_number_of_steps() < warmup_steps else dyn2.get_number_of_steps()
    total_step = step + step_offset
    filename = os.path.join(structure_dir, f"POSCAR_step{total_step:06d}")
    write(filename, atoms, format='vasp')    

def get_adsorbate_com(atoms, z_thr):
    pos = atoms.get_positions()
    mask = pos[:, 2] > z_thr
    if not mask.any():
        raise ValueError(f"No atoms found with z > {z_thr} Å")
    ads = atoms[mask]
    return ads.get_center_of_mass()

def print_com(atoms=atoms):
    com_ads = get_adsorbate_com(atoms, z_threshold)
    print(f"Adsorbate COM: {com_ads}")
    with open(com_output_file, "a") as f:
        f.write(f"{com_ads[0]:.6f} {com_ads[1]:.6f} {com_ads[2]:.6f}\n")   

# --- Warm-up: Berendsen ---
dyn1 = NVTBerendsen(atoms,
                    timestep=time_step * units.fs,
                    temperature_K=temperature,
                    taut=tau_berendsen * units.fs,
                    trajectory=traj_filename,
                    loginterval=num_interval)

dyn1.attach(lambda: print_dyn(dyn1, atoms), interval=num_interval)
dyn1.attach(MDLogger(dyn1, atoms, log_filename, header=True, stress=True, peratom=True, mode="w"), interval=num_interval)
dyn1.attach(print_com, interval=500)
dyn1.attach(write_poscar, interval=save_interval)

print("\n--- Warmup phase (Berendsen) ---")
start_time = perf_counter()
print(f"    imd     Etot(eV)    T(K)    stress(mean,xx,yy,zz,yz,xz,xy)(GPa)  elapsed_time(sec)")
dyn1.run(warmup_steps)
step_offset = warmup_steps 

# --- Main run: Nosé–Hoover ---

dyn2 = NPT(atoms,
          timestep= time_step*units.fs,
          temperature_K = temperature,
          externalstress = 0.1e-6 * units.GPa,  # Ignored in NVT
          ttime = ttime*units.fs,
          pfactor = None,   # None for NVT
          loginterval=num_interval,
          trajectory=traj_filename
          )

dyn2.attach(lambda: print_dyn(dyn2, atoms), interval=num_interval)
dyn2.attach(MDLogger(dyn2, atoms, log_filename, header=False, stress=True, peratom=True, mode="a"), interval=num_interval)
dyn2.attach(write_poscar, interval=save_interval)
dyn2.attach(print_com, interval=500)

print("\n--- Production phase (Nose–Hoover) ---")
start_time = perf_counter()
print(f"    imd     Etot(eV)    T(K)    stress(mean,xx,yy,zz,yz,xz,xy)(GPa)  elapsed_time(sec)")
dyn2.run(prod_steps)

write(os.path.join(structure_dir, "final.cif"), atoms)