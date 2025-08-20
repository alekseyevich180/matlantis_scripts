import os
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
estimator = Estimator(model_version="v2.0.0",calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3)
calculator = ASECalculator(estimator)


from ase import units
#from ase.md.nvt import NVT
from ase.io import read, write
from ase.build import add_adsorbate
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md import MDLogger
from ase.units import fs, GPa
import numpy as np
import os
from time import perf_counter

# 结构设置
surface_file = "surface.cif"
mol_files = ["2-ketone.cif", "3-ketone.cif", "4-ketone.cif", "5-ketone.cif","2-6ketone.cif","3-6ketone.cif"]  # 替换为你实际的文件名

# Input parameters
time_step = 1.0      # fs
temperature = 300    # K
warmup_steps = 10000
prod_steps = 50000
num_interval = 10
save_interval = 10
tau_berendsen = 10.0  # fs
tau_nose = 100.0      # fs
sigma   = 1.0     # External pressure in bar
ttime   = 100.0    # Time constant in fs
#pfactor = 2e6     # Barostat parameter in GPa
step_offset = 0
z_threshold = 9.0
height_init = 2.0


from ase.constraints import FixAtoms

# 固定 slab 中 z 坐标小于阈值的原子
z_fixed_threshold = 9.0  # 🛠️ 你可以根据实际结构调整这个数值
slab_positions = slab.get_positions()
fixed_indices = [i for i, pos in enumerate(slab_positions) if pos[2] < z_fixed_threshold]

# 设置约束
slab.set_constraint(FixAtoms(indices=fixed_indices))

# 函数定义
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

def get_adsorbate_com(atoms, z_thr):
    pos = atoms.get_positions()
    mask = pos[:, 2] > z_thr
    if not mask.any():
        raise ValueError(f"No atoms found with z > {z_thr} Å")
    return atoms[mask].get_center_of_mass()

# 🔁 多分子循环
for mol_file in mol_files:
    print(f"\n🔬 Starting MD for {mol_file}...\n")

    # 创建输出目录
    base_name = os.path.splitext(os.path.basename(mol_file))[0]
    output_dir = f"MD_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 读取结构
    slab = read(surface_file)
    mol = read(mol_file)
    slab.calc = calculator
    mol.calc = calculator

    # 分子旋转（可选）
    mol.euler_rotate(phi=30, theta=45, psi=60)

    # 吸附
    xy_pos = np.mean(slab.get_positions()[:, :2], axis=0)
    add_adsorbate(slab, mol, height=height_init, position=xy_pos)
    atoms = slab
    atoms.pbc = True
    atoms.calc = calculator

    # 初始化动量
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

    def write_poscar():
        step = dyn1.get_number_of_steps() if dyn1.get_number_of_steps() < warmup_steps else dyn2.get_number_of_steps()
        fname = os.path.join(structure_dir, f"POSCAR_step{step:06d}.vasp")
        write(fname, atoms, format="vasp")

    # --- Warmup ---
    traj_file = os.path.join(output_dir, "warmup.traj")
    log_file = os.path.join(output_dir, "md.log")

    dyn1 = NVTBerendsen(atoms, timestep=time_step * fs,
                        temperature_K=temperature,
                        taut=tau_berendsen,
                        trajectory=traj_file,
                        loginterval=num_interval)

    start_time = perf_counter()
    dyn1.attach(lambda: print_dyn(dyn1, atoms, start_time), interval=num_interval)
    dyn1.attach(MDLogger(dyn1, atoms, log_file, header=True, stress=True, peratom=True, mode="w"), interval=num_interval)
    dyn1.attach(print_com, interval=500)
    dyn1.attach(write_poscar, interval=save_interval)

    print("--- Warmup phase ---")
    dyn1.run(warmup_steps)

    # --- Main MD ---
    traj_file2 = os.path.join(output_dir, "prod.traj")

    dyn2 = NPT(atoms, timestep=time_step * fs,
               temperature_K=temperature,
               externalstress=0.1e-6 * GPa,
               ttime=ttime,
               pfactor=None,
               loginterval=num_interval,
               trajectory=traj_file2)

    start_time = perf_counter()
    dyn2.attach(lambda: print_dyn(dyn2, atoms, start_time), interval=num_interval)
    dyn2.attach(MDLogger(dyn2, atoms, log_file, header=False, stress=True, peratom=True, mode="a"), interval=num_interval)
    dyn2.attach(print_com, interval=500)
    dyn2.attach(write_poscar, interval=save_interval)

    print("--- Production phase ---")
    dyn2.run(prod_steps)

    # Final output
    write(os.path.join(output_dir, "final.cif"), atoms)
    print(f"✅ Finished MD for {mol_file}\n")