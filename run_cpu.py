# run_cpu.py
import os, torch, numpy as np
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.io import write
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ——仅用 CPU——
device = "cpu"
# 可选：限制 CPU 线程数以更稳定
n = max(1, min(8, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(n))
os.environ.setdefault("MKL_NUM_THREADS", str(n))
torch.set_num_threads(n)

# UMA 小模型 + OC20 任务
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
calc = FAIRChemCalculator(predictor, task_name="oc20")

# 构建 Cu(100) 3×3×3 表面（~8 Å 真空）
slab = fcc100("Cu", (3, 3, 3), vacuum=8.0, periodic=True)

# 固定最底层原子，避免整体漂移
z = slab.get_positions()[:, 2]
mask = np.isclose(z, z.min())
slab.set_constraint(FixAtoms(mask=mask))

# 加 CO，bridge 位，高度 2.0 Å
ads = molecule("CO")
add_adsorbate(slab, ads, height=2.0, position="bridge")

slab.calc = calc

# 优化
opt = LBFGS(slab, logfile="opt.log")
opt.run(fmax=0.05, steps=100)

print("Final energy (eV):", slab.get_potential_energy())
write("final.cif", slab)  # 保存结果结构
