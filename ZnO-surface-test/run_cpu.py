import io
import os
import pandas as pd
import tempfile
import os, torch, numpy as np
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.io import write, read
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import Atoms
from ase.build import bulk, fcc111, molecule, add_adsorbate
from ase.constraints import ExpCellFilter, StrainFilter
from ase.io.jsonio import write_json, read_json
from ase.optimize import LBFGS, FIRE
#from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#import optuna
from ase.visualize import view

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = "cpu"
# 可选：限制 CPU 线程数以更稳定
n = max(1, min(8, os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(n))
os.environ.setdefault("MKL_NUM_THREADS", str(n))
torch.set_num_threads(n)

# UMA 小模型 + OC20 任务
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
calc = FAIRChemCalculator(predictor, task_name="oc20")

# ---------- read slab ----------
slab = read("surface.cif", index=0)  
if slab is None:
    raise RuntimeError("surface.cif 读取失败：返回 None")

if slab.pbc is None or tuple(slab.pbc) == (False, False, False):
    slab.pbc = (True, True, False)

# ---------- fix bottom layer ----------
z = slab.get_positions()[:, 2]
zmin, zmax = z.min(), z.max()

bottom_mask = z < (zmin + 0.3)
slab.set_constraint(FixAtoms(mask=bottom_mask))


from itertools import combinations

top_idx = np.where(z > (zmax - 0.3))[0]
if len(top_idx) < 2:
    top_idx = np.argsort(z)[-2:]  

# 找顶层里距离最近的一对（使用最小镜像）
best_pair = None
best_d = 1e9
for a, b in combinations(top_idx, 2):
    d = slab.get_distance(a, b, mic=True)
    if d < best_d:
        best_d = d
        best_pair = (a, b)

i, j = best_pair
# 向量（最小镜像）和中点（xy平面）
vec_ij = slab.get_distance(i, j, mic=True, vector=True)
ri = slab.positions[i]
rj_mic = ri + vec_ij  
xy_bridge = ((ri[0] + rj_mic[0]) / 2.0, (ri[1] + rj_mic[1]) / 2.0)

# ---------- add adsorbate ----------
ads = molecule("H2O")

height = 2.0  # Å
add_adsorbate(slab, ads, height=height, position=xy_bridge)

# optimize
slab.calc = calc
opt = LBFGS(slab, logfile="opt.log")
opt.run(fmax=0.05, steps=100)

print("Final energy (eV):", slab.get_potential_energy())
write("final.cif", slab)