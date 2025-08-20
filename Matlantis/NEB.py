#prepare define functional
!pip install pfp-api-client
!pip install pandas tqdm matplotlib seaborn optuna sella scikit-learn torch torch_dftd

#from ase.neb import SingleCalculatorNEB
from ase.calculators.dftd3 import DFTD3
from ase.mep import neb

# 汎用モジュール
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from IPython.display import display_png
from IPython.display import Image as ImageWidget
import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.animation import PillowWriter
import seaborn as sns
import math
import optuna
import nglview as nv
import os,sys,csv,glob,shutil,re,time
from pathlib import Path
from PIL import Image, ImageDraw

# sklearn
from sklearn.metrics import mean_absolute_error


# ASE
import ase
from ase import Atoms, units
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal
from ase.io import read, write
from ase.build import surface, molecule, add_adsorbate
from ase.cluster.cubic import FaceCenteredCubic
from ase.constraints import FixAtoms, FixedPlane, FixBondLength, ExpCellFilter
#from ase.neb import SingleCalculatorNEB
from ase.neb import NEB
from ase.vibrations import Vibrations
from ase.visualize import view
from ase.optimize import QuasiNewton
from ase.thermochemistry import IdealGasThermo
from ase.build.rotate import minimize_rotation_and_translation
from ase.visualize import view
from ase.optimize import BFGS, LBFGS, FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write, Trajectory
# from ase.calculators.dftd3 import DFTD3
from ase.build import sort

from sella import Sella, Constraints
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

# PFP
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator
from pfp_api_client.pfp.estimator import EstimatorCalcMode

estimator = Estimator(calc_mode="CRYSTAL")
calculator = ASECalculator(estimator)

def myopt(m,sn = 10,constraintatoms=[],cbonds=[]):
    fa = FixAtoms(indices=constraintatoms)
    fb = FixBondLengths(cbonds,tolerance=1e-5,)
    m.set_constraint([fa,fb])
    m.set_calculator(calculator)
    maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
    print("ini   pot:{:.4f},maxforce:{:.4f}".format(m.get_potential_energy(),maxf))
    de = -1 
    s = 1
    ita = 50
    while ( de  < -0.001 or de > 0.001 ) and s <= sn :
        opt = BFGS(m,maxstep=0.04*(0.9**s),logfile=None)
        old  =  m.get_potential_energy() 
        opt.run(fmax=0.0005,steps =ita)
        maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
        de =  m.get_potential_energy()  - old
        print("{} pot:{:.4f},maxforce:{:.4f},delta:{:.4f}".format(s*ita,m.get_potential_energy(),maxf,de))
        s += 1
    return m

def opt_cell_size(m,sn = 10, iter_count = False): # m:Atomsオブジェクト
    m.set_constraint() # clear constraint
    m.set_calculator(calculator)
    maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max()) # √(fx^2 + fy^2 + fz^2)の一番大きいものを取得
    ucf = ExpCellFilter(m)
    print("ini   pot:{:.4f},maxforce:{:.4f}".format(m.get_potential_energy(),maxf))
    de = -1 
    s = 1
    ita = 50
    while ( de  < -0.01 or de > 0.01 ) and s <= sn :
        opt = BFGS(ucf,maxstep=0.04*(0.9**s),logfile=None)
        old  =  m.get_potential_energy() 
        opt.run(fmax=0.005,steps =ita)
        maxf = np.sqrt(((m.get_forces())**2).sum(axis=1).max())
        de =  m.get_potential_energy()  - old
        print("{} pot:{:.4f},maxforce:{:.4f},delta:{:.4f}".format(s*ita,m.get_potential_energy(),maxf,de))
        s += 1
    if iter_count == True:
        return m, s*ita
    else:
        return m
    
#表面を作る
def makesurface(atoms,miller_indices=(1,1,1),layers=4,rep=[4,4,1]):
    s1 = surface(atoms, miller_indices,layers)
    s1.center(vacuum=20.0, axis=2)
    s1 = s1.repeat(rep)
    s1.set_positions(s1.get_positions() - [0,0,min(s1.get_positions()[:,2])])
    s1.pbc = True
    return s1

import threading
import time
from math import pi
from typing import Dict, List, Optional

import nglview as nv
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.visualize import view
from IPython.display import display
from ipywidgets import (
    Button,
    Checkbox,
    FloatSlider,
    GridspecLayout,
    HBox,
    IntSlider,
    Label,
    Text,
    Textarea,
)
from nglview.widget import NGLWidget


def save_image(filename: str, v: NGLWidget):
    """Save nglview image.

    Note that it should be run on another thread.
    See: https://github.com/nglviewer/nglview/blob/master/docs/FAQ.md#how-to-make-nglview-view-object-write-png-file

    Args:
        filename (str):
        v (NGLWidget):
    """
    image = v.render_image()
    while not image.value:
        time.sleep(0.1)
    with open(filename, "wb") as fh:
        fh.write(image.value)


class SurfaceEditor:
    """Structure viewer/editor"""

    struct: List[Dict]  # structure used for nglview drawing.

    def __init__(self, atoms: Atoms):
        self.atoms = atoms
        self.vh = view(atoms, viewer="ngl")
        self.v: NGLWidget = self.vh.children[0]  # VIEW
        self.v._remote_call("setSize", args=["450px", "450px"])
        self.recont()  # Add controller
        self.set_representation()
        self.set_atoms()
        self.pots = []
        self.traj = []
        self.cal_nnp()

    def display(self):
        display(self.vh)

    def recont(self):
        self.vh.setatoms = FloatSlider(
            min=0, max=50, step=0.1, value=8, description="atoms z>"
        )
        self.vh.setatoms.observe(self.set_atoms)
        self.vh.selected_atoms_label = Label("Selected atoms:")
        self.vh.selected_atoms_textarea = Textarea()
        selected_atoms_hbox = HBox(
            [self.vh.selected_atoms_label, self.vh.selected_atoms_textarea]
        )
        self.vh.move = FloatSlider(
            min=0.1, max=2, step=0.1, value=0.5, description="move"
        )

        grid1 = GridspecLayout(2, 3)
        self.vh.xplus = Button(description="X+")
        self.vh.xminus = Button(description="X-")
        self.vh.yplus = Button(description="Y+")
        self.vh.yminus = Button(description="Y-")
        self.vh.zplus = Button(description="Z+")
        self.vh.zminus = Button(description="Z-")
        self.vh.xplus.on_click(self.move)
        self.vh.xminus.on_click(self.move)
        self.vh.yplus.on_click(self.move)
        self.vh.yminus.on_click(self.move)
        self.vh.zplus.on_click(self.move)
        self.vh.zminus.on_click(self.move)
        grid1[0, 0] = self.vh.xplus
        grid1[0, 1] = self.vh.yplus
        grid1[0, 2] = self.vh.zplus
        grid1[1, 0] = self.vh.xminus
        grid1[1, 1] = self.vh.yminus
        grid1[1, 2] = self.vh.zminus

        self.vh.rotate = FloatSlider(
            min=1, max=90, step=1, value=30, description="rotate"
        )
        grid2 = GridspecLayout(2, 3)
        self.vh.xplus2 = Button(description="X+")
        self.vh.xminus2 = Button(description="X-")
        self.vh.yplus2 = Button(description="Y+")
        self.vh.yminus2 = Button(description="Y-")
        self.vh.zplus2 = Button(description="Z+")
        self.vh.zminus2 = Button(description="Z-")
        self.vh.xplus2.on_click(self.rotate)
        self.vh.xminus2.on_click(self.rotate)
        self.vh.yplus2.on_click(self.rotate)
        self.vh.yminus2.on_click(self.rotate)
        self.vh.zplus2.on_click(self.rotate)
        self.vh.zminus2.on_click(self.rotate)
        grid2[0, 0] = self.vh.xplus2
        grid2[0, 1] = self.vh.yplus2
        grid2[0, 2] = self.vh.zplus2
        grid2[1, 0] = self.vh.xminus2
        grid2[1, 1] = self.vh.yminus2
        grid2[1, 2] = self.vh.zminus2

        self.vh.nnptext = Textarea(disabled=True)

        self.vh.opt_step = IntSlider(
            min=0,
            max=100,
            step=1,
            value=10,
            description="Opt steps",
        )
        self.vh.constraint_checkbox = Checkbox(
            value=True,
            description="Opt only selected atoms",
        )
        self.vh.run_opt_button = Button(
            description="Run mini opt",
            tooltip="Execute BFGS optimization with small step update."
        )
        self.vh.run_opt_button.on_click(self.run_opt)
        opt_hbox = HBox([self.vh.constraint_checkbox, self.vh.run_opt_button])

        self.vh.filename_text = Text(value="screenshot.png", description="filename: ")
        self.vh.download_image_button = Button(
            description="download image",
            tooltip="Download current frame to your local PC",
        )
        self.vh.download_image_button.on_click(self.download_image)
        self.vh.save_image_button = Button(
            description="save image",
            tooltip="Save current frame to file.\n"
                    "Currently .png and .html are supported.\n"
                    "It takes a bit time, please be patient.",
        )
        self.vh.save_image_button.on_click(self.save_image)

        self.vh.update_display = Button(
            description="update_display",
            tooltip="Refresh display. It can be used when target atoms is updated in another cell..",
        )
        self.vh.update_display.on_click(self.update_display)

        r = list(self.vh.control_box.children)
        r += [
            self.vh.setatoms,
            selected_atoms_hbox,
            self.vh.move,
            grid1,
            self.vh.rotate,
            grid2,
            self.vh.nnptext,
            self.vh.opt_step,
            opt_hbox,
            self.vh.filename_text,
            HBox([self.vh.download_image_button, self.vh.save_image_button]),
            self.vh.update_display,
        ]
        self.vh.control_box.children = tuple(r)

    def set_representation(self, bcolor: str = "white", unitcell: bool = True):
        self.v.background = bcolor
        self.struct = self.get_struct(self.atoms)
        self.v.add_representation(repr_type="ball+stick")
        self.v.control.spin([0, 1, 0], pi * 1.1)
        self.v.control.spin([1, 0, 0], -pi * 0.45)
        thread = threading.Thread(target=self.changestr)
        thread.start()

    def changestr(self):
        time.sleep(2)
        self.v._remote_call("replaceStructure", target="Widget", args=self.struct)

    def get_struct(self, atoms: Atoms, ext="pdb") -> List[Dict]:
        struct = nv.ASEStructure(atoms, ext=ext).get_structure_string()
        for c in range(len(atoms)):
            struct = struct.replace("MOL     1", "M0    " + str(c).zfill(3), 1)
        struct = [dict(data=struct, ext=ext)]
        return struct

    def cal_nnp(self):
        pot = self.atoms.get_potential_energy()
        mforce = (((self.atoms.get_forces()) ** 2).sum(axis=1).max()) ** 0.5
        self.pot = pot
        self.mforce = mforce
        self.vh.nnptext.value = f"pot energy: {pot} eV\nmax force : {mforce} eV/A"
        self.pots += [pot]
        self.traj += [self.atoms.copy()]

    def update_display(self, clicked_button: Optional[Button] = None):
        print("update display!")
        struct = self.get_struct(self.atoms)
        self.struct = struct
        self.v._remote_call("replaceStructure", target="Widget", args=struct)
        self.cal_nnp()

    def set_atoms(self, slider: Optional[FloatSlider] = None):
        """Update text area based on the atoms position `z` greater than specified value."""
        smols = [
            i for i, atom in enumerate(self.atoms) if atom.z >= self.vh.setatoms.value
        ]
        self.vh.selected_atoms_textarea.value = ", ".join(map(str, smols))

    def get_selected_atom_indices(self) -> List[int]:
        selected_atom_indices = self.vh.selected_atoms_textarea.value.split(",")
        selected_atom_indices = [int(a) for a in selected_atom_indices]
        return selected_atom_indices

    def move(self, clicked_button: Button):
        a = self.vh.move.value

        for index in self.get_selected_atom_indices():
            if clicked_button.description == "X+":
                self.atoms[index].position += [a, 0, 0]
            elif clicked_button.description == "X-":
                self.atoms[index].position -= [a, 0, 0]
            elif clicked_button.description == "Y+":
                self.atoms[index].position += [0, a, 0]
            elif clicked_button.description == "Y-":
                self.atoms[index].position -= [0, a, 0]
            elif clicked_button.description == "Z+":
                self.atoms[index].position += [0, 0, a]
            elif clicked_button.description == "Z-":
                self.atoms[index].position -= [0, 0, a]
        self.update_display()

    def rotate(self, clicked_button: Button):
        atom_indices = self.get_selected_atom_indices()
        deg = self.vh.rotate.value
        temp = self.atoms[atom_indices]

        if clicked_button.description == "X+":
            temp.rotate(deg, "x", center="COP")
        elif clicked_button.description == "X-":
            temp.rotate(-deg, "x", center="COP")
        elif clicked_button.description == "Y+":
            temp.rotate(deg, "y", center="COP")
        elif clicked_button.description == "Y-":
            temp.rotate(-deg, "y", center="COP")
        elif clicked_button.description == "Z+":
            temp.rotate(deg, "z", center="COP")
        elif clicked_button.description == "Z-":
            temp.rotate(-deg, "z", center="COP")
        rotep = temp.positions
        for i, atom in enumerate(atom_indices):
            self.atoms[atom].position = rotep[i]
        self.update_display()

    def run_opt(self, clicked_button: Button):
        """OPT only specified steps and FIX atoms if NOT in text atoms list"""
        if self.vh.constraint_checkbox.value:
            # Fix non selected atoms. Only opt selected atoms.
            print("Opt with selected atoms: fix non selected atoms")
            atom_indices = self.get_selected_atom_indices()
            constraint_atom_indices = [
                i for i in range(len(self.atoms)) if i not in atom_indices
            ]
            self.atoms.set_constraint(FixAtoms(indices=constraint_atom_indices))
        opt = BFGS(self.atoms, maxstep=0.04, logfile=None)
        steps: Optional[int] = self.vh.opt_step.value
        if steps < 0:
            steps = None  # When steps=-1, opt until converged.
        opt.run(fmax=0.0001, steps=steps)
        print(f"Run opt for {steps} steps")
        self.update_display()

    def download_image(self, clicked_button: Optional[Button] = None):
        filename = self.vh.filename_text.value
        self.v.download_image(filename=filename)

    def save_image(self, clicked_button: Optional[Button] = None):
        filename = self.vh.filename_text.value
        if filename.endswith(".png"):
            thread = threading.Thread(
                target=save_image, args=(filename, self.v), daemon=True
            )
            # thread.daemon = True
            thread.start()
        elif filename.endswith(".html"):
            nv.write_html(filename, [self.v])  # type: ignore
        else:
            print(f"filename {filename}: extension not supported!")

#---------------------------------------------------------------------------------------------------------------
#build bulk and surface

bulk = read("ZnO.cif")
print("Number of atoms =", len(bulk))
print("Initial lattice constant =", bulk.cell.cellpar())

bulk.calc = calculator
opt = LBFGS(ExpCellFilter(bulk))
opt.run()
print ("Optimized lattice constant =", bulk.cell.cellpar())

bulk = bulk.repeat([2,2,2])               
bulk = sort(bulk)
bulk.positions += [0.01,0,0]   # 面を切るときに変なところで切れるのを防ぐために少し下駄を履かせます。
v = view(bulk, viewer='ngl')
v.view.add_representation("ball+stick")
display(v)

slab = makesurface(bulk, miller_indices=(0,1,0), layers=2, rep=[2,3,1])
slab = sort(slab)
slab.positions += [1,1,0]         # 少しだけ位置調整
slab.wrap()                       # してからWRAP

v = view(slab, viewer='ngl')
v.view.add_representation("ball+stick")
display(v)

# 原子のz_positionを確認。
z_pos = pd.DataFrame({
    "symbol": slab.get_chemical_symbols(),
    "z": slab.get_positions()[:, 2]
})

plt.scatter(z_pos.index, z_pos["z"])
plt.grid(True)
plt.xlabel("atom_index")
plt.ylabel("z_position")
plt.show()
print("highest position (z) =", z_pos["z"].max())

%%time
c = FixAtoms(indices=[atom.index for atom in slab if atom.position[2] <= 5])       # 1A以下を固定
slab.set_constraint(c)
slab.calc = calculator

os.makedirs("output", exist_ok=True)
BFGS_opt = BFGS(slab, trajectory="output/slab_opt.traj")#, logfile=None)
BFGS_opt.run(fmax=0.005)

# v = view(slab, viewer='ngl')
v = view(Trajectory("output/slab_opt.traj"), viewer='ngl')
v.view.add_representation("ball+stick")
display(v)

slabE = slab.get_potential_energy()
print(f"slab E = {slabE} eV")

# 作成したslab構造を保存
os.makedirs("structures/", exist_ok=True)      # structures/というフォルダを作成します。
write("structures/Slab_ZnO.cif", slab)      # 任意でファイル名を変更してください。

#---------------------------------------------------------------------------------------------------------------
#molecule build
#print(os.getcwd()) 
#molec = molecule('NO')
molec = read("structures/acid.cif") # sdf fileの読み込み例

molec.calc = calculator
BFGS_opt = BFGS(molec, trajectory="output/molec_opt.traj", logfile=None)
BFGS_opt.run(fmax=0.005)
molecE = molec.get_potential_energy()
print(f"molecE =　{molecE} eV")

v = view(Trajectory("output/molec_opt.traj"), viewer='ngl')
v.view.add_representation("ball+stick")
display(v)

mol_on_slab = slab.copy()

# slab最表面から分子を配置する高さと、x,y positionを入力してください。
# あとで調整できるので、適当で大丈夫です。
add_adsorbate(mol_on_slab, molec, height=3, position=(8, 4))
c = FixAtoms(indices=[atom.index for atom in mol_on_slab if atom.position[2] <= 1])
mol_on_slab.set_constraint(c)

# SurfaceEditor にはcalculator がSet されている必要があります。
mol_on_slab.calc = calculator

se = SurfaceEditor(mol_on_slab)
se.display()

c = FixAtoms(indices=[atom.index for atom in mol_on_slab if atom.position[2] <= 1])
mol_on_slab.set_constraint(c)
BFGS_opt = BFGS(mol_on_slab, logfile=None)
BFGS_opt.run(fmax=0.005)
mol_on_slabE = mol_on_slab.get_potential_energy()
print(f"mol_on_slabE = {mol_on_slabE} eV")

os.makedirs("ad_structures/",  exist_ok=True)
write("ad_structures/mol_on_ZnO.cif", mol_on_slab)

#---------------------------------------------------------------------------------------------------------------
#build FS and IS
# Calculate adsorption energy
adsorpE = slabE + molecE - mol_on_slabE
print(f"Adsorption Energy: {adsorpE} eV")

ad_st_path = "ad_structures/*"
ad_stru_list = [(filepath, read(filepath)) for filepath in glob.glob(ad_st_path)]

pd.DataFrame(ad_stru_list)

No = 0
view(ad_stru_list[No][1] , viewer="ngl")

#filepath, atoms = ad_stru_list[No]
#print(filepath)
#IS = read(surface_IS.cif)
#IS = atoms.copy()

IS = read("ad_structures/IS.cif")
FS = read("ad_structures/FS.cif")

IS.calc = calculator
SurfaceEditor(IS).display()

c = FixAtoms(indices=[atom.index for atom in IS if atom.position[2] <= 1])
IS.set_constraint(c)
BFGS_opt = BFGS(IS, logfile=None)
BFGS_opt.run(fmax=0.05)
IS.get_potential_energy()

FS = IS.copy()
FS.calc = calculator
SurfaceEditor(FS).display()

FS.calc = calculator
c = FixAtoms(indices=[atom.index for atom in FS if atom.position[2] <= 1])
FS.set_constraint(c)
BFGS_opt = BFGS(FS, logfile=None)
BFGS_opt.run(fmax=0.005)
FS.get_potential_energy()

filepath = Path(filepath).stem
# filepath = Path(ad_stru_list[No][0]).stem
os.makedirs(filepath, exist_ok=True)
write(filepath+"/IS.cif", IS)
write(filepath+"/FS.cif", FS)

#---------------------------------------------------------------------------------------------------------------


#NEB calculation
filepath = "ad_structures"
# filepath = "NO_dissociation_NO(fcc)_N(hcp)_O(hcp)"

def fix_subsurface(atoms, z_threshold):
    indices = [atom.index for atom in atoms if atom.position[2] < z_threshold]
    atoms.set_constraint(FixAtoms(indices=indices))

# 读取结构
IS = read(filepath+"/IS.cif")
FS = read(filepath+"/FS.cif")

# 计算z阈值（如表面z的最大值-1.0 Å，1.0可根据实际情况调整）
z_threshold = 5.5

v = view([IS, FS], viewer='ngl')
#v.view.add_representation("ball+stick")
display(v)

c = FixAtoms(indices=[atom.index for atom in IS if atom.position[2] <= z_threshold])
IS.calc = calculator
IS.set_constraint(c)
BFGS_opt = BFGS(IS, logfile=None)
BFGS_opt.run(fmax=0.005)
print(f"IS {IS.get_potential_energy()} eV")

c = FixAtoms(indices=[atom.index for atom in FS if atom.position[2] <= z_threshold])
FS.calc = calculator
FS.set_constraint(c)
BFGS_opt = BFGS(FS, logfile=None)
BFGS_opt.run(fmax=0.005)
print(f"FS {FS.get_potential_energy()} eV")

beads = 21

b0 = IS.copy()
b1 = FS.copy()
configs = [b0.copy() for i in range(beads-1)] + [b1.copy()]
for config in configs:
    estimator = Estimator()                     # NEBでparallel=True, allowed_shared_calculator=Falseにする場合に必要
    calculator = ASECalculator(estimator)       # NEBでparallel=True, allowed_shared_calculator=Falseにする場合に必要
    config.calc = calculator

%%time
steps=2000

# k：ばねの定数　最終的に0.05とか下げた方が安定する。
# NEBでparallel = True, allowed_shared_calculator=Falseにしたほうが、他のjobの影響を受けにくいので高速に処理が進みます。
neb = NEB(configs, k=0.05, parallel=True, climb=True, allow_shared_calculator=False)   
neb.interpolate()
relax = FIRE(neb, trajectory=None, logfile=filepath+"/neb_log.txt")

# fmaxは0.05以下が推奨。小さすぎると収束に時間がかかります。
# 一回目のNEB計算は収束条件を緩め（0.2など）で実行し、無理のない反応経路が描けていたら収束条件を厳しくするほうが安定して計算できます。
# 緩めの収束条件で異常な反応経路となる場合はIS, FS構造を見直してください。
relax.run(fmax=0.1, steps=steps)

# additional calculation
steps=10000
relax.run(fmax=0.05, steps=steps)              

write(filepath+"/NEB_images.xyz", configs)
    