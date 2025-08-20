#---------------------------------------------ase install--------------------------------------------------
!pip install -U optuna
#!pip install -U pfp-api-client
#!pip install pfcc_extras
!pip install -U pfp-api-client matlantis-features
!pip install pfcc-extras-v0.11.1.zip
!pip install pfcc-ase-extras-v0.3.0.zip
#In addition, please install `pfcc_extras`.

import io
import os
import pandas as pd
import tempfile

from ase import Atoms
from ase.build import bulk, fcc111, molecule, add_adsorbate
from ase.constraints import ExpCellFilter, StrainFilter
from ase.io import write, read
from ase.io.jsonio import write_json, read_json
from ase.optimize import LBFGS, FIRE
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import optuna
from ase.visualize import view


import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

from pfcc_extras.visualize.view import view_ngl
from pfcc_extras.visualize.ase import view_ase_atoms

print(f"pfp_api_client: {pfp_api_client.__version__}")

# estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL, model_version="latest")
estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0, model_version="v3.0.0")
calculator = ASECalculator(estimator)


#---------------------------------------------catch care--------------------------------------------------
from care import gen_blueprint

#---------------------------------------------Option B: Network blueprint from list of SMILES----------------------------------------------
chemical_space = ['C(CO)O', 'CCO']  # SMILES strings
rearr_rxns = True
electro = False

inters, rxns = gen_blueprint(cs=chemical_space, 
                             additional_rxns=rearr_rxns,
                             electro=electro)

#----------------------------------------elementary reaction----------------------------------
r = rxns[10]
print(r)
print(type(r))
print(r.repr_hr)  # human-readable text representation
print(r.r_type)  # bond-breaking type
print(r.components)  # components of the reaction
print(r.stoic)  # Soichiometry coefficients in the reaction
print(r.e_rxn)  # reaction energy
print(r.e_act)  # reaction activation energy

a = inters['WSFSSNUMVMOOMR-UHFFFAOYSA-N*']
print(a)
print(type(a))
print(a.phase)
print(a.smiles)
print(a.is_closed_shell())
print(a['C'])  # number of carbon atoms
print(a['H'])  # number of hydrogen atoms
print(a['O'])  # number of oxygen atoms
print(a['N'])  # number of nitrogen atoms
print(a['S'])  # number of sulfur atoms
print(a.ads_configs)  # adsorption configurations


#from care.evaluators import load_surface
#build surface from ase not use contcar

#----------------------------------------load contcar----------------------------------
path = 'CONTCAR'  # Path to your CONTCAR surface file

surface = load_surface(path=path)

print(surface)
print(type(surface))
print(surface.slab)
print(surface.vacuum_height)


from ase.visualize.plot import plot_atoms
plot_atoms(surface.slab, rotation='-90x,0y,0z', show_unit_cell=2)


#----------------------------------------load energy evaluator----------------------------------
from care.evaluators import get_available_evaluators, GameNetUQInter, OCPIntermediateEvaluator, MACEIntermediateEvaluator, PETMADIntermediateEvaluator, ORBIntermediateEvaluator, SevenNetIntermediateEvaluator

from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

from pfcc_extras.visualize.view import view_ngl
from pfcc_extras.visualize.ase import view_ase_atoms

print(f"pfp_api_client: {pfp_api_client.__version__}")

# estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL, model_version="latest")
estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0, model_version="v3.0.0")
calculator = ASECalculator(estimator)

import tqdm
get_available_evaluators()

# GAME-Net-UQ
#inter_evaluator = GameNetUQInter(surface=surface,  num_configs=1,  use_uq=False)

def get_opt_energy(atoms, fmax=0.001, opt_mode: str = "normal"):    
    atoms.set_calculator(calculator)
    if opt_mode == "scale":
        opt1 = LBFGS(StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
    elif opt_mode == "all":
        opt1 = LBFGS(ExpCellFilter(atoms), logfile=None)
    else:
        opt1 = LBFGS(atoms, logfile=None)
    opt1.run(fmax=fmax)
    return atoms.get_total_energy()

class ASEInterEvaluator:
    def __init__(self, calculator, num_configs=1, opt_mode="normal", fmax=0.001):
        self.calculator = calculator
        self.num_configs = num_configs
        self.opt_mode = opt_mode
        self.fmax = fmax

    def __call__(self, atoms):
        atoms.set_calculator(self.calculator)
        if self.opt_mode == "scale":
            opt = LBFGS(StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
        elif self.opt_mode == "all":
            opt = LBFGS(ExpCellFilter(atoms), logfile=None)
        else:
            opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=self.fmax)
        energy = atoms.get_total_energy()
        atoms.info["energy"] = energy
        return energy

#inter_evaluator = get_opt_energy(atoms, fmax=0.001, opt_mode: str = "normal")
#inter_evaluator = ASEInterEvaluator(num_configs=1, opt_mode="normal", fmax=0.001)
inter_evaluator = ASEInterEvaluator(calculator, num_configs=1, opt_mode="normal", fmax=0.001)

#----------------------------------------add adsorbate----------------------------------
from care.adsorption import place_adsorbate

from rdkit.Chem import MolFromSmiles
from care import Intermediate

adsorbate_smiles = 'C(CO)O'  # SMILES string of ethylene glycol
adsorbate = MolFromSmiles(adsorbate_smiles) # Convert SMILES to RDKit Mol object

adsorbate_intermediate = Intermediate(molecule=adsorbate, phase='ads')

from care.evaluators import load_surface # Importing the load_surface function (only necessary if not already done)

metal = 'Pt'
facet = '110'  # hkl notation

surface = load_surface(metal, facet)

from care.adsorption import connectivity_analysis # Perform connectivity analysis to find anchoring points for the adsorbate
from care.adsorption import get_active_sites # Get active sites on the surface for adsorption
from care.adsorption import adapt_surface # Adapt the surface according to the size of the adsorbate

# NOTE: CARE has implemented a streamlined version of DockOnSurf for the adsorbate placement. To check the input parameters used for DockOnSurf check the following function

from care.adsorption import generate_inp_vars

list_ads_configs = place_adsorbate(adsorbate_intermediate,
                                   surface=surface,
                                   num_configs=10,
                                   )

#----------------------------------------Sequential evaluation----------------------------------
print(f'{len(inters)*inter_evaluator.num_configs} intermediate configurations to evaluate')
for k, inter in tqdm.tqdm(inters.items()):
    inter_evaluator(inter)
    
inters_evaluated = inters

#----------------------------------------Parallel evaluation----------------------------------
from dask.distributed import Client
CORES = 12

def mp_func(inter):
    print(inter.code + "\n")
    inter_evaluator(inter)
    return inter

tasks = [inter for inter in inters.values()]
futures = client.map(mp_func, tasks)
results = client.gather(futures)
inters_evaluated = {i.code: i for i in results}
client = Client(threads_per_worker=1, n_workers=CORES)

# get random key and print the adsorption configurations
key = list(inters_evaluated.keys())[18]
inters_evaluated[key].ads_configs