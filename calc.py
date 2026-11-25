import pathlib
import os
import contextlib
from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ase.io
from scipy.sparse.csgraph import connected_components
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from pyscf import dft, gto
from pyscf.pbc.tools import pyscf_ase

data_path = pathlib.Path("data")

class RunParams:
    # Rotation Params
    init_ang_deg: float
    final_ang_deg: float
    step_num: float
    dihedral_atoms: list[int]
    supress_output: bool
    
class Results:
    def __init__(self, energy: float, angle: float, geom: ase.Atoms):
        self.energy: float = energy
        self.angle: float = angle
        self.geom: ase.Atoms = geom

class Molecule:
    def __init__(self, path: str, name: str, spin=1, charge=0):
        self.name: str = name
        self.init_geom: ase.Atoms = ase.io.read(path + "/" + self.get_xyz())
        self.init_geom.info.update({
            'spin': spin,
            'charge': charge
        })
        self.init_dihedral: int = 0
        self.results: list[Results] = []
    
    def get_xyz(self) -> str:
        return self.name + ".xyz"
    
    # calc is a lambda that accepts a molecule and outputs an energy
    def run(self, params: RunParams, calc: Callable[[ase.Atoms], float]) -> bool:
        # Initialize folder, copy geometry file, and route to it
        path = data_path / self.name
        data_path.mkdir(exist_ok=True)
        path.mkdir(exist_ok=True)
        
        # If already calculaed 
        #if any(path.iterdir()):
        #    return False
    
        # Initialize directory
        os.chdir(path)
        init_filename = f"{self.name}_intial.xyz"
        self.init_geom.write(init_filename)
        
        self.init_dihedral = self.init_geom.get_dihedral(params.dihedral_atoms[0], params.dihedral_atoms[1], params.dihedral_atoms[2], params.dihedral_atoms[3])
        
        # Create dihedral rotations
        for i, angle in enumerate(np.arange(params.init_ang_deg, params.final_ang_deg, params.step_num)):
            rotated = self.init_geom.copy()
            
            # Get the groups to rotate
            cutoffs = natural_cutoffs(rotated)
            nl = build_neighbor_list(rotated, cutoffs=cutoffs, self_interaction=False, bothways=True)
            conn = nl.get_connectivity_matrix(sparse=False)
            
            # Sever connection at rotated angle
            conn[params.dihedral_atoms[1], params.dihedral_atoms[2]] = 0
            conn[params.dihedral_atoms[2], params.dihedral_atoms[1]] = 0
            
            # Find all components conneced to first side
            _, labels = connected_components(conn, directed=False)
            mask = (labels == labels[params.dihedral_atoms[1]])
            
            # Commence rotation
            rotated.set_dihedral(
                params.dihedral_atoms[0], params.dihedral_atoms[1], params.dihedral_atoms[2], params.dihedral_atoms[3],
                angle=self.init_dihedral + angle,
                mask=mask
            )
            
            # Calculate energies and store
            # Muted output since the calculators like to output a lot of stuff
            if params.supress_output:
                with open(os.devnull, 'w') as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        energy = calc(rotated)
            else:
                energy = calc(rotated)
            self.results.append(Results(energy, angle, rotated))
            
            rotated.write(f"{self.name}_rot{angle}.xyz")
            ase.io.write("trajectory.xyz", rotated, append=True)  # Trajectory file
            
            print(f"Completed Task: {i}")
            
        os.chdir("../..")
        
        return True

    def gen_energy_surface(self, title: str):
        if not self.results:
            raise ValueError("Dihedral angles have not been generated!")
    
        angles = [result.angle for result in self.results]
        energies = [result.energy for result in self.results]
        
        plt.plot(angles, energies)
        plt.title(title)
        plt.xlabel("Dihedral Angle (deg)")
        plt.ylabel(f"Energy (Hartrees)")
        plt.show()