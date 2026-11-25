import pyscf
import ase

dft_basis = "3-21g"
dft_method = 'pbe0'

def dft(molecule: ase.Atoms) -> float:
    pyscf_atoms = [(atom.symbol, atom.position) for atom in molecule]
    rotated_pyscf = pyscf.gto.M(atom=pyscf_atoms, basis=dft_basis)
    dft_method = pyscf.dft.RKS(rotated_pyscf, xc=dft_method)
    dft_method.kernel()
    return dft_method.energy_tot()