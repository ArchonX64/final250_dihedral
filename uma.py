import ase
from fairchem.core import FAIRChemCalculator, pretrained_mlip

uma_model = 'uma-s-1p2p1-ft-d4rev'

# Fairchem UMA
predictor = pretrained_mlip.get_predict_unit(uma_model, device='cuda')
uma_calc = FAIRChemCalculator(predictor)

def uma(molecule: ase.Atoms) -> float:
    molecule.calc = uma_calc
    return molecule.get_total_energy()