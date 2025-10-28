import os

import torch
from pysisyphus.constants import AU2EV, BOHR2ANG
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
import scine_utilities as su


def get_uma_calculator(device="cpu", ver="finetuned", task_name="omol", model="uma-s-1p1"):
    # Disable CUDA if using CPU to prevent CUDA initialization errors
    if device == "cpu":
        # Temporarily disable CUDA to prevent initialization errors
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

    kwargs = {
        "device": device,
    }
    if ver == "finetuned":
        ckpt_path = os.getenv("REACTBENCH_CKPT_PATH", "/root/ReactBench/ckpt/")
        kwargs["path"] = str(os.path.join(ckpt_path, "mace.ckpt"))
    elif ver == "pretrain":
        kwargs["model_name"] = model
    else:
        raise ValueError(f"Unsupported version: {ver}")

    try:
        if ver == 'finetuned':
            predictor = load_predict_unit(**kwargs)
        elif ver == 'pretrain':
            predictor = pretrained_mlip.get_predict_unit(**kwargs)
        else:
            raise ValueError(f"Unsupported version: {ver}")
        return FAIRChemCalculator(predictor, task_name=task_name)
    finally:
        # Restore original CUDA availability check
        if device == "cpu":
            torch.cuda.is_available = original_cuda_available


class UMAMLFF:
    """MACE calculator for pysisyphus"""
    
    def __init__(self, device: str = "cpu", ver: str = "finetuned", task_name: str = "omol",
                 model: str = "uma-s-1p1"):
        """
        Initialize UMA calculator
        
        Parameters
        ----------
        device : str
            Device to run calculations on ('cpu' or 'cuda')
        ver : str
            Version of UMA model to use ('finetuned' or 'pretrain')
        task_name : str
            Task name for the UMA model
        model : str
            Model name for the UMA pre-trained model
        """
        self.device = device
        self.model = get_uma_calculator(device, ver, task_name, model)
        from ase_scine_bridge import AseForScineCalculator
        from ase_scine_bridge.data_transformations import atoms_to_atomcollection

        def get_calc(_):
            return self.model

        AseForScineCalculator.get_ase_calculator = get_calc
        self.scine_calc = AseForScineCalculator()
        self.atom_transformation = atoms_to_atomcollection

    def get_energy(self, molecule):
        """Get energy for pysisyphus interface"""
        
        molecule.calc = self.model
        energy = molecule.get_potential_energy() / AU2EV
        
        results = {
            "energy": energy,
        }
        return results
    
    def get_forces(self, molecule):
        """Get forces for pysisyphus interface"""

        molecule.calc = self.model
        energy = molecule.get_potential_energy() / AU2EV
        forces = molecule.get_forces() / AU2EV * BOHR2ANG
        
        results = {
            "energy": energy,
            "forces": forces.flatten(),
        }
        return results
    
    def get_hessian(self, molecule):
        """Get Hessian for pysisyphus interface"""

        self.scine_calc.structure = self.atom_transformation(molecule)
        self.scine_calc.set_required_properties([su.Property.Energy, su.Property.Hessian])
        results = self.scine_calc.calculate()

        results = {
            "energy": results.energy,
            "hessian": results.hessian,
        }
        return results




