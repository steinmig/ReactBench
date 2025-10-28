from pysisyphus.constants import AU2EV, BOHR2ANG
from mace.calculators import mace_off_finetuned, mace_off
import torch

def get_mace_calculator(device="cpu", ver='finetuned'):
    import mace
    import e3nn
    torch.serialization.add_safe_globals([mace.modules.models.ScaleShiftMACE])
    torch.serialization.add_safe_globals([mace.modules.blocks.LinearNodeEmbeddingBlock])
    torch.serialization.add_safe_globals([e3nn.o3._linear.Linear])
    torch.serialization.add_safe_globals([e3nn.o3._irreps.Irreps])
    torch.serialization.add_safe_globals([e3nn.o3._irreps._MulIr])
    torch.serialization.add_safe_globals([e3nn.o3._irreps.Irrep])
    torch.serialization.add_safe_globals([e3nn.o3._linear.Instruction])
    torch.serialization.add_safe_globals([mace.modules.blocks.RadialEmbeddingBlock])
    torch.serialization.add_safe_globals([mace.modules.radial.BesselBasis])
    torch.serialization.add_safe_globals([mace.modules.radial.PolynomialCutoff])
    torch.serialization.add_safe_globals([e3nn.o3._spherical_harmonics.SphericalHarmonics])
    torch.serialization.add_safe_globals([mace.modules.blocks.AtomicEnergiesBlock])
    torch.serialization.add_safe_globals([torch.nn.modules.container.ModuleList])
    torch.serialization.add_safe_globals([mace.modules.blocks.RealAgnosticInteractionBlock])
    torch.serialization.add_safe_globals([e3nn.o3._tensor_product._tensor_product.TensorProduct])
    torch.serialization.add_safe_globals([e3nn.o3._tensor_product._instruction.Instruction])
    torch.serialization.add_safe_globals([e3nn.nn._fc.FullyConnectedNet])
    torch.serialization.add_safe_globals([e3nn.nn._fc._Layer])
    torch.serialization.add_safe_globals([e3nn.math._normalize_activation.normalize2mom])
    torch.serialization.add_safe_globals([torch.nn.functional.silu])
    torch.serialization.add_safe_globals([e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct])
    torch.serialization.add_safe_globals([mace.modules.irreps_tools.reshape_irreps])
    torch.serialization.add_safe_globals([mace.modules.blocks.RealAgnosticResidualInteractionBlock])
    torch.serialization.add_safe_globals([mace.modules.blocks.EquivariantProductBasisBlock])
    torch.serialization.add_safe_globals([mace.modules.symmetric_contraction.SymmetricContraction])
    torch.serialization.add_safe_globals([mace.modules.symmetric_contraction.Contraction])
    torch.serialization.add_safe_globals([torch.fx.graph_module.reduce_graph_module])
    torch.serialization.add_safe_globals([torch.fx._symbolic_trace.Tracer])
    torch.serialization.add_safe_globals([torch.nn.modules.container.ParameterList])
    torch.serialization.add_safe_globals([mace.modules.blocks.LinearReadoutBlock])
    torch.serialization.add_safe_globals([mace.modules.blocks.NonLinearReadoutBlock])
    torch.serialization.add_safe_globals([e3nn.nn._activation.Activation])
    torch.serialization.add_safe_globals([mace.modules.blocks.ScaleShiftBlock])
    # Disable CUDA if using CPU to prevent CUDA initialization errors
    if device == "cpu":
        # Temporarily disable CUDA to prevent initialization errors
        original_cuda_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
    
    try:
        if ver == 'finetuned':
            return mace_off_finetuned(device=device, model='/root/ReactBench/ckpt/mace.ckpt', weights_only=False)
        elif ver == 'pretrain':
            return mace_off(model="medium", default_dtypes='float64', weights_only=False) 
    finally:
        # Restore original CUDA availability check
        if device == "cpu":
            torch.cuda.is_available = original_cuda_available


class MACEMLFF:
    """MACE calculator for pysisyphus"""
    
    def __init__(self, device="cpu", ver='finetuned'):
        """
        Initialize MACE calculator
        
        Parameters
        ----------
        device : str
            Device to run calculations on ('cpu' or 'cuda')
        ver : str
            Version of MACE model to use ('finetuned' or 'pretrain')
        """
        self.device = device
        self.model = get_mace_calculator(device=device, ver=ver)
    
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

        molecule.calc = self.model
        hessian = self.model.get_hessian(atoms=molecule).reshape(molecule.get_number_of_atoms()*3,\
                                                                    molecule.get_number_of_atoms()*3) / AU2EV * BOHR2ANG * BOHR2ANG
        energy = molecule.get_potential_energy() / AU2EV 
        
        results = {
            "energy": energy,
            "hessian": hessian,
        }
        return results




