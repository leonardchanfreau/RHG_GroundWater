from typing import Tuple, Dict, List

import json
import numpy as np
import scipy.sparse as sp

from scipy.signal import convolve2d

from ..ParameterInterface import ParameterInterface
from .SquareCoarseModel import SquareCoarseModel

class IrrigationSchedulingModel(SquareCoarseModel):
    """Very similar model as SquareCoarseModel, except for the control vector.
    In this model, the pumped water is also allocated to certain type of crops.
    The size of u_t is therefore N_well x N_crop."""
    
    def __init__(self, param_handler: ParameterInterface, boundary_condition: str="Free-flow"):
        self.n_crops = len(param_handler.crop_repartition)
        super().__init__(param_handler, boundary_condition)
    
    def _compute_input_to_command(self) -> List[sp.csr_array]:
        # The pumping command is the sum of inputs for each crops
        self.unit_input_dim = self.n_crops
        return [sp.kron(sp.eye(n_c), np.ones((1,self.n_crops)))
                for n_c in self.n_command_v]