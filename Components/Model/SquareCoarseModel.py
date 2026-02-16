from typing import Tuple, Dict, List

import math
import numpy as np
import scipy.sparse as sp

from scipy.signal import convolve2d

from ..ParameterInterface import ParameterInterface
from .CoarseModel import CoarseModel

class SquareCoarseModel(CoarseModel):
    
    def __init__(self, param_handler: ParameterInterface, boundary_condition: str="Free-flow"):
        super().__init__(param_handler, boundary_condition)
    
    def _average_value(self, var_HD: np.ndarray) -> np.ndarray:
        # TODO: adapt method to manage other original geometries
        # Compute discretization ratio
        n_x, n_y = var_HD.shape
        x_ratio = n_x / self.n_col
        y_ratio = n_y / self.n_row
        # Construct averaging filter
        avg_filter = np.ones((math.floor(x_ratio), math.floor(y_ratio)))
        avg_filter /= np.sum(avg_filter)
        # Apply the filter
        var = convolve2d(var_HD, avg_filter, mode="valid")[::math.ceil(x_ratio), ::math.ceil(y_ratio)]
        return var
    
    def _compute_well_to_cell(self, well_pos: List[List[int]],
                              Lx: float, Ly: float) -> sp.csr_array:
        # Construct well to cell mapping
        mapping = sp.lil_array((self.n_row * self.n_col, len(well_pos)))
        for k, coord in enumerate(well_pos):
            row = int(coord[1]*self.n_row/Lx)
            col = int(coord[2]*self.n_col/Ly)
            mapping[row * self.n_col + col, k] += 1
        return mapping.tocsr()
    
    def _compute_command_to_cell(self) -> List[sp.csr_array]:
        unique, index =\
            np.unique(self.well_to_cell.toarray(), axis=1, return_index=True)
        to_player_agg = self.well_to_agent[index]
        return [sp.csr_array(unique[:, to_player_agg==i]) for i in range(self.n_agent)]
    
    def _compute_input_to_command(self) -> List[sp.csr_array]:
        # In the simple case, an input correspond to one pumping command
        self.unit_input_dim = 1
        return [sp.eye(n_c) for n_c in self.n_command_v]