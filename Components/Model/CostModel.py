from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from ..ParameterInterface import ParameterInterface
from .CoarseModel import CoarseModel

class CostModel(ABC):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_handler: ParameterInterface, physical_model: CoarseModel):
        # Dimensions parameters
        self.nx = param_handler.nx
        self.n_agent = param_handler.n_agent
        
        # Well positions
        self.well_to_cell = physical_model.well_to_cell
        self.input_v_to_state = [c2x @ u2c for c2x, u2c in
                                 zip(physical_model.command_v_to_cell,
                                     physical_model.input_v_to_command_v)]
        self.n_well_per_command = physical_model.n_well_per_command
        self.input_v_to_command_v = physical_model.input_v_to_command_v
        self.well_to_agent = param_handler.well_to_agent
        
        # Heads at the top of the wells
        self.top_head = param_handler.get_initial_state()
                
        # Correct cost using corrected heads
        correction = param_handler.correction
        if (correction is not None) and (correction['correct_cost']):
            self.correct_cost = True
            self.cost_correction = correction.get("cost")
            if physical_model.correct_heads:
                self.heads_correction = physical_model.heads_correction
        
        # Compute cost matrices
        self.extraction_cost_v, self.selling_crop_price_c = param_handler.get_cost_coefficients()
        self._compute_base_cost_matrices(physical_model)
    
    
    @abstractmethod
    def get_cost_matrices(self, time: int = None, horizon: int = None) -> \
        Tuple[List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array]]:
        """Return cost matrices used by the controller for a cost of shape:
        x^T Q x + uv^T R uv + u^-v^T R^- uv + x^T M uv + L uv"""
        pass
            
    @abstractmethod
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time:int = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def _compute_base_cost_matrices(self, physical_model: CoarseModel = None):
        """Compute the base matrices used to form the cost matrices. If the cost
        is time invariant, those matrices are directly Q, R, R^-, M, L """
        pass