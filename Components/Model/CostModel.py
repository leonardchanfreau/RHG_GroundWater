from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from .CoarseModel import CoarseModel

class CostModel(ABC):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_file: str, physical_model: CoarseModel):
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
        # Dimensions parameters
        self.nx = physical_model.nx
        self.n_agent = physical_model.n_agent
        
        # Well positions
        self.well_to_cell = physical_model.well_to_cell
        self.input_v_to_state = [c2x @ u2c for c2x, u2c in
                                 zip(physical_model.command_v_to_cell,
                                     physical_model.input_v_to_command_v)]
        self.n_well_per_command = physical_model.n_well_per_command
        self.input_v_to_command_v = physical_model.input_v_to_command_v
        self.well_to_agent = physical_model.well_to_agent
        
        # Heads at the top of the wells
        self.top_head = physical_model.initial_state
                
        # Correct cost using corrected heads
        correction = params.get("correction")
        if (correction is not None) and (correction['correct_cost']):
            self.correct_cost = True
            self.cost_correction = correction.get("cost")
            if physical_model.correct_heads:
                self.heads_correction = physical_model.heads_correction
        
        # Compute cost matrices
        self._compute_base_cost_matrices(params['costs'], physical_model)
    
    
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
    def _compute_base_cost_matrices(self, cost_params: Dict[str, float], physical_model: CoarseModel = None):
        """Compute the base matrices used to form the cost matrices. If the cost
        is time invariant, those matrices are directly Q, R, R^-, M, L """
        pass