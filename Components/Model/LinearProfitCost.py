from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from .CostModel import CostModel, CoarseModel

class LinearProfitCost(CostModel):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_file: str, physical_model: CoarseModel):
        super().__init__(param_file, physical_model)
            
    
    def get_cost_matrices(self, time: int = None, horizon: int = None) -> \
        Tuple[List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array]]:
        """Return cost matrices used by the controller for a cost of shape:
        x^T Q x + uv^T R uv + u^-v^T R^- uv + x^T M uv + L uv"""
        return self.Q, self.R_v, self.R_v_minus, self.M_v, self.L_v
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time:int = None) -> np.ndarray:
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            cost[v] = (-self.extraction_cost[v] * (self.top_head[input_indices] -\
                x[input_indices]) * u[self.well_to_agent==v] + \
                    self.production_profit[v] * u[self.well_to_agent==v]).sum()
        return cost
    
    def _compute_base_cost_matrices(self, cost_params: Dict[str, float], physical_model: CoarseModel = None):
        # Store cost parameters
        self.extraction_cost = np.array(cost_params['extraction_cost'])
        self.production_profit = np.array(cost_params['production_profit'])
        # State-input cross term
        state_input = [extr_cost * self.input_v_to_state[v] for v, extr_cost
                        in enumerate(self.extraction_cost)]
        # Linear input cost
        nu_v = [el.shape[1] for el in self.input_v_to_state]
        linear_input = [profit * np.ones((1,nu_v[v])) - \
            (self.extraction_cost[v] * self.top_head[np.nonzero(self.input_v_to_state[v].T)[1]]).T
                     for v, profit in enumerate(self.production_profit)]
        # Define the cost matrices depending on the options
        look_ahead = cost_params.get('look_ahead')
        if (look_ahead is None) or (look_ahead==False):
            # Pumping cost is computed using head before pumping
            return None, None, None, state_input, linear_input
        # Pumping cost is computed using head after pumping
        self.R_v = [B.T @ G for B, G in zip(physical_model.B_v, state_input)]
        B_minus_v = [sp.hstack(physical_model.B_v[:v]+physical_model.B_v[v+1:])
                     for v in range(self.n_agent)]
        self.R_v_minus = [B.T @ G for B, G in zip(B_minus_v, state_input)]
        self.M_v = [physical_model.A.T @ G for G in state_input]
        self.L_v = [physical_model.C.T @ G + Xi for G, Xi in zip(state_input, linear_input)]
        if self.correct_cost:
            # Use in-well heads to compute cost
            if self.heads_correction is not None:
                # Reverberate correction on cost
                self.cost_correction = self.extraction_cost * self.heads_correction
            correction = [-c * (u2c.T @ np.diag(n_w_p_i) @ u2c)
                          for c, n_w_p_i, u2c in
                          zip(self.cost_correction, self.n_well_per_command, self.input_v_to_command_v)]
            self.R_v = [R + c for R, c in zip(self.R_v, correction)]
        # Other cost matrices
        self.Q = None