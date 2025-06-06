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
            
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            cost[v] = (-self.extraction_cost[v] * (self.top_head[input_indices] -\
                x[input_indices]) * u[self.well_to_agent==v] + \
                    self.production_profit[v] * u[self.well_to_agent==v]).sum()
        return cost
    
    
    def _compute_cost_matrices(self, cost_params: Dict[str, float], physical_model: CoarseModel = None) -> \
        Tuple[List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array]]:
        # Store cost parameters
        self.extraction_cost = np.array(cost_params['extraction_cost'])
        self.production_profit = np.array(cost_params['production_profit'])
        # State-input cross term
        state_input = [extr_cost * self.input_to_state_v[v] for v, extr_cost
                        in enumerate(self.extraction_cost)]
        # Linear input cost
        nu_v = [el.shape[1] for el in self.input_to_state_v]
        linear_input = [profit * np.ones((1,nu_v[v])) - \
            (self.extraction_cost[v] * self.top_head[np.nonzero(self.input_to_state_v[v].T)[1]]).T
                     for v, profit in enumerate(self.production_profit)]
        # Define the cost matrices depending on the options
        look_ahead = cost_params.get('look_ahead')
        if (look_ahead is None) or (look_ahead==False):
            # Pumping cost is computed using head before pumping
            return None, None, None, state_input, linear_input
        # Pumping cost is computed using head after pumping
        R_v = [B.T @ G for B, G in zip(physical_model.B_v, state_input)]
        B_minus_v = [sp.hstack(physical_model.B_v[:v]+physical_model.B_v[v+1:])
                     for v in range(self.n_agent)]
        R_v_minus = [B.T @ G for B, G in zip(B_minus_v, state_input)]
        M_v = [physical_model.A.T @ G for G in state_input]
        L_v = [physical_model.C.T @ G + Xi for G, Xi in zip(state_input, linear_input)]
        if self.correct_cost:
            # Use in-well heads to compute cost
            if self.heads_correction is not None:
                # Reverberate correction on cost
                self.cost_correction = self.extraction_cost * self.heads_correction
            n_well_per_input = [self.input_to_well[self.well_to_agent==v, :].sum(axis=0) for v in range(self.n_agent)]
            n_well_per_input = [n_w_p_i[n_w_p_i>0] for n_w_p_i in n_well_per_input]
            correction = [-c*np.diag(n_w_p_i) for c, n_w_p_i in zip(self.cost_correction, n_well_per_input)]
            R_v = [R + c for R, c in zip(R_v, correction)]
        return None, R_v, R_v_minus, M_v, L_v