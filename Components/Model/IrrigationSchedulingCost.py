from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from ..ParameterInterface import ParameterInterface
from .utils.rain import get_available_water, spread_rain
from .CostModel import CostModel
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCost(CostModel):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_handler: ParameterInterface, physical_model: IrrigationSchedulingModel):
        # Additional dimension
        self.n_crops = physical_model.n_crops
        # init base
        super().__init__(param_handler, physical_model)
        # Time-varying yield profit parameters
        self.a_yield_v_c_t, self.b_yield_v_c_t, self.c_yield_v_c_t = \
            param_handler.get_yield_profit_parameters()
            
    
    def get_cost_matrices(self, time: int = None, horizon: int = None) -> \
        Tuple[List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array]]:
        """Return cost matrices used by the controller for a cost of shape:
        x^T Q x + uv^T R uv + u^-v^T R^- uv + x^T M uv + L uv"""
        self._compute_irrigation_cost_matrices(time, horizon)
        # Add regularization to R_v to ensure non-singularity
        reg = [1e-10 * np.eye(self.Rp_v[v].shape[0])
               for v in range(self.n_agent)]
        # Form matrices
        self.Q = None
        self.R_v = [[self.Rp_v[v] + self.Ri_v[v][t] + reg[v] for t in range(horizon)]
                    for v in range(self.n_agent)]
        self.R_v_minus = [[self.Rp_v_minus[v] for t in range(horizon)]
                          for v in range(self.n_agent)]
        self.M_v = [[self.Mp_v[v] for t in range(horizon)]
                    for v in range(self.n_agent)]
        self.L_v = [[self.Lp_v[v] + self.Li_v[v][t] for t in range(horizon)]
                    for v in range(self.n_agent)]
        return self.Q, self.R_v, self.R_v_minus, self.M_v, self.L_v
    
    
    @abstractmethod
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time: int = None) -> np.ndarray:
        """parameter u of shape [n_well, n_crops] contains negative values"""
        pass
    
    
    def _compute_base_cost_matrices(self, physical_model: IrrigationSchedulingModel = None) -> \
        Tuple[List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array]]:
        # Utils
        B_minus_v = [sp.hstack(physical_model.B_v[:v]+physical_model.B_v[v+1:])
                     for v in range(self.n_agent)]
        # Pumping cost
        state_input = [extr_cost * self.input_v_to_state[v] for v, extr_cost
                        in enumerate(self.extraction_cost_v)]
        self.Rp_v = [B.T @ xu
                 for B, xu in zip(physical_model.B_v, state_input)]
        self.Rp_v_minus = [B.T @ xu for B, xu in zip(B_minus_v, state_input)]
        self.Mp_v = [physical_model.A.T @ xu for xu in state_input]
        self.Lp_v = [(physical_model.C - self.top_head).T @ xu
                     for xu in state_input]
        # Correct cost if needed
        if self.correct_cost:
            # Use in-well heads to compute cost
            if self.heads_correction is not None:
                # Reverberate correction on cost
                self.cost_correction = self.extraction_cost_v * self.heads_correction
            correction = [-c * (u2c.T @ np.diag(n_w_p_i) @ u2c)
                          for c, n_w_p_i, u2c in
                          zip(self.cost_correction, self.n_well_per_command, self.input_v_to_command_v)]
            self.Rp_v = [R + c for R, c in zip(self.Rp_v, correction)]
        # Under-irrigation cost utility matrices
        self.input_to_crop_irrigation_v = [
            np.kron(np.ones((1,n_cmd)), np.eye(self.n_crops)) for n_cmd in 
            physical_model.n_command_v]
        
    @abstractmethod
    def _compute_irrigation_cost_matrices(self, time:int, horizon: int):
        pass