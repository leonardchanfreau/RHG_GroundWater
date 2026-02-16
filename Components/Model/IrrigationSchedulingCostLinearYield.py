from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from ..ParameterInterface import ParameterInterface
from .IrrigationSchedulingCost import IrrigationSchedulingCost
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCostLinearYield(IrrigationSchedulingCost):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_handler: ParameterInterface, physical_model: IrrigationSchedulingModel):
        super().__init__(param_handler, physical_model)
    
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time: int = None) -> np.ndarray:
        """parameter u of shape [n_well, n_crops] contains negative values"""
        u = u.reshape((np.hstack(self.n_well_per_command).sum(dtype=int), self.n_crops))
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            well_indices = np.arange(np.hstack(self.n_well_per_command).sum(), dtype=int)[self.well_to_agent==v]
            pumping_cost = (-self.extraction_cost_v[v] * (self.top_head[input_indices] -\
                x[well_indices]).reshape(-1,1) * u[self.well_to_agent==v,:].sum(axis=1).reshape(-1,1)).sum()
            irrigation = u[self.well_to_agent==v,:].sum(axis=0).reshape(-1,1)
            irrigation_income = self.b_yield_v_c_t[v,:,time].T @ np.diag(self.selling_crop_price_c) @ irrigation
            cost[v] = pumping_cost - irrigation_income
        return cost
        
    def _compute_irrigation_cost_matrices(self, time:int, horizon: int):
        times = np.arange(horizon) + time
        # Arrange k in flat matrices of size n_crops for all v,t
        diag_k_yield_v_c_t = [[k_yield_c.flatten().reshape(1, self.n_crops)
                        for k_yield_c in np.split(self.b_yield_v_c_t[v,:,times], horizon, axis=1)]
                       for v in range(self.n_agent)]
        # Form cost matrices
        self.Ri_v = [[np.zeros((u2c.shape[1], u2c.shape[1])) for t in range(horizon)]
                     for u2c in self.input_to_crop_irrigation_v]
        self.Li_v = [[k_yield[t] @ np.diag(self.selling_crop_price_c) @ u2c for t  in range(horizon)]
                     for u2c, k_yield in
                     zip(self.input_to_crop_irrigation_v, diag_k_yield_v_c_t)]