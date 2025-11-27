from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from .utils.rain import get_available_water, spread_rain
from .IrrigationSchedulingCost import IrrigationSchedulingCost
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCostLinearYield(IrrigationSchedulingCost):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_file: str, physical_model: IrrigationSchedulingModel,
                 crop_repartition: np.ndarray):
        super().__init__(param_file, physical_model, crop_repartition)
        # Additional parameters
        with open(param_file, 'r') as file:
            params = json.load(file)
        self.selling_crop_price = np.array(params['costs']['selling_crop_price'])  # [n_crops,]
    
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time: int = None) -> np.ndarray:
        """parameter u of shape [n_well, n_crops] contains negative values"""
        u = u.reshape((np.hstack(self.n_well_per_command).sum(dtype=int), self.n_crops))
        k_yield_v_c = self._get_yield_response_to_water_coefficient(time, 1)
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            well_indices = np.arange(np.hstack(self.n_well_per_command).sum(), dtype=int)[self.well_to_agent==v]
            pumping_cost = (-self.extraction_cost[v] * (self.top_head[input_indices] -\
                x[well_indices]).reshape(-1,1) * u[self.well_to_agent==v,:].sum(axis=1).reshape(-1,1)).sum()
            irrigation = u[self.well_to_agent==v,:].sum(axis=0).reshape(-1,1)
            irrigation_income = k_yield_v_c[v,:,:].T @ irrigation
            cost[v] = pumping_cost - irrigation_income
        return cost
        
    def _compute_irrigation_cost_matrices(self, time:int, horizon: int):
        # Get cost data
        k_yield_v_c_t = self._get_yield_response_to_water_coefficient(time, horizon)    # [n_agent, n_crop, horizon]
        # Arrange k in flat matrices of size n_crops for all v,t
        diag_k_yield_v_c_t = [[k_yield_c.flatten().reshape(1, self.n_crops)
                        for k_yield_c in np.split(k_yield_v_c_t[v,:,:], horizon, axis=1)]
                       for v in range(self.n_agent)]
        # Form cost matrices
        self.Ri_v = [[np.zeros((u2c.shape[1], u2c.shape[1])) for t in range(horizon)]
                     for u2c in self.input_to_crop_irrigation_v]
        self.Li_v = [[- k_yield[t] @ np.diag(self.selling_crop_price) @ u2c for t  in range(horizon)]
                     for u2c, k_yield in
                     zip(self.input_to_crop_irrigation_v, diag_k_yield_v_c_t)]
    
    def _construct_crop_yield_data(self, 
                                   crops_param: Dict[str, Dict[str, List[float]]]):
        """ Time-varying info about sensitivity. Ex: Yield coeff, curvature of
        yield-to-water response curve, etc. """
        self.Y_max_c_t = []
        for crop in crops_param.values():
            Y_max_c_t = np.array(crop['Y_max_t'])
            # Shift to plant date
            if crop['plant_date'] != 0:
                Y_max_c_t = np.hstack([[0], Y_max_c_t])
            self.Y_max_c_t.append(Y_max_c_t)
        # Create vectorized version of _yield_coefficient
        self._yield_max = np.vectorize(self._scalar_yield_max, signature='(k)->(n,k)')
        
    def _get_yield_response_to_water_coefficient(self, time: int, horizon: int) -> np.ndarray:
        """ slope of the linear function. """
        # Get yield data
        times = np.arange(time, time+horizon)
        Ymax_v_c_t = self._yield_max(times)                         # [n_crop, horizon]
        ETmax_v_c_t = self._local_evapotranspiration(times).T       # [n_crop, horizon]
        # Safely inverse values
        inv_ETmax_v_c_t = np.where(ETmax_v_c_t !=0, 1/ETmax_v_c_t, 0)
        # Construct slope
        k_yield_v_c_t = Ymax_v_c_t * inv_ETmax_v_c_t    # [n_crop, horizon]
        # Extend it to all players
        crop_surface = self.land_surface_v[:, None] * self.crop_repartition.T # [n_agent, n_crops]
        null_surface_mask = np.where(crop_surface !=0, 1, 0)                  # [n_agent, n_crops]
        k_yield_v_c_t = null_surface_mask[:, :, None] * k_yield_v_c_t[None, :, :]
        return k_yield_v_c_t                            # [n_agent, n_crop, horizon]
    
    def _get_max_global_yield(self, time: int, horizon: int) -> np.ndarray:
        times = np.arange(time, time+horizon)
        local_y_max = self._yield_max(times)                                    # [n_crops, horizon]
        crop_surface = self.land_surface_v[:, None] * self.crop_repartition.T   # [n_agent, n_crops]
        return crop_surface[:, :, None] * local_y_max[None, :, :]               # [n_agent, n_crop, horizon]
    
    def _scalar_yield_max(self, t:int) -> np.ndarray:
        indices = [np.maximum(np.searchsorted(self.Kc_stages[c], t+1)-1, 0)
                   for c in range(self.n_crops)]
        Ky = np.array([self.Y_max_c_t[c][i] for c, i in enumerate(indices)])
        return Ky