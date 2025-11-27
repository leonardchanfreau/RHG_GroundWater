from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from .utils.rain import get_available_water, spread_rain
from .IrrigationSchedulingCost import IrrigationSchedulingCost
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCostQuadraticDistance(IrrigationSchedulingCost):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_file: str, physical_model: IrrigationSchedulingModel,
                 crop_repartition: np.ndarray):
        super().__init__(param_file, physical_model)
        # Additional parameters
        with open(param_file, 'r') as file:
            params = json.load(file)
        self.yield_sensitivity = 1e3*np.array(params['costs']['yield_sensitivity'])  # [n_crops, n_agent]
    
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time: int = None) -> np.ndarray:
        """parameter u of shape [n_well, n_crops] contains negative values"""
        u = u.reshape((np.hstack(self.n_well_per_command).sum(dtype=int), self.n_crops))
        ET = self._get_water_requirement(time, 1)[:,:,0]
        Ry = self._get_under_irrigation_weight(time, 1)
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            well_indices = np.arange(np.hstack(self.n_well_per_command).sum(), dtype=int)[self.well_to_agent==v]
            pumping_cost = (-self.extraction_cost[v] * (self.top_head[input_indices] -\
                x[well_indices]).reshape(-1,1) * u[self.well_to_agent==v,:].sum(axis=1).reshape(-1,1)).sum()
            under_irrigation = ET[v, :] + u[self.well_to_agent==v,:].sum(axis=0)    # TODO: check dimensions
            under_irrigation_cost = \
                under_irrigation.T @ sp.diags(Ry[v,:].flatten()) @ under_irrigation.T
            cost[v] = pumping_cost + under_irrigation_cost
        return cost
        
    def _compute_irrigation_cost_matrices(self, time:int, horizon: int):
        # Get cost data
        Ry_v_c_t = self._get_under_irrigation_weight(time, horizon)     # [n_agent, n_crop, horizon]
        ET_v_t = self._get_water_requirement(time, horizon)             # [n_agent, n_crop, horizon]
        # Arrange data in square diagonal matrices of size n_crops for all v,t
        diag_Ry_v_c_t = [[sp.diags(Ry_c.flatten())
                        for Ry_c in np.split(Ry_v_c_t[v,:,:], horizon, axis=1)]
                       for v in range(self.n_agent)]
        # Form cost matrices
        self.Ri_v = [[(u2c.T @ Ry[t] @ u2c )for t in range(horizon)]
                     for u2c, Ry in
                     zip(self.input_to_crop_irrigation_v, diag_Ry_v_c_t)]
        self.Li_v = [[2 * ET_v_t[v,:,t].reshape(1,self.n_crops) @ Ry[t] @ u2c
                      for t  in range(horizon)]
                     for v, (u2c, Ry) in
                     enumerate(zip(self.input_to_crop_irrigation_v, diag_Ry_v_c_t))]
    
    def _construct_crop_yield_data(self, 
                                   crops_param: Dict[str, Dict[str, List[float]]]):
        self.Ky_c_t = []
        self.Ky_stages = []
        for crop in crops_param.values():
            Ky_stages = np.array(crop['Ky_stages'])
            Ky_c_t = np.array(crop['Ky_t'])
            # Shift to plant date
            if crop['plant_date'] != 0:
                Ky_stages = np.hstack([[0], Ky_stages + crop['plant_date']])
                Ky_c_t = np.hstack([[0], Ky_c_t])
            self.Ky_stages.append(Ky_stages)
            self.Ky_c_t.append(Ky_c_t)
        # Create vectorized version of _yield_coefficient
        self._yield_coefficient = np.vectorize(self._scalar_yield_coefficient, signature='(k)->(n,k)')
        
    def _get_under_irrigation_weight(self, time: int, horizon: int) -> np.ndarray:
        Ky_c_t = self._get_yield_coefficient(time, horizon)                 # [n_crop, horizon]
        ETmax_v_c_t = self._get_global_evapotranspiration(time, horizon)       # [n_agent, n_crop, horizon]
        inv_ETmax_v_c_t = np.where(ETmax_v_c_t !=0, 1/ETmax_v_c_t, 0)
        Ry_v_c_t = (((inv_ETmax_v_c_t * Ky_c_t)**2).T * self.yield_sensitivity).T # [n_agent, n_crop, horizon]
        return Ry_v_c_t
    
    def _get_yield_coefficient(self, time: int, horizon: int) -> np.ndarray:
        times = np.arange(time, time+horizon)
        return self._yield_coefficient(times)
    
    
    def _scalar_yield_coefficient(self, t:int) -> np.ndarray:
        indices = [np.maximum(np.searchsorted(self.Ky_stages[c], t+1)-1, 0)
                   for c in range(self.n_crops)]
        Ky = np.array([self.Ky_c_t[c][i] for c, i in enumerate(indices)])
        return Ky