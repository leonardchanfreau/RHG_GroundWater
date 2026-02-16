from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from ..ParameterInterface import ParameterInterface
from .utils.rain import get_available_water, spread_rain
from .IrrigationSchedulingCost import IrrigationSchedulingCost
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCostQuadraticDistance(IrrigationSchedulingCost):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_handler: ParameterInterface, physical_model: IrrigationSchedulingModel):
        super().__init__(param_handler, physical_model)
        # Rain data
        self._construct_available_water(param_handler.rain)
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time: int = None) -> np.ndarray:
        """parameter u of shape [n_well, n_crops] contains negative values"""
        u = u.reshape((np.hstack(self.n_well_per_command).sum(dtype=int), self.n_crops))
        ET = self._get_water_requirement(time, 1)[:,:,0]
        Ry = self.a_yield_v_c_t[:, :, time]
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            well_indices = np.arange(np.hstack(self.n_well_per_command).sum(), dtype=int)[self.well_to_agent==v]
            pumping_cost = (-self.extraction_cost_v[v] * (self.top_head[input_indices] -\
                x[well_indices]).reshape(-1,1) * u[self.well_to_agent==v,:].sum(axis=1).reshape(-1,1)).sum()
            under_irrigation = ET[v, :] + u[self.well_to_agent==v,:].sum(axis=0)    # TODO: check dimensions
            under_irrigation_cost = \
                under_irrigation.T @ sp.diags(Ry[v,:].flatten()) @ under_irrigation.T
            cost[v] = pumping_cost + under_irrigation_cost
        return cost
        
    def _compute_irrigation_cost_matrices(self, time:int, horizon: int):
        times = np.arange(horizon) + time
        # Get cost data
        Ry_v_c_t = self.a_yield_v_c_t[:, :, times]                  # [n_agent, n_crop, horizon]
        ET_v_t = self._get_water_requirement(time, horizon)         # [n_agent, n_crop, horizon]
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
    
    
    def _construct_available_water(self, 
                                   rain_params: Dict[str, Any]):
        self.available_water = get_available_water(
            rain_params.get('file'), rain_params.get('year'),
            rain_params.get('state'), rain_params.get('county'))
        self.available_water = np.kron(
            (np.diag(self.land_surface_v) @ self.crop_repartition.T)[:,:,np.newaxis],
            self.available_water[np.newaxis, :])
        if bool(rain_params.get('spread_rain')):
            storage_coefficient = rain_params.get('storage_coefficient')
            water_requirement_prediction = self._get_global_evapotranspiration(0, self.available_water.shape[-1])
            extra_rain_pred = spread_rain(
                water_requirement_prediction - self.available_water,
                storage_coefficient)
            self.available_water = water_requirement_prediction - extra_rain_pred
    
    def _get_water_requirement(self, time:int, horizon:int) -> np.ndarray:
        times = np.arange(horizon) + time
        ET_v_t = self.b_yield_v_c_t[:, :, times]                    # [n_agent, n_crop, horizon]
        rain_v_t = self._get_available_water(time, horizon)         # [n_agent, n_crop, horizon]
        requirement = ET_v_t - rain_v_t
        return np.maximum(requirement, 0.)
    
    def _get_available_water(self, time: int, horizon: int) -> np.ndarray:
        pad = np.zeros(self.available_water[...,0].shape + \
            (max(time+horizon - self.available_water.shape[-1],0),))
        return np.concatenate((self.available_water[...,time:time+horizon], pad), axis=-1)