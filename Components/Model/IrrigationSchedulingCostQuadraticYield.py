from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from .utils.rain import get_available_water, spread_rain
from .IrrigationSchedulingCost import IrrigationSchedulingCost
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCostQuadraticYield(IrrigationSchedulingCost):
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
        self.intermediate_irrigation_value = np.array(params['costs']['intermediate_point'])
    
    
    def compute_cost(self, x: np.ndarray, u: np.ndarray, time: int = None) -> np.ndarray:
        """parameter u of shape [n_well, n_crops] contains negative values"""
        u = u.reshape((np.hstack(self.n_well_per_command).sum(dtype=int), self.n_crops))
        a_yield_v_c, b_yield_v_c, c_yield_v_c = self._get_yield_response_to_water_coefficients(time, 1)
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            well_indices = np.arange(np.hstack(self.n_well_per_command).sum(), dtype=int)[self.well_to_agent==v]
            pumping_cost = (-self.extraction_cost[v] * (self.top_head[input_indices] -\
                x[well_indices]).reshape(-1,1) * u[self.well_to_agent==v,:].sum(axis=1).reshape(-1,1)).sum()
            irrigation = u[self.well_to_agent==v,:].sum(axis=0).reshape(-1,1)
            irrigation_income = irrigation.T @ np.diag(self.selling_crop_price) @ np.diag(a_yield_v_c[v,:,0]) @ irrigation - \
                b_yield_v_c[v,:,:].T @ np.diag(self.selling_crop_price) @ irrigation + (np.diag(self.selling_crop_price) @ c_yield_v_c[v,:,:]).sum() #TODO:check result
            cost[v] = pumping_cost - irrigation_income
        return cost
        
    def _compute_irrigation_cost_matrices(self, time:int, horizon: int):
        # Get cost data
        a_yield_v_c_t, b_yield_v_c_t, _ = self._get_yield_response_to_water_coefficients(time, horizon)    # [n_agent, n_crop, horizon]
        # Arrange a in square diagonal matrices of size n_crops for all v,t
        diag_a_yield_v_c_t = [[sp.diags(a_yield_c.flatten())
                        for a_yield_c in np.split(a_yield_v_c_t[v,:,:], horizon, axis=1)]
                       for v in range(self.n_agent)]
        # Arrange b in flat matrices of size n_crops for all v,t
        diag_b_yield_v_c_t = [[b_yield_c.flatten().reshape(1, self.n_crops)
                        for b_yield_c in np.split(b_yield_v_c_t[v,:,:], horizon, axis=1)]
                       for v in range(self.n_agent)]
        # Form cost matrices
        self.Ri_v = [[- (u2c.T @ np.diag(self.selling_crop_price) @ a_yield[t] @ u2c) for t in range(horizon)]
                     for u2c, a_yield in
                     zip(self.input_to_crop_irrigation_v, diag_a_yield_v_c_t)]
        self.Li_v = [[b_yield[t] @ np.diag(self.selling_crop_price) @ u2c for t  in range(horizon)]
                     for u2c, b_yield in
                     zip(self.input_to_crop_irrigation_v, diag_b_yield_v_c_t)]
    
    def _construct_crop_yield_data(self, 
                                   crops_param: Dict[str, Dict[str, List[float]]]):
        """ Time-varying info about sensitivity. Ex: Yield coeff, curvature of
        yield-to-water response curve, etc. """
        self.Y_max_c_t = []
        self.Y_intermediate_c_t = []
        self.curvature_stages = []
        for crop in crops_param.values():
            Y_max_c_t  = np.array(crop['Y_max_t'])
            Y_intermediate_c_t = np.array(crop['Y_intermediate_t'])
            curvature_stages = np.array(crop['curvature_stages'])
            # Shift to plant date
            if crop['plant_date'] != 0:
                Y_max_c_t = np.hstack([[0], [0], Y_max_c_t])
                Y_intermediate_c_t = np.hstack([[0], Y_intermediate_c_t])
                curvature_stages = np.hstack([[0], curvature_stages + crop['plant_date']])
            self.Y_max_c_t.append(np.hstack([Y_max_c_t, [0], [0]]))
            self.Y_intermediate_c_t.append(Y_intermediate_c_t)
            self.curvature_stages.append(curvature_stages)
        # Create vectorized version of _yield_coefficient
        self._yield_max = np.vectorize(self._scalar_yield_max, signature='(k)->(n,k)')
        self._yield_intermediate = np.vectorize(self._scalar_yield_intermediate, signature='(k)->(n,k)')
        
    def _get_yield_response_to_water_coefficients(self, time: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Coefficients of the quadratic function. """
        # Get yield data
        times = np.arange(time, time+horizon)
        Ymax_v_c_t = self._yield_max(times)             # [n_crop, horizon]
        ETmax_v_c_t = self._local_evapotranspiration(
            times).T             # [n_crop, horizon]
        Y_q_v_c_t = self._yield_intermediate(times)     # [n_crop, horizon]
        # Safely inverse values
        inv_ETmax_v_c_t = np.where(ETmax_v_c_t !=0, 1/ETmax_v_c_t, 0)
        inv_quad_ETmax_v_c_t = np.where(ETmax_v_c_t !=0, 1/(ETmax_v_c_t**2), 0)
        # Construct local coefficients
        a_yield_v_c_t = - Ymax_v_c_t * inv_quad_ETmax_v_c_t * \
            (1 - Y_q_v_c_t)/ \
                ((1 - self.intermediate_irrigation_value)**2)
        b_yield_v_c_t = 2* Ymax_v_c_t * inv_ETmax_v_c_t * \
            (1 - Y_q_v_c_t)/ \
                ((1 - self.intermediate_irrigation_value)**2)
        c_yield_v_c_t = Ymax_v_c_t * (1 - \
            (1 - Y_q_v_c_t)/ \
                ((1 - self.intermediate_irrigation_value)**2))
        # Scale to obtain global coefficients
        crop_surface = self.land_surface_v[:, None] * self.crop_repartition.T   # [n_agent, n_crops]
        inv_crop_surface = np.divide(np.ones_like(crop_surface), crop_surface,
                                     out=np.zeros_like(crop_surface),
                                     where=crop_surface!=0)                     # [n_agent, n_crops]
        null_surface_mask = np.where(crop_surface !=0, 1, 0)                    # [n_agent, n_crops]
        a_yield_v_c_t = inv_crop_surface[:, :, None] * a_yield_v_c_t[None, :, :]
        b_yield_v_c_t = null_surface_mask[:, :, None] * b_yield_v_c_t[None, :, :]
        c_yield_v_c_t = crop_surface[:, :, None] * c_yield_v_c_t[None, :, :]
        return a_yield_v_c_t, b_yield_v_c_t, c_yield_v_c_t                      # [n_agent, n_crops, horizon]
    
    def _get_max_global_yield(self, time: int, horizon: int) -> np.ndarray:
        times = np.arange(time, time+horizon)
        local_y_max = self._yield_max(times)                                    # [n_crops, horizon]
        crop_surface = self.land_surface_v[:, None] * self.crop_repartition.T   # [n_agent, n_crops] 
        return crop_surface[:, :, None] * local_y_max[None, :, :]               # [n_agent, n_crop, horizon]
    
    def _get_global_yield_at_irrigation_q(self, time: int, horizon: int) -> np.ndarray:
        times = np.arange(time, time+horizon)
        local_y_intermediate = self._yield_intermediate(times)                  # [n_crops, horizon]
        crop_surface = self.land_surface_v[:, None] * self.crop_repartition.T   # [n_agent, n_crops] 
        return crop_surface[:, :, None] * local_y_intermediate[None, :, :]      # [n_agent, n_crop, horizon]
    
    def _scalar_yield_max(self, t:int) -> np.ndarray:
        indices = [np.maximum(np.searchsorted(self.Kc_stages[c], t+1)-1, 0)
                   for c in range(self.n_crops)]
        Ky = np.array([self.Y_max_c_t[c][i] for c, i in enumerate(indices)])
        return Ky
    
    def _scalar_yield_intermediate(self, t:int) -> np.ndarray:
        indices = [np.maximum(np.searchsorted(self.curvature_stages[c], t+1)-1, 0)
                   for c in range(self.n_crops)]
        Ky = np.array([self.Y_intermediate_c_t[c][i] for c, i in enumerate(indices)])
        return Ky