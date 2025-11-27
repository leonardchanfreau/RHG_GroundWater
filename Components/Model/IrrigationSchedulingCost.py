from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from .utils.rain import get_available_water, spread_rain
from .CostModel import CostModel
from .IrrigationSchedulingModel import IrrigationSchedulingModel

class IrrigationSchedulingCost(CostModel):
    """
    Coarse model of the cost function.
    """
    
    def __init__(self, param_file: str, physical_model: IrrigationSchedulingModel,
                 crop_repartition: np.ndarray):
        # Additional dimension
        self.n_crops = physical_model.n_crops
        # init base
        super().__init__(param_file, physical_model)
        # Parameter from upper controller
        self.crop_repartition = crop_repartition    # dim [n_crops, n_agent]
        # Additional parameters
        with open(param_file, 'r') as file:
            params = json.load(file)
        self.land_surface_v = np.array(params['land']['surface'])
        self.ET0 = params['physical']['ET0']
        # Time-varying utility data
        self._construct_crop_yield_data(params['crops'])
        self._construct_max_water_demand_data(params['crops'])
        self._construct_available_water(params['rain'])
            
    
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
    
    
    def _compute_base_cost_matrices(self, cost_params: Dict[str, float], physical_model: IrrigationSchedulingModel = None) -> \
        Tuple[List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array], List[sp.csr_array]]:
        # Utils
        B_minus_v = [sp.hstack(physical_model.B_v[:v]+physical_model.B_v[v+1:])
                     for v in range(self.n_agent)]
        # Pumping cost
        self.extraction_cost = 10e-5 * np.array(cost_params['extraction_cost'])     # TODO: justification and avoid hardcoding (careful hardcoded)
        state_input = [extr_cost * self.input_v_to_state[v] for v, extr_cost
                        in enumerate(self.extraction_cost)]
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
                self.cost_correction = self.extraction_cost * self.heads_correction
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
    
    @abstractmethod
    def _construct_crop_yield_data(self, 
                                   crops_param: Dict[str, Dict[str, List[float]]]):
        """ Time-varying info about sensitivity. Ex: Yield coeff, curvature of
        yield-to-water response curve, etc. """
        pass
    
    def _construct_max_water_demand_data(self,
                                         crops_param: Dict[str, Dict[str, List[float]]]):
        """ Time-varying info about maximum water demand. Ex: max evapotranspiration"""
        self.ET_c_t = []
        self.Kc_stages = []
        for crop in crops_param.values():
            ET_c_t = np.array(crop['Kc_t'])*self.ET0
            self.Kc_stages.append(np.array(crop['Kc_stages']))
            # Shift to plant date
            Kc_stages = self.Kc_stages[-1] + crop['plant_date']
            if crop['plant_date'] != 0:
                Kc_stages = np.hstack([[0], Kc_stages[0]-1, Kc_stages])
                ET_c_t = np.hstack([[0], [0], ET_c_t])
            self.Kc_stages[-1] = np.hstack([Kc_stages, Kc_stages[-1]+1, [10000]])
            ET_c_t = np.hstack([ET_c_t, [0], [0]])
            self.ET_c_t.append(ET_c_t)
        # Create vectorized version of evapotranspiration
        self._local_evapotranspiration = np.vectorize(self._scalar_local_evapotranspiration,
                                                signature="() -> (n)")
    
    def _get_water_requirement(self, time:int, horizon:int) -> np.ndarray:
        ET_v_t = self._get_global_evapotranspiration(time, horizon)            # [n_agent, n_crop, horizon]
        rain_v_t = self._get_available_water(time, horizon)             # [n_agent, n_crop, horizon]
        requirement = ET_v_t - rain_v_t
        return np.maximum(requirement, 0.)
        
    
    def _get_available_water(self, time: int, horizon: int) -> np.ndarray:
        pad = np.zeros(self.available_water[...,0].shape + \
            (max(time+horizon - self.available_water.shape[-1],0),))
        return np.concatenate((self.available_water[...,time:time+horizon], pad), axis=-1)
    
    def _get_global_evapotranspiration(self, time: int, horizon: int) -> List[np.ndarray]:
        times = np.arange(time, time+horizon)
        local_ET = self._local_evapotranspiration(times).T                        # [n_crops, horizon] TODO:check shape
        crop_surface = self.land_surface_v[:, None] * self.crop_repartition.T     # [n_agent, n_crops] 
        return crop_surface[:, :, None] * local_ET[None, :, :]                    # [n_agent, n_crop, horizon]
    
    def _get_max_evapotranspiration(self, time: int, horizon: int) -> List[np.ndarray]:
        """ depreciated: do not used in code. """
        times = np.arange(time, time+horizon)
        local_ET = self._local_evapotranspiration(times).T # [n_crops, horizon] TODO:check shape
        crop_surface = self.land_surface_v[:, None] * np.ones_like(self.crop_repartition.T)# [n_agent, n_crops] 
        return crop_surface[:, :, None] * local_ET[None, :, :]                    # [n_agent, n_crop, horizon]
    
    def _scalar_local_evapotranspiration(self, t:int) -> np.ndarray:
        indices = [np.maximum(np.searchsorted(self.Kc_stages[c], t)-1, 0)
                   for c in range(self.n_crops)]
        ET = []
        for c, i in enumerate(indices):
            ET.append(self.ET_c_t[c][i] + \
                (self.ET_c_t[c][i+1]- self.ET_c_t[c][i])*\
                    ((t - self.Kc_stages[c][i]) / \
                        (self.Kc_stages[c][i+1] - self.Kc_stages[c][i])))
        return np.array(ET)