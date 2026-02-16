from typing import Tuple, Dict, List, Any
import json
import numpy as np
import scipy.sparse as sp

from .ParameterInterface import ParameterInterface

class PhysicalParameterInterface(ParameterInterface):
    
    def __init__(self, parameter_file: str):
        super().__init__(parameter_file)
        
        # Read parameter file
        with open(parameter_file, 'r') as file:
            params = json.load(file)
        
        # Physical parameters
        self.hk = params['physical']['hk']          # [m/d]
        self.sy = params['physical']['sy']          # [dimless]
        self.ss = params['physical']['ss']          # [1/d]
        if self.uniform_params:
            self.conductivity_k = sp.csr_array(
                self.hk * np.ones((self.n_col, self.n_row)))
            self.storativity_k = sp.csr_array(
                self.sy * np.ones((self.n_col, self.n_row)))
        else:
            self.conductivity_k =  sp.csr_array(self._average_value(self.hk))
            self.storativity_k = sp.csr_array(self._average_value(self.sy))
        # Flatten array
        self.conductivity_k = self.conductivity_k.reshape((self.nx, 1)).tocsr()
        self.storativity_k = self.storativity_k.reshape((self.nx, 1)).tocsr()
        
        # Set dynamic parameters
        self._compute_dynamics_parameters()
        
        # Boundary
        self.boundary_head = self.boundary_head
        
        # Cost coefficients
        self.extraction_cost     = np.array(params["costs"]["extraction_cost"])
        self.selling_crop_price  = np.array(params["costs"]["selling_crop_price"])
    
    
    def get_constraints_parameters(self):
        return self.well_depth, self.min_head, self.max_pumping_rates
    
    def get_initial_state(self):
        return self.init_heads.reshape((self.nx, 1)) \
            if type(self.init_heads) is np.ndarray else \
                self.init_heads * np.ones((self.nx, 1))
                
    def get_cost_coefficients(self):
        return 10e-5 * self.extraction_cost, self.selling_crop_price            # TODO: justification and avoid hardcoding (careful hardcoded)
    
    def get_physical_parameters(self):
        return self.hk, self.sy, self.ss
    
    def _compute_dynamics_parameters(self):
        width = self.ztop - self.zbot
        non_zero = self.distance_kl.nonzero()
        self.a = sp.lil_array(self.distance_kl.shape)
        self.a[non_zero] = (self.cross_section_kl[non_zero] / self.distance_kl[non_zero])
        self.a *= self.dt * (self.conductivity_k / (self.surface_k * self.storativity_k)) # * width))
        
        self.b = self.dt*sp.csr_array((np.ones((self.nx,1)) / \
            (self.surface_k * self.storativity_k ).toarray())) #* width
        
        self.c = self.dt * self.recharge_k / (self.surface_k * self.storativity_k) # * width)
    
    def _compute_yield_function_parameters(self):
        if self.yield_function_type == "quadratic":
            return self._compute_quadratic_yield_function_parameters()
        elif self.yield_function_type == "quadratic_distance":
            return self._compute_quadratic_distance_yield_function_parameters()
        return self._compute_linear_yield_function_parameters()
        
    def _compute_quadratic_yield_function_parameters(self):
        # Safely inverse values
        inv_ETmax_c_t = np.where(self.ETmax_c_t !=0, 1/self.ETmax_c_t, 0)
        inv_quad_ETmax_c_t = np.where(self.ETmax_c_t !=0, 1/(self.ETmax_c_t**2), 0)
        # Construct local coefficients
        a_yield_v_c_t = - self.Ymax_c_t * inv_quad_ETmax_c_t * \
            (1 - self.Y_q_c_t)/ \
                ((1 - self.intermediate_irrigation_value)**2)
        b_yield_v_c_t = 2* self.Ymax_c_t * inv_ETmax_c_t * \
            (1 - self.Y_q_c_t)/ \
                ((1 - self.intermediate_irrigation_value)**2)
        c_yield_v_c_t = self.Ymax_c_t * (1 - \
            (1 - self.Y_q_c_t)/ \
                ((1 - self.intermediate_irrigation_value)**2))
        # Scale to obtain global coefficients
        inv_crop_surface = np.divide(np.ones_like(self.crop_surface), self.crop_surface,
                                     out=np.zeros_like(self.crop_surface),
                                     where=self.crop_surface!=0)                # [n_agent, n_crops]
        null_surface_mask = np.where(self.crop_surface !=0, 1, 0)               # [n_agent, n_crops]
        a_yield_v_c_t = inv_crop_surface[:, :, None] * a_yield_v_c_t[None, :, :]
        b_yield_v_c_t = null_surface_mask[:, :, None] * b_yield_v_c_t[None, :, :]
        c_yield_v_c_t = self.crop_surface[:, :, None] * c_yield_v_c_t[None, :, :]
        return a_yield_v_c_t, b_yield_v_c_t, c_yield_v_c_t                      # [n_agent, n_crops, horizon]
    
    def _compute_linear_yield_function_parameters(self):
        # Safely inverse values
        inv_ETmax_c_t = np.where(self.ETmax_c_t !=0, 1/self.ETmax_c_t, 0)
        # Construct slope
        k_yield_v_c_t = self.Ymax_c_t * inv_ETmax_c_t                           # [n_crop, horizon]
        # Extend it to all players
        null_surface_mask = np.where(self.crop_surface !=0, 1, 0)               # [n_agent, n_crops]
        k_yield_v_c_t = null_surface_mask[:, :, None] * k_yield_v_c_t[None, :, :]
        return None, k_yield_v_c_t, None                                        # [n_agent, n_crop, horizon]

    def _compute_quadratic_distance_yield_function_parameters(self):
        # Water Requirement
        ETmax_v_c_t = self.crop_surface[:, :, None] * self.ETmax_c_t[None, :, :]       # [n_agent, n_crop, horizon]
        # Under-irrigation weight
        inv_ETmax_v_c_t = np.where(ETmax_v_c_t !=0, 1/ETmax_v_c_t, 0)
        Ry_v_c_t = (((inv_ETmax_v_c_t * self.Ky_c_t)**2).T * self.yield_sensitivity).T # [n_agent, n_crop, horizon]
        return Ry_v_c_t, ETmax_v_c_t, None