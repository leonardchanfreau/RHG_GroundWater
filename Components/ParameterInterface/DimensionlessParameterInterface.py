from typing import Tuple, Dict, List, Any
import math
import json
import numpy as np
import scipy.sparse as sp

from .ParameterInterface import ParameterInterface

class DimensionlessParameterInterface(ParameterInterface):
    
    def __init__(self, parameter_file: str):
        super().__init__(parameter_file)
        
        # Read parameter file
        with open(parameter_file, 'r') as file:
            params = json.load(file)
        
        # TODO: if using recharge either nondimensionalize or use relative rates in setup file
        
        # Reference dimensions
        self.R_p = params['dimensions']['R_p']
        self.tau_p = params['dimensions']['tau_p']
        self.k_yield_0 = params['crops']['corn']['Y_max_t'][0]/ \
            (params['crops']['corn']['Kc_t'][0]*params['physical']['ET0'])
        self.economic_sensitivity = params['dimensions']['economic_sensitivity']
        self.depth_p = self.economic_sensitivity *\
            params['costs']['selling_crop_price'][0]*self.k_yield_0 / params['costs']['extraction_cost'][0]
        self.Q_ref = params['dimensions']['reference_pumping']
        # Relative sampling parameters
        self.r_dx = (params['grid']['Lx']/params['coarse_model']['n_row'])/self.R_p
        self.r_dt = params['time_step']['dt']/self.tau_p
        # Relative physical parameters
        self.r_transmitivity = -(1/(2*math.pi))*math.log(2*self.R_p/params['grid']['Lx'])
        self.r_storativity = -((1.5**2)/(2*math.pi))*math.log(2*self.R_p/params['grid']['Lx'])
        
        # Set dynamic parameters
        self._compute_dynamics_parameters()
        
        # Boundary
        self.boundary_head = self.boundary_head / self.depth_p
    
    
    def get_constraints_parameters(self):
        return self.well_depth/self.depth_p, \
            self.min_head/self.depth_p, \
            self.max_pumping_rates/self.Q_ref
    
    def get_initial_state(self):
        init_state = self.init_heads.reshape((self.nx, 1)) \
            if type(self.init_heads) is np.ndarray else \
                self.init_heads * np.ones((self.nx, 1))
        return init_state/self.depth_p
                
    def get_cost_coefficients(self):
        return self.economic_sensitivity * np.ones((self.n_agent,)), \
            (1/self.k_yield_0) * np.ones((self.n_crops,))    # TODO check sign of cost matrices
            
    def get_physical_parameters(self):
        depth = self.ztop - self.zbot
        conductivity = (self.r_transmitivity*self.Q_ref/(self.depth_p))/depth
        storativity = self.r_storativity*self.Q_ref*self.tau_p/((self.R_p**2)*self.depth_p)
        specific_storage = 0.001 * (storativity/depth)      # TODO: check rigor
        return conductivity, storativity, specific_storage
    
    
    def _compute_dynamics_parameters(self):
        non_zero = self.adjacency.nonzero()
        self.a = sp.lil_array(self.adjacency.shape)
        self.a[non_zero] = (self.r_dt / ((1.5 * self.r_dx)**2))
        
        self.b = self.r_dt / (self.r_storativity * self.r_dx**2)
        
        self.c = (self.r_dt * self.recharge_k / (self.r_storativity * self.r_dx**2)).tocsr()
    
    def _compute_yield_function_parameters(self):
        # if self.yield_function_type == "quadratic":
        #     return self._compute_quadratic_yield_function_parameters()
        return self._compute_linear_yield_function_parameters()
        
    def _compute_quadratic_yield_function_parameters(self):
        pass
    
    def _compute_linear_yield_function_parameters(self):
        # Safely inverse values
        inv_ETmax_c_t = np.where(self.ETmax_c_t !=0, 1/self.ETmax_c_t, 0)
        # Construct slope
        k_yield_v_c_t = self.Ymax_c_t * inv_ETmax_c_t                           # [n_crop, horizon]
        # Extend it to all players
        null_surface_mask = np.where(self.crop_surface !=0, 1, 0)                    # [n_agent, n_crops]
        k_yield_v_c_t = null_surface_mask[:, :, None] * k_yield_v_c_t[None, :, :]
        return None, k_yield_v_c_t, None                                        # [n_agent, n_crop, horizon]