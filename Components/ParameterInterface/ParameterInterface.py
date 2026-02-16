from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import math
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse as sp

from scipy.signal import convolve2d

class ParameterInterface(ABC):
    
    def __init__(self, parameter_file: str):
        # Read parameter file
        self.parameter_file = parameter_file
        with open(parameter_file, 'r') as file:
            params = json.load(file)
        
        # Dimension of the system
        self.Lx     = params['grid']['Lx']
        self.Ly     = params['grid']['Ly']
        self.ztop   = params['grid']['ztop']
        self.zbot   = params['grid']['zbot']
        
        # Modflow discretization parameters
        self.mf_n_col = params['grid']['ncol']
        self.mf_n_row = params['grid']['nrow']
        self.mf_n_lay = params['grid']['nlay']
        self.mf_layer_type = params['physical']['laytyp']
        self.mf_workspace = params['workspace']
        self.mf_time_unit = params['time_step']['unit']
        self.vka = params['physical']['vka']    # Physical ration for Modflow
        
        # Dimensions of the coarse grid
        self.n_col  = params['coarse_model']['n_col']
        self.n_row  = params['coarse_model']['n_row']
        self.nx     = self.n_col * self.n_row
        # Time-step
        self.dt = params['time_step']['dt']
        self.n_steps = 635 // self.dt
        self.controller_time_horizon = params['controller']['time_horizon']
        # Number of agent
        self.n_agent = params['wells']['n_player']
        
        # Position of the wells
        self.well_coords = params['wells']['pos']
        # Wells belonging to each agent
        self.well_to_agent = np.array(params['wells']['to_player'])
        
        # Uniform parameters ?
        self.uniform_params = (params['physical']['type'] == "uniform")
        
        # Recharge rate
        self.recharge_coords = params['recharge']['coords']
        self._compute_recharge()
        
        # Initial heads state (absolute)
        self.init_heads = params['init']['z']
        
        # Crops parameters
        self._compute_crops_parameters(params)
        
        # Rearrange parameters in grid form
        self._set_grid_parameter_matrices()
        
        # Model correction
        self.correction = params.get("correction")
        
        # Boundary condition
        self.boundary_head = params['boundary']['head']
        
        # State maximum depth constraints
        self.well_depth         = sp.csr_array([params["constraints"]["well_depth"]])
        self.min_head           = sp.csr_array(params["constraints"].get("min_head"))   # TODO check if works with None
        # Input constraints
        self.max_pumping_rates  = sp.csr_array(params["constraints"]['max_pumping_rates'])
        
        # Set yield profit function parameters
        self.a_yield_v_c_t, self.b_yield_v_c_t, self.c_yield_v_c_t = \
            self._compute_yield_function_parameters()
    
    def get_dynamics_parameters(self):
        return self.a, self.b, self.c
    
    @abstractmethod
    def get_constraints_parameters(self):
        pass
    
    @abstractmethod
    def get_initial_state(self):
        pass
    
    @abstractmethod
    def get_cost_coefficients(self):
        pass
    
    def get_yield_profit_parameters(self):
        return self.a_yield_v_c_t, self.b_yield_v_c_t, self.c_yield_v_c_t
    
    @abstractmethod
    def get_physical_parameters(self):
        pass
    
    @abstractmethod
    def _compute_dynamics_parameters(self):
        pass
    
    @abstractmethod
    def _compute_yield_function_parameters(self):
        pass
    
    def _set_grid_parameter_matrices(self) -> None:
        """ Compute geometric parameters of the grid with square cells.
            
            Defines:
                adjacency: adjacency matrix of the coarse grid [n_x*n_y, n_x*n_y].
                
                distance_kl: array of size [n_x*n_y, n_x*n_y] with non-zero
                value for neighboring cells corresponding to the distance between
                the cells.
                
                surface_k: array of size [n_y,n_x] containing the surface of
                each cell.
                
                cross_section_kl: array of size [n_x*n_y, n_x*n_y] with non-zero
                value for neighboring cells corresponding to the cross-section
                surface between the cells.
        """
        # Compute adjacency matrices
        horizontal_adjacency = sp.kron(
            sp.eye(self.n_row),
            sp.diags([1, 1], offsets=[-1, 1], shape=(self.n_col, self.n_col))).tocsr()
        vertical_adjacency = sp.kron(
            sp.diags([1, 1], offsets=[-1, 1], shape=(self.n_row, self.n_row)),
            sp.eye(self.n_col)).tocsr()
        self.adjacency = horizontal_adjacency + vertical_adjacency
        # Compute horizontal parameters
        dlx, dly = self.Lx / self.n_col, self.Ly / self.n_row
        self.distance_kl = dlx * horizontal_adjacency + dly * vertical_adjacency
        self.surface_k = (dlx * dly) * sp.csr_array(np.ones((self.n_row, self.n_col)))
        # Compute vertical cross-sections
        b = self.ztop - self.zbot
        if self.uniform_params:
            self.cross_section_kl = b * dly * horizontal_adjacency + \
                b * dlx * vertical_adjacency
        else:
            # Compute average depth along horizontal and vertical links
            data = self._average_value(b).reshape((1, self.nx))
            b_avg_h = (horizontal_adjacency*data + horizontal_adjacency*data.T)/2
            b_avg_v = (vertical_adjacency*data + vertical_adjacency*data.T)/2
            self.cross_section_kl = b_avg_h * dly + b_avg_v * dlx
        # Reshape vectors
        self.surface_k = self.surface_k.reshape((self.nx, 1))
    
    def _compute_recharge(self) -> None:
        recharge = sp.lil_array((self.n_row, self.n_col))
        for coord, r in self.recharge_coords.items():
            row = int(coord[0]*self.n_row/self.Lx)
            col = int(coord[1]*self.n_row/self.Ly)
            recharge[row, col] += r                         # TODO: either nondimensionalize or use relative rates in setup file
        self.recharge_k = recharge.tocsr().reshape((self.nx, 1))
        
    def _compute_crops_parameters(self, params: Dict[str, Any]):
        # Crop surface
        self.land_surface_v     = np.array(params['land']['surface'])
        self.crop_repartition   = np.array(params['crops_repartition'])
        self.crop_surface       = self.land_surface_v[:, None] * self.crop_repartition.T
        self.n_crops            = self.crop_repartition.shape[0]
        # Water demand
        self.ET0            = params['physical']['ET0']
        self.ETmax_c_t      = self._compute_max_water_demand_data(params['crops'])
        # Stage Yield
        self.yield_function_type    = "linear"
        self.Ymax_c_t               = self._construct_yield_data_vector(params['crops'])
        self.intermediate_irrigation_value = params['costs'].get('intermediate_point')
        if self.intermediate_irrigation_value is not None:  # Quadratic yield
            self.yield_function_type= "quadratic"
            self.Y_q_c_t            = self._construct_yield_data_vector(params['crops'],
                                                                'Y_intermediate_t',
                                                                'curvature_stages')
        elif params['crops'][next(iter(params['crops']))].get('Ky_t'):
            self.yield_function_type= "quadratic distance"
            self.Ky_c_t             = self._construct_yield_data_vector(params['crops'],
                                                                'Ky_t',
                                                                'Ky_stages')
            self.yield_sensitivity  = 1e3*np.array(params['costs']['yield_sensitivity']) # TODO: justification and avoid hardcoding (careful hardcoded)
            self.rain               = params['rain']
    
    def _compute_max_water_demand_data(self,
                                       crops_param: Dict[str, Dict[str, List[float]]]) -> List[np.ndarray]:
        """ Time-varying info about maximum water demand. Ex: max evapotranspiration"""
        ts = np.arange(0, self.n_steps)
        ETmax_c_t = []
        for crop in crops_param.values():
            ET_c_t = np.array(crop['Kc_t'])*self.ET0
            # Shift to plant date
            Kc_stages = np.array(crop['Kc_stages']) + crop['plant_date']
            if crop['plant_date'] != 0:
                Kc_stages = np.hstack([[0], Kc_stages[0]-1, Kc_stages])
                ET_c_t = np.hstack([[0], [0], ET_c_t])
            Kc_stages = np.hstack([Kc_stages, Kc_stages[-1]+1])
            ET_c_t = np.hstack([ET_c_t, [0]])
            ETmax_c_t.append(np.interp(ts, Kc_stages, ET_c_t))
        return np.array(ETmax_c_t)
            
    def _construct_yield_data_vector(self,
                                     crops_param: Dict[str, Dict[str, List[float]]],
                                     yield_type: str = 'Y_max_t',
                                     stages_type: str = 'Kc_stages') -> List[np.ndarray]:
        ts = np.arange(0, self.n_steps)
        Y_c_t_vec = []
        for crop in crops_param.values():
            Y_c_t  = np.array(crop[yield_type])
            stages = np.array(crop[stages_type])
            # Shift to plant date
            if crop['plant_date'] != 0:
                Y_c_t = np.hstack([[0], Y_c_t])
                stages = np.hstack([[0], stages + crop['plant_date']])
            Y_c_t = np.hstack([Y_c_t, [0]])
            stages = np.hstack([stages, stages[-1]+1])
            indices = np.searchsorted(stages, ts, side="right") - 1
            Y_c_t_vec.append(Y_c_t[indices])
        return np.array(Y_c_t_vec)
    
    def _average_value(self, var_HD: np.ndarray) -> np.ndarray:
        """ Compute average value in each square cell of the coarse grid. """
        # Compute discretization ratio
        n_x, n_y = var_HD.shape
        x_ratio = n_x / self.n_col
        y_ratio = n_y / self.n_row
        # Construct averaging filter
        avg_filter = np.ones((math.floor(x_ratio), math.floor(y_ratio)))
        avg_filter /= np.sum(avg_filter)
        # Apply the filter
        var = convolve2d(var_HD, avg_filter, mode="valid")[::math.ceil(x_ratio), ::math.ceil(y_ratio)]
        return var