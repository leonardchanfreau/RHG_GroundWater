from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

class CoarseModel(ABC):
    """
    Coarse model of the GroundWater system.
    
    Contains the dynamics matrices and constraints matrices.
    
    It ensures the translation between the high-dimensional real system and the
    coarse model used by the controller.
    """
    
    def __init__(self, n_col: int, n_row: int,
                 param_file: str):
        # Dimensions of the coarse grid
        self.n_col = n_col
        self.n_row = n_row
        
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
        # Grid parameter matrices
        adjacency, distance_kl, surface_k, cross_section_kl =\
            self._get_grid_parameter_matrices(params)
        # Coarse physical parameters
        if params['physical']['type'] == "uniform":
            conductivity_k = params['physical']['hk'] * adjacency
            storativity_k = params['physical']['sy'] * adjacency
        elif params['physical']['type'] == "variable":
            conductivity_k = self._average_value(params['physical']['hk'])
            storativity_k = self._average_value(params['physical']['sy'])
        # Exogenous parameters
        recharge_k = self._compute_recharge(params['recharge']['coords'],
                                            params['grid']['Lx'],
                                            params['grid']['Ly'])
        # Well positions
        self.well_to_cell = self._compute_well_to_cell(
            params['wells']['pos'], params['grid']['Lx'], params['grid']['Ly'])
        self.input_to_state_v, self.input_to_state_minus_v =\
            self._compute_input_to_state(
                params['wells']['pos'], params['grid']['Lx'], params['grid']['Ly'],
                params['wells']['n_player'], params['wells']['to_player'])
        self.input_to_well = self._compute_input_to_well()
        
        # Compute System matrices
        self._compute_dynamics_matrices(distance_kl,
                                        surface_k.reshape(self.n_col*self.n_row),
                                        cross_section_kl,
                                        conductivity_k.reshape(self.n_col*self.n_row),
                                        storativity_k.reshape(self.n_col*self.n_row),
                                        recharge_k.reshape(self.n_col*self.n_row))
        dL = min(params['grid']['Lx'] / self.n_col, params['grid']['Ly'] / self.n_row)
        S = np.min(storativity_k)
        K = np.max(conductivity_k)
        b = np.max(params['grid']['ztop'] - params['grid']['zbot'])
        self._refine_dynamics_matrices(params['time_step']['dt'], dL,
                                       S, K, b)
        # Compute constraint matrices
        self._compute_constraint_matrices(params["constraints"])
        
    
    def get_dynamics_matrices(self) -> Tuple[sp.lil_array, sp.lil_array, sp.lil_array]:
        return self.A, self.B, self.C
    
    
    def translate_state(self, x_HD: np.ndarray) -> np.ndarray:
        return self._average_value(x_HD).reshape((self.n_col * self.n_row, 1))
    
    def translate_input(self, input_LD: np.ndarray) -> np.ndarray:
        weighted_input_to_well = self.weighted_input_to_well.copy()
        weighted_input_to_well /= weighted_input_to_well.sum(axis=1)[:,np.newaxis]
        return weighted_input_to_well @ input_LD
    
    
    def _compute_dynamics_matrices(self,
                                   distance_kl: sp.lil_array,
                                   surface_k: sp.lil_array,
                                   cross_section_kl: sp.lil_array,
                                   conductivity_k: sp.lil_array,
                                   storativity_k: sp.lil_array,
                                   recharge_k: sp.lil_array) -> None:
        non_zero = distance_kl.nonzero()
        self.A = sp.lil_array(distance_kl.shape)
        self.A[non_zero] = (cross_section_kl[non_zero] / distance_kl[non_zero]) * \
            (conductivity_k / (surface_k * storativity_k))
        self.A += sp.diags(1 - np.sum(self.A, axis=1)).tolil()
        
        self.B_v = [self.input_to_state_v[v]/(surface_k * storativity_k) \
            for v in len(self.input_to_state_v)]
        self.B_minus_v = [self.input_to_state_minus_v[v]/(surface_k * storativity_k) \
            for v in len(self.input_to_state_minus_v)]
        
        self.C = recharge_k / (surface_k * storativity_k)
    
    
    def _compute_constraint_matrices(self, constraints: Dict[str, Any]) -> None:
        if "min_head" in constraints:
            # Regulatory constraint on the heads
            self.H_x = -sp.eye(self.n_col*self.n_row)
            self.h_x = sp.lil_array(
                sp.lil_array([constraints["min_head"]]).T
                * np.ones((self.n_col*self.n_row, 1)))
            # Intersect with the operational constraints on the well depth
            self.h_x = self.h_x.maximum(
                (self.well_to_cell * sp.lil_array([constraints["well_depth"]]))\
                    .max(axis=1).reshape(self.n_col*self.n_row,1))
            # Inverse sign as it is a superior bound
            self.h_x = -self.h_x
        else:
            # Constrain only the cell containing a well
            self.H_x = -self.well_to_cell.copy()
            self.h_x = -sp.lil_array([constraints["well_depth"]]).T
        # Operational constraints on the pumping rates
        n_u = sum([el.shape[1] for el in self.input_to_state_v])
        self.H_u = sp.eye(n_u)
        self.weighted_input_to_well = \
            self.input_to_well.minimum(1).toarray().T \
                * sp.lil_array([constraints["max_pumping_rates"]])
        self.h_u = self.weighted_input_to_well.sum(axis=1).reshape(n_u, 1)
        
        
    def _refine_dynamics_matrices(self, dt: float, dL: float,
                                  S: float, K:float, b: float) -> None:
        """ Adapt the dynamics depending on time-space discretization ratio
        (dt/dL^2) to avoid exponential oscillation of the heads."""
        ratio = dt/(dL**2)
        ratio_max = (1/10) * (S / (K * b))
        while ratio > ratio_max:
            A = self.A @ self.A
            B = self.A @ self.B + self.B
            C = self.A @ self.C + self.C
            self.A, self.B, self.C = A, B, C
            ratio /= 2
    
    @abstractmethod
    def _get_grid_parameter_matrices(self, params) -> \
        Tuple[sp.lil_array, sp.lil_array, sp.lil_array, sp.lil_array]:
        """ Compute geometric parameters of the grid.
            
            Return:
                adjacency: adjacency matrix of the coarse grid [n_x*n_y, n_x*n_y].
                
                distance_kl: array of size [n_x*n_y, n_x*n_y] with non-zero
                value for neighboring cells corresponding to the distance between
                the cells.
                
                surface_k: array of size [n_y,n_x] containing the surface of
                each cell.
                
                cross_section_kl: array of size [n_x*n_y, n_x*n_y] with non-zero
                value for neighboring cells corresponding to the cross-section
                surface between the cells."""
        pass
    
    @abstractmethod
    def _average_value(self, var_HD: np.ndarray) -> np.ndarray:
        """ Compute average value in each cell of the coarse grid. """
        pass
    
    @abstractmethod
    def _compute_recharge(self,
                          recharge_coord: Dict[Tuple[int, int], float],
                          Lx: float, Ly: float) -> sp.lil_array:
        pass
    
    @abstractmethod
    def _compute_well_to_cell(self, well_pos: List[List[int]],
                              Lx: float, Ly: float) -> sp.lil_array:
        pass
    
    @abstractmethod
    def _compute_input_to_state(self, well_pos: List[List[int]],
                                  Lx: float, Ly: float,
                                  n_player: int, to_player: List[int]) -> \
                                      Tuple[List[sp.lil_array], list[sp.lil_array]]:
        pass
    
    @abstractmethod
    def _compute_input_to_well(self) -> sp.lil_array:
        pass