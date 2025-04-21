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
    
    def __init__(self, param_file: str):
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
        # Dimensions of the coarse grid
        self.n_col = params['coarse_model']['n_col']
        self.n_row = params['coarse_model']['n_row']
        
        # Grid parameter matrices
        adjacency, distance_kl, surface_k, cross_section_kl =\
            self._get_grid_parameter_matrices(params)
        # Physical parameters
        hk = params['physical']['hk']           # [m/d]
        sy = params['physical']['sy']           # [dimless]
        if params['physical']['type'] == "uniform":
            conductivity_k = sp.csr_array(
                hk * np.ones((self.n_col, self.n_row)))
            storativity_k = sp.csr_array(
                sy * np.ones((self.n_col, self.n_row)))
        elif params['physical']['type'] == "variable":
            conductivity_k =  sp.csr_array(self._average_value(hk))
            storativity_k = sp.csr_array(self._average_value(sy))
        # Exogenous parameters
        recharge_k = self._compute_recharge(params['recharge']['coords'],
                                            params['grid']['Lx'],
                                            params['grid']['Ly'])
        # Well positions
        self.well_to_cell = self._compute_well_to_cell(
            params['wells']['pos'], params['grid']['Lx'], params['grid']['Ly'])
        self.input_to_state_v, self.input_to_state_minus_v =\
            self._compute_input_to_state(
                params['wells']['n_player'], params['wells']['to_player'])
        self.input_to_well = self._compute_input_to_well()
        
        # Compute System matrices
        self._compute_dynamics_matrices(distance_kl,
                                        surface_k.reshape((self.n_col*self.n_row, 1)),
                                        cross_section_kl,
                                        conductivity_k.reshape((self.n_col*self.n_row, 1)),
                                        storativity_k.reshape((self.n_col*self.n_row, 1)),
                                        recharge_k.reshape((self.n_col*self.n_row, 1)),
                                        params['time_step']['dt'])
        dL = min(params['grid']['Lx'] / self.n_col, params['grid']['Ly'] / self.n_row)
        S = storativity_k.min()
        K = conductivity_k.min()
        b = np.max(params['grid']['ztop'] - params['grid']['zbot'])
        # self._refine_dynamics_matrices(params['time_step']['dt'], dL,
        #                                S, K, b)
        # Compute constraint matrices
        self._compute_constraint_matrices(params["constraints"])
        # Compute cost matrices
        self._compute_cost_matrices(params['costs'], params['grid']['ztop'])
        
    
    def get_dynamics_matrices(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        return self.A, self.B, self.C
    
    
    def translate_state(self, x_HD: np.ndarray) -> np.ndarray:
        return self._average_value(x_HD).reshape((self.n_col * self.n_row, 1))
    
    def translate_input(self, input_LD: np.ndarray) -> np.ndarray:
        weighted_input_to_well = self.weighted_input_to_well.copy()
        weighted_input_to_well /= weighted_input_to_well.sum(axis=0)[np.newaxis, :]
        return weighted_input_to_well @ input_LD
    
    
    def _compute_dynamics_matrices(self,
                                   distance_kl: sp.csr_array,
                                   surface_k: sp.csr_array,
                                   cross_section_kl: sp.csr_array,
                                   conductivity_k: sp.csr_array,
                                   storativity_k: sp.csr_array,
                                   recharge_k: sp.csr_array,
                                   dt: int) -> None:
        non_zero = distance_kl.nonzero()
        self.A = sp.lil_array(distance_kl.shape)
        self.A[non_zero] = (cross_section_kl[non_zero] / distance_kl[non_zero])
        self.A *= (conductivity_k / (surface_k * storativity_k))
        self.A += sp.diags(1 - np.sum(self.A, axis=1)).tolil()
        self.A = dt * self.A.tocsr()
        
        self.B_v = [sp.csr_array(
            dt * self.input_to_state_v[v]/(surface_k * storativity_k).toarray()) \
                for v in range(len(self.input_to_state_v))]
        self.B_minus_v = [sp.csr_array(
            dt * self.input_to_state_minus_v[v]/(surface_k * storativity_k).toarray()) \
                for v in range(len(self.input_to_state_minus_v))]
        
        self.C = dt * recharge_k / (surface_k * storativity_k)
        
        
    def _refine_dynamics_matrices(self, dt: float, dL: float,
                                  S: float, K:float, b: float) -> None:
        """ Adapt the dynamics depending on time-space discretization ratio
        (dt/dL^2) to avoid exponential oscillation of the heads.
        
        TODO: review criteria -> current one seem to be wrong. """
        ratio = dt/(dL**2)
        ratio_max = (1/10) * (S / (K * b))
        while ratio > ratio_max:
            A = self.A @ self.A
            B_v = [self.A @ B + B for B in self.B_v]
            B_minus_v = [self.A @ B + B for B in self.B_minus_v]
            C = self.A @ self.C + self.C
            self.A, self.B_v, self.B_minus_v, self.C = A, B_v, B_minus_v, C
            ratio /= 2
    
    
    def _compute_constraint_matrices(self, constraints: Dict[str, Any]) -> None:
        # Well depth constraint: select limiting depth on each cell
        max_pumping_cell_depth = ((self.input_to_well).minimum(1).T * sp.lil_array([constraints["well_depth"]])).toarray()
        max_pumping_cell_depth = \
            np.where(max_pumping_cell_depth, max_pumping_cell_depth, -np.inf).max(axis=1)
        input_to_cell = (self.well_to_cell @ self.input_to_well).minimum(1)
        if "min_head" in constraints:
            # Regulatory constraint on the heads
            self.H_x = -sp.eye(self.n_col*self.n_row)
            self.h_x = sp.csr_matrix(
                sp.lil_array([constraints["min_head"]]).T
                * np.ones((self.n_col*self.n_row, 1)))
            # Intersect with the operational constraints on the well depth
            pumping_cell_index = input_to_cell.T @ np.array(range(self.n_col * self.n_row))
            self.h_x[pumping_cell_index] = self.h_x[pumping_cell_index].maximum(
                max_pumping_cell_depth[:,np.newaxis])
            # Inverse sign as it is a superior bound
            self.h_x = -self.h_x
        else:
            # Constrain only the cell containing a well
            self.H_x = -input_to_cell.T
            self.h_x = -sp.csr_array(max_pumping_cell_depth[:,np.newaxis])
        # Operational constraints on the pumping rates
        n_u = sum([el.shape[1] for el in self.input_to_state_v])
        self.H_u = sp.kron([[1],[-1]], sp.eye(n_u)).tocsr() 
        self.weighted_input_to_well = \
            (self.input_to_well.minimum(1).T \
                * sp.lil_array([constraints["max_pumping_rates"]])).T
        self.h_u = sp.vstack((
            sp.csr_array((n_u, 1)),
            sp.csr_array(self.weighted_input_to_well.sum(axis=0).reshape(n_u, 1))))
        
        
    def _compute_cost_matrices(self, cost: Dict[str, List[float]],
                               ztop: np.ndarray|float):
        # Input-state cross term
        self.Gamma_v = [extr_cost * self.input_to_state_v[v] for v, extr_cost
                        in enumerate(cost['extraction_cost'])]
        # Linear input cost
        nu_v = [el.shape[1] for el in self.input_to_state_v]
        self.Xi_v = [profit * np.ones((1,nu_v[v])) - cost['extraction_cost'][v] * ztop
                     for v, profit in enumerate(cost['production_profit'])]
    
    
    @abstractmethod
    def _get_grid_parameter_matrices(self, params) -> \
        Tuple[sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array]:
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
                          Lx: float, Ly: float) -> sp.csr_array:
        pass
    
    @abstractmethod
    def _compute_well_to_cell(self, well_pos: List[List[int]],
                              Lx: float, Ly: float) -> sp.csr_array:
        pass
    
    @abstractmethod
    def _compute_input_to_state(self, well_pos: List[List[int]],
                                  Lx: float, Ly: float,
                                  n_player: int, to_player: List[int]) -> \
                                      Tuple[List[sp.csr_array], list[sp.csr_array]]:
        pass
    
    @abstractmethod
    def _compute_input_to_well(self) -> sp.csr_array:
        pass