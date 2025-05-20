from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from ..utils.mat_utils import matrix_poly_sum

class CoarseModel(ABC):
    """
    Coarse model of the GroundWater system.
    
    Contains the dynamics matrices and constraints matrices.
    
    It ensures the translation between the high-dimensional real system and the
    coarse model used by the controller.
    """
    
    def __init__(self, param_file: str, boundary_condition: str="Free-flow"):
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
        # Dimensions of the coarse grid
        self.n_col = params['coarse_model']['n_col']
        self.n_row = params['coarse_model']['n_row']
        self.nx = self.n_col * self.n_row
        # Number of agent
        self.n_agent = params['wells']['n_player']
        # Wells belonging to each agent
        self.well_to_agent = np.array(params['wells']['to_player'])
        
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
        self.input_to_state_v = self._compute_input_to_state(
            params['wells']['n_player'], self.well_to_agent)
        self.input_to_well = self._compute_input_to_well()
        # Number of input per agent
        self.nu_v = [np.count_nonzero(self.input_to_well.T @ \
                     (self.well_to_agent==v)) \
                         for v in range(self.n_agent)]
        
        # Compute System matrices
        self._compute_dynamics_matrices(distance_kl,
                                        surface_k.reshape((self.nx, 1)),
                                        cross_section_kl,
                                        conductivity_k.reshape((self.nx, 1)),
                                        storativity_k.reshape((self.nx, 1)),
                                        recharge_k.reshape((self.nx, 1)),
                                        params['time_step']['dt'])
        if boundary_condition == "Constant Head":
            self._add_constant_head_condition(params,
                                              surface_k.reshape((self.nx, 1)),
                                              conductivity_k.reshape((self.nx, 1)),
                                              storativity_k.reshape((self.nx, 1)))
        self._refine_dynamics_matrices(tol=1e-10)
        # Compute constraint matrices
        self._compute_constraint_matrices(params["constraints"])
        # Compute cost matrices
        self.extraction_cost = params['costs']['extraction_cost']
        self.production_profit = params['costs']['production_profit']
        self.initial_state = params['grid']['ztop'].reshape((self.nx, 1)) \
            if type(params['grid']['ztop']) is np.ndarray else \
                params['grid']['ztop'] * np.ones((self.nx, 1))
        self._compute_cost_matrices()
        # Keep track of state
        self.state = self.initial_state
        
    
    def get_dynamics_matrices(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        return self.A, self.B_v, self.C
    
    
    def translate_state(self, x_HD: np.ndarray) -> np.ndarray:
        self.state = self._average_value(x_HD).reshape((self.nx, 1))
        return self.state
    
    def get_initial_state(self) -> np.ndarray:
        return self.state
    
    def translate_input(self, input_LD: np.ndarray) -> np.ndarray:
        weighted_input_to_well = self.weighted_input_to_well.copy()
        weighted_input_to_well /= weighted_input_to_well.sum(axis=0)[np.newaxis, :]
        return weighted_input_to_well @ input_LD
    
    def compute_cost(self, u: np.ndarray) -> np.ndarray:
        cost = np.zeros((self.n_agent, 1))
        for v in range(self.n_agent):
            input_indices = np.nonzero(self.well_to_cell[:,self.well_to_agent==v].T)[1]
            cost[v] = (-self.extraction_cost[v] * (self.initial_state[input_indices] -\
                self.state[input_indices]) * u[self.well_to_agent==v] + \
                    self.production_profit[v] * u[self.well_to_agent==v]).sum()
        return cost
    
    def get_heads_at_wells(self) -> np.ndarray:
        return self.state[np.nonzero(self.well_to_cell.T)[1]]
    
    def select_wells(self, states: np.ndarray) -> np.ndarray:
        return states[np.nonzero(self.well_to_cell.T)[1], :]
    
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
        self.A *= dt * (conductivity_k / (surface_k * storativity_k))
        self.A += sp.diags(1 - np.sum(self.A, axis=1)).tolil()
        self.A = self.A.tocsr()
        
        self.B_v = [sp.csr_array(
            dt * self.input_to_state_v[v]/(surface_k * storativity_k).toarray()) \
                for v in range(self.n_agent)]
        
        self.C = dt * recharge_k / (surface_k * storativity_k)
        
        
    def _add_constant_head_condition(self,
                                     params,
                                     surface_k: sp.csr_array,
                                     conductivity_k: sp.csr_array,
                                     storativity_k: sp.csr_array) -> None:
        # Get indices of the boundary cells (corners are counted double on purpose)
        bound_index = [
            np.arange(self.n_col),                                      # Top row
            (self.n_row - 1) * self.n_col + np.arange(self.n_col),      # Bottom row
            np.arange(0, self.n_row) * self.n_col,                      # Left column (including corners)
            np.arange(0, self.n_row) * self.n_col + (self.n_col - 1)    # Right column (including corners)
        ]
        # Get boundary parameters
        distance_bound, cross_section_bound = self._get_boundary_parameters(params)
        # Add flows coming from the virtual neighboring constant head cells
        for i in range(4):
            self.A[(bound_index[i], bound_index[i])] -= \
                ((cross_section_bound[i] / distance_bound[i]) * \
                    (conductivity_k / (surface_k * storativity_k))[bound_index[i]].T).flatten()       # TODO: indexing 
            self.C[bound_index[i]] += (params['boundary']['head'] * (cross_section_bound[i] / distance_bound[i]) * \
                (conductivity_k / (surface_k * storativity_k))[bound_index[i]].T).T
        
        
    def _refine_dynamics_matrices(self, tol:float) -> None:
        """ Adapt the dynamics depending on time-space discretization ratio
        (dt/dL^2) to avoid exponential oscillation of the heads."""
        
        # Find the amount r of reduction of the time-step needed to ensure stability
        r = 1
        max_eig = np.abs(np.linalg.eigvals(self.A.toarray())).max()
        while (max_eig > 1 + tol):      # TODO: find criteria for model accuracy
            r += 1
            new_A_base = self.A.toarray()/r + (1-1/r)*np.eye(self.A.shape[0])
            max_eig = np.abs(np.linalg.eigvals(new_A_base)).max()
        # Construct new A and B matrices corresponding to a constant command for r time-steps
        if r > 1:
            poly_A = sp.csr_array(matrix_poly_sum(new_A_base, r-1))
            new_B_base = [B_v/r for B_v in self.B_v]
            new_C_base = self.C/r
            self.A = sp.csr_array(np.linalg.matrix_power(new_A_base, r))
            self.B_v = [poly_A @ B_v for B_v in new_B_base]
            self.C = poly_A @ new_C_base
    
    
    def _compute_constraint_matrices(self, constraints: Dict[str, Any]) -> None:
        # Well depth constraint: select limiting depth on each cell
        max_pumping_cell_depth = ((self.input_to_well).T * sp.lil_array([constraints["well_depth"]])).toarray()
        max_pumping_cell_depth = \
            np.where(max_pumping_cell_depth, max_pumping_cell_depth, -np.inf).max(axis=1)
        input_to_cell = (self.well_to_cell @ self.input_to_well)
        if "min_head" in constraints:
            # Regulatory constraint on the heads
            self.H_x = -sp.eye(self.nx)
            self.h_x = sp.csr_matrix(
                sp.lil_array([constraints["min_head"]]).T
                * np.ones((self.nx, 1)))
            # Intersect with the operational constraints on the well depth
            pumping_cell_index = input_to_cell.T @ np.array(range(self.nx))
            self.h_x[pumping_cell_index] = self.h_x[pumping_cell_index].maximum(
                max_pumping_cell_depth[:,np.newaxis])
            # Inverse sign as it is a superior bound
            self.h_x = -self.h_x
        else:
            # Constrain only the cell containing a well
            self.H_x = -input_to_cell.T
            self.h_x = -sp.csr_array(max_pumping_cell_depth[:,np.newaxis])
        # Upper constraint on pumping rate is given by summing cstr in each cells
        self.weighted_input_to_well = \
            (self.input_to_well.T \
                * sp.lil_array([constraints["max_pumping_rates"]])).T
        up_cstr = self.weighted_input_to_well.sum(axis=0)
        # Assemble operational constraints on the pumping rates
        self.H_u_v = []
        self.h_u_v = []
        for v in range(self.n_agent):
            self.H_u_v.append(sp.kron([[1],[-1]], sp.eye(self.nu_v[v])).tocsr())
            up_cstr_v = up_cstr[sum(self.nu_v[:v]):sum(self.nu_v[:v+1])]
            self.h_u_v.append(
                sp.vstack((sp.csr_array((self.nu_v[v], 1)),
                           sp.csr_array(up_cstr_v.reshape(self.nu_v[v], 1)))))
        
    def _compute_cost_matrices(self):
        # Input-state cross term
        self.Gamma_v = [extr_cost * self.input_to_state_v[v] for v, extr_cost
                        in enumerate(self.extraction_cost)]
        # Linear input cost
        nu_v = [el.shape[1] for el in self.input_to_state_v]
        self.Xi_v = [profit * np.ones((1,nu_v[v])) - \
            (self.extraction_cost[v] * self.initial_state[np.nonzero(self.input_to_state_v[v].T)[1]]).T
                     for v, profit in enumerate(self.production_profit)]
    
    
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
    def _get_boundary_parameters(self, params) -> Tuple[List[sp.csr_array], List[sp.csr_array]]:
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
                                      List[sp.csr_array]:
        pass
    
    @abstractmethod
    def _compute_input_to_well(self) -> sp.csr_array:
        pass