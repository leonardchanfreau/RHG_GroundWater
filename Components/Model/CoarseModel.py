from typing import Tuple, Dict, List, Any
from abc import ABC, abstractmethod

import json
import numpy as np
import scipy.sparse as sp

from ..ParameterInterface import ParameterInterface
from ..utils.mat_utils import matrix_poly_sum

class CoarseModel(ABC):
    """
    Coarse model of the GroundWater system.
    
    Contains the dynamics matrices and constraints matrices.
    
    It ensures the translation between the high-dimensional real system and the
    coarse model used by the controller.
    """
    
    def __init__(self, param_handler: ParameterInterface, boundary_condition: str="Free-flow"):
        # Dimensions of the coarse grid
        self.n_col = param_handler.n_col
        self.n_row = param_handler.n_row
        self.nx = param_handler.nx
        # Number of agent
        self.n_agent = param_handler.n_agent
        # Wells belonging to each agent
        self.well_to_agent = param_handler.well_to_agent
        
        # mapping from well to coarse cells
        self.well_to_cell = self._compute_well_to_cell(
            param_handler.well_coords, param_handler.Lx, param_handler.Ly)
        # wells in the same cell are controlled with one pumping command
        self.command_v_to_cell = self._compute_command_to_cell()
        self.well_to_command_v = [c2x.T @ self.well_to_cell for c2x in self.command_v_to_cell]
        self.n_command_v = [w2c.shape[0] for w2c in self.well_to_command_v]
        self.n_well_per_command = [w2c.sum(axis=1) for w2c in self.well_to_command_v]
        # inputs can be more detailed than pumping command (e.g. crop-water allocation)
        self.input_v_to_command_v = self._compute_input_to_command()
        self.input_to_pumping_in_well = sp.hstack([w2c.T @ u2c for u2c, w2c in zip(
            self.input_v_to_command_v, self.well_to_command_v)]) / np.kron(
                np.hstack(self.n_well_per_command), np.ones((1,self.unit_input_dim)))
        self.input_per_well = sp.kron(
            sp.vstack(self.well_to_command_v).T, np.eye(self.n_crops)) / \
                np.kron(np.hstack(self.n_well_per_command),
                        np.ones((1,self.unit_input_dim)))
        # input dimension per agent
        self.nu_v = [u2c.shape[1] for u2c in self.input_v_to_command_v]
        
        # Compute System matrices
        a, b, c = param_handler.get_dynamics_parameters()
        self._compute_dynamics_matrices(a, b, c)
        if boundary_condition == "Constant Head":
            self._add_constant_head_condition(a, param_handler.boundary_head)
        self._refine_dynamics_matrices(tol=1e-10)
        # Compute constraint matrices
        well_depth, min_head, max_pumping_rates = param_handler.get_constraints_parameters()
        self._compute_constraint_matrices(well_depth, min_head, max_pumping_rates)
        # Model correction
        self.correct_heads = False
        correction = param_handler.correction
        if (correction is not None):
            self.correct_heads = bool(correction['correct_heads'])
            if self.correct_heads:
                self.heads_correction = correction.get("head")
                self._tighten_constraints()
        # Keep track of state
        self.initial_state = param_handler.get_initial_state()
        self.state = self.initial_state
        # Get scaling terms (especially useful for dimensionless model)
        self.state_scaling, self.input_scaling = param_handler.get_scaling_terms()
        
    
    def get_dynamics_matrices(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        return self.A, self.B_v, self.C
    
    def translate_state(self, x_HD: np.ndarray) -> np.ndarray:
        self.state = self._average_value(x_HD).reshape((self.nx, 1)) * self.state_scaling
        return self.state
    
    def get_initial_state(self) -> np.ndarray:
        return self.state
    
    def translate_input(self, input_LD: np.ndarray) -> np.ndarray:
        return self.input_to_pumping_in_well @ input_LD / self.input_scaling
    
    def reorganize_input_per_well(self, inputs:np.ndarray) -> np.ndarray:
        return self.input_per_well @ inputs
    
    def get_heads_at_wells(self) -> np.ndarray:
        return self.state[np.nonzero(self.well_to_cell.T)[1]]
    
    def select_wells(self, states: np.ndarray) -> np.ndarray:
        return states[np.nonzero(self.well_to_cell.T)[1], :]
    
    def apply_head_correction(self, avg_heads: np.ndarray, last_pumping_rates: np.ndarray):
        if self.correct_heads:
            return avg_heads - self.heads_correction * last_pumping_rates
        return avg_heads
    
    
    def _compute_dynamics_matrices(self, a: sp.csr_array, b: sp.csr_array, c: sp.csr_array) -> None:
        self.A = a
        self.A += sp.diags(1 - np.sum(self.A, axis=1)).tolil()
        self.A = self.A.tocsr()
        
        self.B_v = [sp.csr_array(
            (self.command_v_to_cell[v] @ self.input_v_to_command_v[v]) * b) \
                for v in range(self.n_agent)]
        
        self.C = c
        
        
    def _add_constant_head_condition(self, a: sp.csr_array, boundary_head: float) -> None:
        # Get indices of the boundary cells (corners are counted double on purpose)
        bound_index = [
            np.arange(self.n_col),                                      # Top row
            (self.n_row - 1) * self.n_col + np.arange(self.n_col),      # Bottom row
            np.arange(0, self.n_row) * self.n_col,                      # Left column (including corners)
            np.arange(0, self.n_row) * self.n_col + (self.n_col - 1)    # Right column (including corners)
        ]
        # Get boundary parameters
        # distance_bound, cross_section_bound = self._get_boundary_parameters(params) TODO: delete
        # Add flows coming from the virtual neighboring constant head cells
        for i in range(4):
            # Mean of non-zero terms
            counts = np.count_nonzero(a.toarray()[bound_index[i],:], axis=1)
            mean = np.divide(a.toarray()[bound_index[i],:].sum(axis=1), counts, where=counts!=0)
            self.A[(bound_index[i], bound_index[i])] -= mean
            self.C[bound_index[i]] += (boundary_head * mean).reshape(-1, 1)
        
        
    def _refine_dynamics_matrices(self, tol:float) -> None:
        """ Adapt the dynamics depending on time-space discretization ratio
        (dt/dL^2) to avoid exponential oscillation of the heads."""
        
        # Find the amount r of reduction of the time-step needed to ensure stability
        r = 1
        max_eig = np.abs(np.linalg.eigvals(self.A.toarray())).max()
        while (max_eig > 1 + tol): #or r<=10:      # TODO: find criteria for model accuracy
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
    
    
    def _compute_constraint_matrices(self, well_depth: sp.csr_array, min_head: sp.csr_array, max_pumping_rates: sp.csr_array) -> None:
        # Well depth constraint: select limiting depth on each cell
        max_pumping_cell_depth = (sp.vstack(self.well_to_command_v) * well_depth).toarray()
        max_pumping_cell_depth = \
            np.where(max_pumping_cell_depth, max_pumping_cell_depth, -np.inf).max(axis=1)
        if min_head is not None:
            # Regulatory constraint on the heads
            self.H_x = -sp.eye(self.nx)
            self.h_x = sp.csr_matrix(min_head.T * np.ones((self.nx, 1)))
            # Intersect with the operational constraints on the well depth
            pumping_cell_index = sp.hstack(self.command_v_to_cell).T @ np.array(range(self.nx))
            self.h_x[pumping_cell_index] = self.h_x[pumping_cell_index].maximum(
                max_pumping_cell_depth[:,np.newaxis])
            # Inverse sign as it is a superior bound
            self.h_x = -self.h_x
        else:
            # Constrain only the cell containing a well
            self.H_x = -sp.hstack(self.command_v_to_cell).T
            self.h_x = -sp.csr_array(max_pumping_cell_depth[:,np.newaxis])
        # Assemble operational constraints on the pumping rates
        max_pumping_rate = (sp.vstack(self.well_to_command_v).T @ max_pumping_rates.T).tocsr()
        self.H_u_v = []
        self.h_u_v = []
        for v in range(self.n_agent):
            H_u_v_low = sp.identity(self.nu_v[v])
            # Upper constraint on pumping rate is given by summing cstr in each cells
            H_u_v_up = -self.input_v_to_command_v[v]
            up_cstr_v = max_pumping_rate[self.well_to_agent==v]
            self.H_u_v.append(sp.vstack([H_u_v_low, H_u_v_up]))
            self.h_u_v.append(
                sp.vstack((sp.csr_array((self.nu_v[v], 1)),
                           sp.csr_array(up_cstr_v))))
        # No mixed constraints by default
        self.H_xu_x, self.H_xu_u, self.h_xu = None, None, None
    
    def _tighten_constraints(self):
        """ Modify state constraint into a mixed constraint between heads and
        pumping rate to count correction. """
        # Everything except the correction term is the same as constraints on x
        self.H_xu_x = self.H_x.copy()
        self.h_xu = self.h_x.copy()
        self.H_x= None
        self.h_x= None
        # Check if all the heads are constrained or only the heads in the well
        all_heads_constrained = (self.H_xu_x.shape[0] == self.nx)
        if all_heads_constrained:
            self.H_xu_u = self.heads_correction * \
                (self.well_to_cell @ self.input_to_pumping_in_well)
        else:
            self.H_xu_u = self.heads_correction * \
                sp.diags(self.input_to_pumping_in_well.sum(axis=0), format="csr")
                
    @abstractmethod
    def _average_value(self, var_HD: np.ndarray) -> np.ndarray:
        """ Compute average value in each cell of the coarse grid. """
        pass
    
    @abstractmethod
    def _compute_well_to_cell(self, well_pos: List[List[int]],
                              Lx: float, Ly: float) -> sp.csr_array:
        pass
    
    @abstractmethod
    def _compute_command_to_cell(self) -> List[sp.csr_array]:
        """Assign each player's commands to the position of the corresponding
        cell on the grid."""
        pass
    
    @abstractmethod
    def _compute_input_to_command(self) -> List[sp.csr_array]:
        """Transform the input vector of each players into aggregated pumping
        command (sum command of wells in the same cell)."""
        pass