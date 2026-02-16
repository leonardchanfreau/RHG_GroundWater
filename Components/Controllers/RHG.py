from abc import abstractmethod
from typing import List
import numpy as np
import json

from ..ParameterInterface import ParameterInterface
from .OptimizationController import OptimizationController

class RHG(OptimizationController):
    """Abstract object of a Receding Horizon Game optimization problem."""
    
    def __init__(self, param_handler: ParameterInterface, nu_v: List[int], overwrite_time_horizon: int = None):
        # Time Horizon
        self.K = param_handler.controller_time_horizon
        if overwrite_time_horizon is not None:
            self.K = overwrite_time_horizon
        # Number of agent
        self.Nv = param_handler.n_agent
        # Control variable dimensions per agent
        self.nu_v = nu_v
        self.nu = sum(self.nu_v)
        
    def step(self, x_0: np.ndarray):
        U_opt, x_opt = super().step(x_0)
        # Check that the result is indeed a nash equilibrium    -- TODO: de-comment and optimize
        # assert self._check_nash(U_opt, x_0), \
        #     "[RHG] Error: solution found is not the Nash equilibrium."
        return U_opt, x_opt
    
    def _solution_to_command(self, U_opt: np.ndarray):
        """ Reorganize solution vector to time-dependant commands """
        u_v_0 = np.cumsum([0] + self.nu_v[:-1])* self.K
        U_opt = np.hstack([
            U_opt[u_v_0[v]:u_v_0[v]+self.nu_v[v]*self.K].reshape(self.K,self.nu_v[v])
            for v in range(self.Nv)])
        return U_opt
    
    def update_cost_matrices(self, 
                             Q_v_t: List[np.ndarray] = None, R_v_t: List[np.ndarray] = None,
                             R_v_minus_t: List[np.ndarray] = None,
                             M_v_t: List[np.ndarray] = None, L_v_t: List[np.ndarray] = None):
        pass
    
    @abstractmethod
    def _compute_pseudo_gradient(sel, x_0: np.ndarray):
        pass
    