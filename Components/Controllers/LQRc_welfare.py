import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from .OptimizationController import OptimizationController
from .RHG import RHG

class LQRc_welfare(OptimizationController):
    """
    Transform a Receding Horizon Game optimization problem into a simple
    quadratic cost optimization problem to find the social welfare.
    
    The social welfare is defined as the sum of the costs of all players:
    
    min_{x,u} ∑ ∑ ( x^T Q x + uv^T R uv + x^T M uv + L uv)
        s.t x^+ = f(x,u)
            H_x @ x < h_x
            H_u @ u < h_u
            x_0 = x
    
    (/!\\ Q symmetric)
    """
    
    def __init__(self, game: RHG):
        # Copy cost matrices
        self.H = game.H.copy()
        self.h_x_0 = game.h_x_0.copy()
        self.h = game.h.copy()
        # Construct social welfare cost matrices
        self.R = game.F - sp.block_diag([R_vv.T*1e3 for R_vv in game.R_vv])   #TODO:check that we do not need a transpose somewhere
        self.M = game.f_x.T
        self.L = game.f.T
        # Copy transition function
        self._apply_commands = game._apply_commands
        self._solution_to_command = game._solution_to_command
        # Copy other parameters
        self.K = game.K
        self.n_players = game.n_players
        self.nu_v = game.nu_v
        self.nu = game.nu
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            self.m = gp.Model("matrix1", env)
        self.u = self.m.addMVar(shape=(self.K*self.nu,1), lb=-np.inf,
                                vtype=gp.GRB.CONTINUOUS, name="U")
    

    def _solve_problem(self, x0: np.ndarray) -> np.ndarray:
        # The social welfare is defined as the sum of the cost function
        self.m.setObjective(
            self.u.T @ self.R @ self.u + x0.T @ self.M @ self.u + self.L @ self.u,
            gp.GRB.MINIMIZE)
        # Add constraint
        h = self.h_x_0 @ x0 + self.h
        self.m.addConstr((self.H @ self.u) <= h, name="H")
        self.m.optimize()
        return self.u.X.copy()
    