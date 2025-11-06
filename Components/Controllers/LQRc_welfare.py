from typing import List
import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from .OptimizationController import OptimizationController
from .RHG import RHG
from ..Solver.optimizer_fbrs import fbrs

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
    
    def __init__(self, game: RHG, solver: str = 'fbrs'):
        # Copy cost matrices
        self.H = game.H.copy()
        self.h_x_0 = game.h_x_0.copy()
        self.h = game.h.copy()
        # Copy base game to access its methods
        self.game = game
        # Copy other parameters
        self.K = game.K
        self.nu_v = game.nu_v
        self.nu = game.nu
        
        # Solver
        self.solver = solver
        if self.solver == 'fbrs':
            self.n_cstrt = self.H.shape[0]
            # Initial guess
            self.u0 = np.zeros((self.K*self.nu,1))
            self.v0 = np.zeros((self.n_cstrt,1))     # Dual
        elif self.solver == 'gurobi':
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                self.m = gp.Model("matrix1", env)
            self.u = self.m.addMVar(shape=(self.K*self.nu,1), lb=-np.inf,
                                    vtype=gp.GRB.CONTINUOUS, name="U")
        else:
            raise NotImplementedError(f"[RHG] Error: solver {self.solver} is not"+
                                      "yet available.")
    

    def update_cost_matrices(self, 
                             Q_v_t: List[np.ndarray] = None, R_v_t: List[np.ndarray] = None,
                             R_v_minus_t: List[np.ndarray] = None,
                             M_v_t: List[np.ndarray] = None, L_v_t: List[np.ndarray] = None):
        self.game.update_cost_matrices(Q_v_t, R_v_t, R_v_minus_t, M_v_t, L_v_t)
        # Construct social welfare cost matrices
        self.R = (self.game.F - sp.block_diag([1e3*R_vv.T for R_vv in self.game.R_vv]))
        self.M = self.game.f_x.T
        self.L = self.game.f.T
    
    def _solve_problem(self, x0: np.ndarray) -> np.ndarray:
        if self.solver == 'gurobi':
            # The social welfare is defined as the sum of the cost function
            self.m.setObjective(
                self.u.T @ self.R @ self.u + x0.T @ self.M @ self.u + self.L @ self.u,
                gp.GRB.MINIMIZE)
            # Add constraint
            h = self.h_x_0 @ x0 + self.h
            self.m.addConstr((self.H @ self.u) <= h, name="H")
            self.m.optimize()
            return self.u.X.copy(), None
        elif self.solver == 'fbrs':
            f = sp.csr_array(self.M.T @ x0 + self.L.T)
            h = sp.csr_array(self.h_x_0 @ x0 + self.h)
            # Call solver
            U_opt, U_dual, _, out = fbrs((self.R+self.R.T).toarray(), f,
                                         self.H.toarray(), h.toarray(),
                                         self.u0, self.v0)
            # Manage non-convergence cases
            if out['eflag'] == -1:
                raise RuntimeError("[RHG] Error: solver fbrs was unable to converge" +\
                                    " (Maximum number of iterations exceeded)")
            # Reset initial guess for future faster convergence
            self.u0 = U_opt.copy()
            self.v0 = U_dual.copy()
            return U_opt, U_dual
        
    
    def _apply_commands(self, x_0: np.ndarray, U_opt: np.ndarray):
        return self.game._apply_commands(x_0, U_opt)
    
    def _solution_to_command(self, U_opt: np.ndarray):
        return self.game._solution_to_command(U_opt)