from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from .RHG import RHG
from ..Solver.optimizer_fbrs import fbrs

class LTI_LQRc_RHG(RHG):
    """
    Receding Horizon Game optimization problem with LTI dynamics and quadratic
    cost function with crossed terms between state and inputs.
    
    Solve the following problem for all player v:
    
    min_{x,u} ∑ ( x^T Q x + uv^T R uv + x^T M uv + L uv)
        s.t x^+ = A @ x + B^v @ u^v + B^-v @ u^-v + C
            H_x @ x < h_x
            H_u @ u < h_u
            x_0 = x
    
    (/!\\ Q symmetric)
    """
    
    def __init__(self, param_file: str,
                 A: np.ndarray, B_v: List[np.ndarray], C: np.ndarray,
                 H_x: np.ndarray, h_x: np.ndarray, H_u_v: List[np.ndarray], h_u_v: List[np.ndarray],
                 Q_v: List[np.ndarray] = None, R_v: List[np.ndarray] = None,
                 M_v: List[np.ndarray] = None, L_v: List[np.ndarray] = None,
                 solver: str = 'fbrs'):                       # TODO: modify to work with sparse
        super().__init__(param_file)
        
        # Variable dimensions
        self.nx = A.shape[0]
        self.nu_v = [B.shape[1] for B in B_v]
        self.nu = sum(self.nu_v)
        self.n_players = len(self.nu_v)
        
        # Dynamics matrices
        self.A = A                      # [nx, nx]
        self.B_v = B_v                  # [Nv, nx, nu_v]
        self.C = C                      # [nx, 1]
        
        #Constraint matrices
        self.H_x = H_x                  # [nx, nx]
        self.h_x = h_x                  # [nx, 1]
        self.H_u_v = H_u_v              # [nu, nu]
        self.h_u_v = h_u_v              # [nu, 1]
        
        self.n_cstrt = self.K * sum([h_u_v.shape[0] for h_u_v in self.h_u_v]) + \
            (self.K + 1) * self.h_x.shape[0]
        
        # Cost matrices
        self.Q_v = Q_v if Q_v is not None\
            else [np.zeros((self.nx, self.nx))]*self.Nv           # [Nv, nx, nx]
        self.R_v = R_v if R_v is not None\
            else [np.zeros((nu_v, nu_v)) for nu_v in self.nu_v]   # [Nv, nu_v, nu_v]
        self.M_v = M_v if M_v is not None\
            else [np.zeros((self.nx, nu_v)) for nu_v in self.nu_v]# [Nv, nx, nu_v]
        self.L_v = L_v if L_v is not None\
            else [np.zeros((1, nu_v)) for nu_v in self.nu_v]      # [Nv, 1, nu_v]
        
        self._compute_transition_matrices()
        self.F, self.f_x, self.f = self._compute_pseudo_gradient()
        self.H, self.h_x_0, self.h = self._compute_constraints()
        
        # Solver
        self.solver = solver
        if self.solver == 'fbrs':
            # Initial guess
            self.u0 = np.zeros((self.K*self.nu,1))
            self.v0 = np.zeros((self.n_cstrt,1))     # Dual
        elif self.solver == 'gurobi':
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.start()
                self.m = gp.Model("matrix1", env)
            self.u = self.m.addMVar(shape=self.K*self.nu, lb=-np.inf,
                                    vtype=gp.GRB.CONTINUOUS, name="U")
        
    
    def step(self, x_0: np.ndarray) -> np.ndarray:
        # Form gradient and constraint based on initial state
        f = sp.csr_array(self.f_x @ x_0 + self.f)
        h = sp.csr_array(self.h_x_0 @ x_0 + self.h)
        # Solve optimization problem
        if self.solver == 'fbrs':
            U_opt, U_dual, _, out = fbrs(self.F.toarray(), f, self.H.toarray(), h.toarray(), self.u0, self.v0)      #TODO: adapt fbrs to keep sparse
            
            # Manage non-convergence cases
            if out['eflag'] == -1:
                # We might be stuck in a local minimum: try re-initializing initial guess
                U_opt, U_dual, _, out = fbrs(self.F.toarray(), f, self.H.toarray(), h.toarray(),
                                             np.zeros_like(self.u0), np.zeros_like(self.v0))
                if out['eflag'] == -1:
                    U_opt, U_dual, _, out = fbrs(self.F.toarray(), f, self.H.toarray(), h.toarray(),
                                                 self.u0, np.zeros_like(self.v0))
                    if out['eflag'] == -1:
                        # Try increasing the steps by upscaling the matrices
                        U_opt, U_dual, _, out = fbrs(self.F.toarray()*10, f*10, self.H.toarray(), h.toarray(),
                                                     np.zeros_like(self.u0), np.zeros_like(self.v0))
                        if out['eflag'] == -1:
                            U_opt, U_dual, _, out = fbrs(self.F.toarray()*0.1, f*0.1, self.H.toarray(), h.toarray(),
                                                     np.zeros_like(self.u0), np.zeros_like(self.v0))
                            if out['eflag'] == -1:
                                raise RuntimeError("[RHG] Error: solver fbrs was unable to converge" +\
                                    " (Maximum number of iterations exceeded)")
            
            # Reset initial guess for faster convergence
            self.u0 = U_opt.copy()
            self.v0 = U_dual.copy()
        
        elif self.solver == 'gurobi':
            self.m.addConstr((self.H @ self.u)[:,np.newaxis] <= h.toarray(), name="H")
            self.m.setObjective(self.u.T @ self.F @ self.u + f.T @ self.u, gp.GRB.MINIMIZE)
            self.m.optimize()
            
            U_opt = self.u.X.copy()
        
        # Reorganize the control vector
        U_opt = np.hstack(
            [U_opt[sum(self.nu_v[:v])*self.K:sum(self.nu_v[:v+1])*self.K]\
                .reshape(self.K,self.nu_v[v]) for v in range(self.n_players)])
        
        # Return optimal control
        return U_opt
        
    
    def _compute_transition_matrices(self):
        # Utils
        A_powers = np.array(
            [np.linalg.matrix_power(self.A, k) for k in range(self.K + 1)])
        AkC = np.einsum('kij,jv->kiv', A_powers, self.C)
        cum_sums_AkC = np.cumsum(AkC[:-1, :, :], axis=0)
        
        # Build substituted dynamics matrices
        self.A_tilde = sp.csr_array(np.vstack(A_powers))
        self.B_v_tilde = self._form_B_v_tilde(A_powers, self.B_v)
        self.C_tilde = np.vstack([np.zeros((1, self.nx, 1)), cum_sums_AkC])\
            .reshape((self.K + 1) * self.nx, 1)
        
    def _compute_pseudo_gradient(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        """ Compute the cost function pseudo-gradient of the form: F@U + f_x @ x_0 + f
        F :  array of shape [nu*K, nu*K]
        f_x: array of shape [nu*K, nx  ]
        f:   array of shape [nu*K, 1   ]
        (/!\\ Assumption: Q symmetric)"""
        F = sp.coo_array((0, self.nu*self.K))
        f_x = sp.coo_array((0, self.nx))
        f = sp.coo_array((0, 1))
        index = 0
        for v in range(self.Nv):
            Q_v_diag = np.kron(np.eye(self.K+1), self.Q_v[v])
            R_v_diag = np.kron(np.eye(self.K), self.R_v[v])
            M_v_diag = np.kron(np.eye(self.K), self.M_v[v])
            M_v_diag = np.vstack([M_v_diag, np.zeros((self.nx, M_v_diag.shape[1]))])
            L_v_stacked = np.tile(self.L_v[v], self.K)
            B_minus_v_tilde = sp.hstack(self.B_v_tilde[:v] + self.B_v_tilde[v+1:])
            # Quadratic cost of input of v (U_v)
            R_vv = self.B_v_tilde[v].T @ Q_v_diag @ self.B_v_tilde[v] +\
                self.B_v_tilde[v].T @ M_v_diag + R_v_diag
            # Coupled cost between inputs of v and -v
            R_v_minus_v = 2 * self.B_v_tilde[v].T @ Q_v_diag @ B_minus_v_tilde +\
                M_v_diag.T @ B_minus_v_tilde
            # Cross cost between state and inputs of v
            M_x_v = 2 * self.A_tilde.T @ Q_v_diag @ self.B_v_tilde[v] +\
                self.A_tilde.T @ M_v_diag
            # Linear term in inputs of v
            L_v = 2 * self.C_tilde.T @ Q_v_diag @ self.B_v_tilde[v] +\
                self.C_tilde.T @ M_v_diag + L_v_stacked
            # Selection matrices to select player v and players -v
            nU_v = self.nu_v[v] * self.K
            S_v = np.pad(np.eye(nU_v),
                         ((0, 0), (index, self.nu * self.K - nU_v - index)))
            S_minus_v = np.vstack([
                np.pad(np.eye(index), ((0, 0), (0, self.nu * self.K - index))),
                (np.pad(np.eye(self.nu * self.K - index - nU_v), ((0, 0), (index + nU_v, 0))))])
            # Construct cost gradient wrt to U_v
            F_rows_v = (R_vv + R_vv.T) @ S_v + R_v_minus_v @ S_minus_v
            # Append it to the pseudo gradient
            F = sp.vstack([F, F_rows_v])
            f_x = sp.vstack([f_x, M_x_v.T])
            f = sp.vstack([f, L_v.T])
            # increase start index
            index += nU_v
        return F.tocsr()*1e3, f_x.tocsr()*1e3, f.tocsr()*1e3
    
    def _compute_constraints(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        """ Compute the constraints of the form: H@U ≤ h_x_0 @ x_0 + h
        H :  array of shape [n_dual, nu*K]
        h_x: array of shape [n_dual, nx  ]
        h:   array of shape [n_dual, 1   ]"""
        # Input constraint
        H = sp.block_diag([sp.kron(sp.identity(self.K), H_u_v) for H_u_v in self.H_u_v])
        h_x_0 = sp.coo_array((H.shape[0], self.nx))
        h = sp.vstack([sp.kron(np.ones((self.K, 1)), h_u_v) for h_u_v in self.h_u_v])
        # Add state constraint
        H_x_diag = sp.kron(sp.identity(self.K+1), self.H_x).tocsr()
        h_x_tile = sp.kron(np.ones((self.K+1, 1)), self.h_x).tocsr()
        H = sp.vstack([H, H_x_diag @ sp.hstack(self.B_v_tilde)])
        # Shift due to the substitution of the dynamics
        h_x_0 = sp.vstack([h_x_0,
                            -H_x_diag @ self.A_tilde])
        h = sp.vstack([h,
                        h_x_tile - H_x_diag @ self.C_tilde])
        return H.tocsr(), h_x_0.tocsr(), h
    
        
    def _form_B_v_tilde(self, A_powers: np.ndarray, B_v: List[np.ndarray]) -> List[sp.csr_array]:
        B_tilde_v = []
        # Loop over agents
        for v, B in enumerate(B_v):
            if type(B) != np.ndarray:       # TODO: change
                B = B.toarray()
            B_tilde_v.append(self._form_B_tilde(A_powers, B))
        
        return B_tilde_v

    def _form_B_tilde(self, A_powers: np.ndarray, B: np.ndarray) -> sp.csr_array:
        # Compute all needed A^k B: shape (K, nx, nu)
        AkB = np.einsum('kij,jv->kiv', A_powers, B)
        # Lower triangular indices
        i_idx, j_idx = np.tril_indices(self.K)

        # Fetch the right A^k B for each block
        blocks = AkB[i_idx - j_idx]  # shape (K!, nx, nu)

        # Step 4: Place blocks into final array
        nu_v = B.shape[1]
        result = sp.lil_array(((self.K+1) * self.nx, self.K * nu_v))
        for idx, (i, j) in enumerate(zip(i_idx, j_idx)):
            result[(i+1)*self.nx:(i+2)*self.nx, j*nu_v:(j+1)*nu_v] = blocks[idx]
        
        return result.tocsr()