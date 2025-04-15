from typing import List, Tuple
import numpy as np
import scipy.sparse as sp

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
    
    def __init__(self, K:int, nx: int, nu: int, Nv: int, nu_v: List[int],
                 A: np.ndarray, B_v: List[np.ndarray], B_minus_v: List[np.ndarray], C: np.ndarray,
                 Q_v: List[np.ndarray], R_v: List[np.ndarray], M_v: List[np.ndarray], L_v: List[np.ndarray],
                 H_x: np.ndarray, h_x: np.ndarray, H_u: np.ndarray, h_u: np.ndarray):
        super().__init__(K, nx, nu, Nv, nu_v)
        # Dynamics matrices
        self.A = A                      # [nx, nx]
        self.B_v = B_v                  # [Nv, nx, nu_v]
        self.B_minus_v = B_minus_v      # [Nv, nx, (nu-nu_v)]
        self.C = C                      # [nx, 1]
        
        # Cost matrices
        self.Q_v = Q_v                  # [Nv, nx, nx]
        self.R_v = R_v                  # [Nv, nu_v, nu_v]
        self.M_v = M_v                  # [Nv, nx, nu_v]
        self.L_v = L_v                  # [Nv, 1, nu_v]
        
        #Constraint matrices
        self.H_x = H_x                  # [nx, nx]
        self.h_x = h_x                  # [nx, 1]
        self.H_u = H_u                  # [nu, nu]
        self.h_u = h_u                  # [nu, 1]
        
        self._compute_substituted_cost()
        self.F = None
        self.H = None
        
    
    def step(self, x_0):
        # Compute pseudo gradient and constraint matrices and solve the null-point problem
        self.F, f = self._compute_pseudo_gradient(x_0)
        self.H, h = self._compute_constraints(x_0)
        u0 = np.zeros((self.K*self.nu,1))
        v0 = np.zeros((h.shape[0],1))     # Dual
        U_opt, U_dual, _, out = fbrs(self.F, f, self.H.toarray(), h.toarray(), u0, v0)      #TODO: adapt fbrs to keep sparse
        
        # Manage non-convergence cases
        if out['eflag'] == -1:
            raise RuntimeError("[RHG] Error: solver fbrs was unable to converge \
                (Maximum number of iterations exceeded)")
        
        # Return optimal control
        return U_opt.reshape((self.K, self.nu))
        
        
    def _compute_pseudo_gradient(self, x_0:np.ndarray) -> Tuple[sp.lil_array, sp.lil_array]:
        """ Compute the cost function pseudo-gradient of the form: F@U + f(x_0) """
        # Compute F only one time
        F = self.F
        if self.F is None:
            F = sp.lil_array((self.nu*self.K, self.nu*self.K))
            index = 0
            for v in range(self.Nv):
                F[index:index+(self.nu_v[v]*self.K),:] = \
                    sp.hstack([self.tilde_R_minus_v_v[v].T[:, :index],
                            self.tilde_R_v_v[v] + self.tilde_R_v_v[v].T,
                            self.tilde_R_minus_v_v[v].T[:, index:]])       # TODO: check that the u_v are in the correct order
                index += self.nu_v[v]*self.K
        # f depends on x_0 and has to be recomputed every time
        f = sp.lil_array((self.nu*self.K, 1))
        index = 0
        for v in range(self.Nv):
            f[index:index+(self.nu_v[v]*self.K),:] = \
                self.tilde_M_x_u_v[v].T @ x_0 + \
                    sp.kron(np.ones((1,self.K)), self.L_v[v]).T
            index += self.nu_v[v]*self.K
        return F, f
    
    def _compute_constraints(self, x_0:np.ndarray) -> Tuple[sp.lil_array, sp.lil_array]:
        H_x_diag = sp.kron(sp.identity(self.K+1), self.H_x)
        # Compute H only one time
        H = self.H
        if self.H is None:
            H = sp.vstack([
                sp.kron(sp.identity(self.K), self.H_u),
                H_x_diag @ self.B_tilde
            ])
        # h depends on x_0 and has to be recomputed every time
        h = sp.vstack([
            sp.kron(np.ones((self.K, 1)), self.h_u),
            sp.kron(np.ones((self.K+1, 1)), self.h_x) - \
                H_x_diag @ self.A_tilde @ x_0 - H_x_diag @ self.C_tilde,
        ])
        return H, h
    
    def _compute_substituted_cost(self):
        # Utils
        A_powers = np.array(
            [np.linalg.matrix_power(self.A, k) for k in range(self.K + 1)])
        AkC = np.einsum('kij,jv->kiv', A_powers, self.C)
        cum_sums_AkC = np.cumsum(AkC[:-1, :, :], axis=0)
        
        # Build substituted dynamics matrices
        self.A_tilde = np.vstack(A_powers)
        B_v_tilde = self._form_B_v_tilde(A_powers, self.B_v)
        B_minus_v_tilde = self._form_B_v_tilde(A_powers, self.B_minus_v)
        self.C_tilde = np.vstack([np.zeros((1, self.nx, 1)), cum_sums_AkC])\
            .reshape((self.K + 1) * self.nx, 1)
        self.B_tilde = self._form_B_tilde(A_powers, np.hstack(self.B_v)) # Used in the constraints
            
        # Build substituted cost matrices (/!\ Q symmetric)
        self.tilde_Q_v = []
        self.tilde_R_v_v, self.tilde_R_minus_v_minus_v, self.tilde_R_minus_v_v = [], [], []
        self.tilde_M_x_u_v, self.tilde_M_x_u_minus_v = [], []
        self.tilde_L_u_v,  self.tilde_L_u_minus_v, self.tilde_L_x_v = [], [], []
        self.tilde_cst = []
        for v in range(self.Nv):
            diag_Q_v = sp.lil_array(((self.K+1) * self.nx, (self.K+1) * self.nx))
            diag_Q_v[:self.K*self.nx, :self.K*self.nx] = sp.kron(sp.identity(self.K), self.Q_v[v])
            diag_M_v = sp.vstack([
                sp.kron(sp.identity(self.K), self.M_v[v]),
                sp.lil_array((self.nx, self.K*self.nu_v[v]))])
            # Initial state
            self.tilde_Q_v.append(
                self.A_tilde.T @ diag_Q_v @ self.A_tilde)
            # Inputs
            self.tilde_R_v_v.append(
                B_v_tilde[v].T @ diag_Q_v @ B_v_tilde[v] + \
                    B_v_tilde[v].T @ diag_M_v + \
                        sp.kron(sp.identity(self.K), self.R_v[v]))
            self.tilde_R_minus_v_minus_v.append(
                B_minus_v_tilde[v].T @ diag_Q_v @ B_minus_v_tilde[v]) #+ sp.kron(sp.identity(self.K), self.R_minus_v[v]))
            self.tilde_R_minus_v_v.append(
                2 * B_minus_v_tilde[v].T @ diag_Q_v @ B_v_tilde[v] + \
                    B_minus_v_tilde[v].T @ diag_M_v)
            # Cross-terms
            self.tilde_M_x_u_v.append(
                2 * self.A_tilde.T @ diag_Q_v @ B_v_tilde[v] + \
                    self.A_tilde.T @ diag_M_v)
            self.tilde_M_x_u_minus_v.append(
                2 * self.A_tilde.T @ diag_Q_v @ B_minus_v_tilde[v]) #+ sp.kron(sp.identity(self.K), self.M_minus_v[v]))
            # Linear terms
            self.tilde_L_u_v.append(
                2 * self.C_tilde.T @ diag_Q_v @ B_v_tilde[v] + \
                    self.C_tilde.T @ diag_M_v + \
                        sp.kron(np.ones((1,self.K)), self.L_v[v]))
            self.tilde_L_u_minus_v.append(
                2 * self.C_tilde.T @ diag_Q_v @ B_minus_v_tilde[v]) #+ sp.kron(sp.identity(self.K), self.L_minus_v[v]))
            self.tilde_L_x_v.append(
                2 * self.C_tilde.T @ diag_Q_v @ self.A_tilde)
            # Constant term
            self.tilde_cst.append(
                self.C_tilde.T @ diag_Q_v @ self.C_tilde)
            
        
    def _form_B_v_tilde(self, A_powers: np.ndarray, B_v: List[np.ndarray]) -> List[sp.lil_array]:
        B_tilde_v = []
        # Loop over agents
        for v, B in enumerate(B_v):
            B_tilde_v.append(self._form_B_tilde(A_powers, B))
        
        return B_tilde_v

    def _form_B_tilde(self, A_powers: np.ndarray, B: np.ndarray) -> sp.lil_array:
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
        
        return result