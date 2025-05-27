from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from .LTI_LQRc_RHG import LTI_LQRc_RHG
from ..Solver.optimizer_fbrs import fbrs

class LTI_LQRc_RHG_mixed_state(LTI_LQRc_RHG):
    """
    Receding Horizon Game optimization problem with LTI dynamics and quadratic
    cost function with crossed terms between state and inputs.
    
    The state is composed of a part x_c common to all players, and a specific
    part x_s different for each player: x = [x_c, x_s].
    
    Solve the following problem for all player v:
    
    min_{x,u} ∑ ( x^T Q x + uv^T R uv + x^T M uv + L uv)
        s.t x_c^+ = A_c @ x + B_c^v @ u^v + B^-v @ u^-v + C_c
            x_s_v^+ = A_s^v @ x_c + B_s^v @ u^v + C_s^v
            H_x @ x < h_x
            H_u @ u < h_u
            x_0 = x
    
    (/!\\ Q symmetric)
    """
    
    def __init__(self, param_file: str,
                 A_c: np.ndarray, B_c_v: List[np.ndarray], C_c: np.ndarray,
                 A_s_v: List[np.ndarray], B_s_v: List[np.ndarray], C_s_v: np.ndarray,
                 H_x_c: np.ndarray, h_x_c: np.ndarray, H_u_v: List[np.ndarray], h_u_v: List[np.ndarray],
                 H_x_s_v: List[np.ndarray], h_x_s_v: List[np.ndarray],
                 Q_v: List[np.ndarray] = None, R_v: List[np.ndarray] = None,
                 M_v: List[np.ndarray] = None, L_v: List[np.ndarray] = None,
                 L_x_v: List[np.ndarray] = None,
                 solver: str = 'fbrs'):                       # TODO: modify to work with sparse
        
        # Specific-part variable dimensions
        self.nx_s_v = [A.shape[0] for A in A_s_v]
        
        # Spread the cost matrices into common and specific parts
        nxc = A_c.shape[0]
        nu_v = [B.shape[1] for B in B_c_v]
        Q_cc_v = [Q[:nxc, :nxc] for Q in Q_v] if Q_v is not None else None
        M_c_v = [M[:nxc,:] for M in M_v] if M_v is not None else None
        self.Q_ss_v = [Q[nxc:, nxc:] for Q in Q_v] if Q_v is not None\
            else [np.zeros((nx_s_v, nx_s_v)) for nx_s_v in self.nx_s_v]
        self.Q_sc_v = [Q[nxc:, :nxc] + Q[:nxc, nxc:].T for Q in Q_v] if Q_v is not None\
            else [np.zeros((nx_s_v, nxc)) for nx_s_v in self.nx_s_v]
        self.M_s_v = [M[nxc:,:] for M in M_v] if M_v is not None\
            else [np.zeros(dim) for dim in zip(self.nx_s_v, nu_v)]
        
        # Specific-part dynamics matrices
        self.A_s_v = A_s_v              # [Nv, nx_s, nx_v]
        self.B_s_v = B_s_v              # [Nv, nx_s, nu_v]
        self.C_s_v = C_s_v              # [Nv, nx_s, 1]
        
        # Specific-part constraint matrices
        self.H_x_s_v = H_x_s_v          # [Nv, nx_s, nx_s]
        self.h_x_s_v = h_x_s_v          # [Nv, nx_s, 1]
        
        # Initialize the common part
        super().__init__(param_file,
                         A_c, B_c_v, C_c,
                         H_x_c, h_x_c, H_u_v, h_u_v,
                         Q_cc_v, R_v, M_c_v, L_v,
                         solver)
        
    
    def _compute_transition_matrices(self):
        # Common part transition matrices
        super()._compute_transition_matrices()
        
        # Specific part transition matrices
        self.A_ss_v_tilde, self.A_sc_v_tilde = [], []
        self.B_s_v_tilde, self.B_s_minus_v_tilde = [], []
        self.C_s_v_tilde = []
        for A_s, B_s, C_s, v in zip(self.A_s_v, self.B_s_v, self.C_s_v, range(self.n_players)):
            B_c_v_tilde = self.B_v_tilde[v]
            B_c_minus_v_tilde = sp.hstack(self.B_v_tilde[:v] + self.B_v_tilde[v+1:])
            # Split A_s into common and specific part
            A_sc = A_s[:, :self.nx]
            A_ss = A_s[:, self.nx:]
            # Compute aggregated matrices treating x_c as an inputs
            A_ss_v_tilde, [B_s_v_tilde, A_sc_v_tilde], C_s_v_tilde = \
                self._aggregate_dynamics(A_ss, [B_s, A_sc], C_s)
            A_sc_v_tilde = sp.hstack([A_sc_v_tilde, sp.csr_array((A_sc_v_tilde.shape[0], self.nx))])
            # Use previously computed aggregated common matrices to substitute x_c
            self.A_ss_v_tilde.append(A_ss_v_tilde)
            self.A_sc_v_tilde.append(A_sc_v_tilde @ self.A_tilde)
            self.B_s_v_tilde.append(A_sc_v_tilde @ B_c_v_tilde + B_s_v_tilde)
            self.B_s_minus_v_tilde.append(A_sc_v_tilde @ B_c_minus_v_tilde)
            self.C_s_v_tilde.append(A_sc_v_tilde @ self.C_tilde + C_s_v_tilde)
        
        
    def _compute_pseudo_gradient(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        """ Compute the cost function pseudo-gradient of the form: F@U + f_x @ x_0 + f
        F :  array of shape [nu*K, nu*K]
        f_x: array of shape [nu*K, nx  ]
        f:   array of shape [nu*K, 1   ]
        (/!\\ Assumption: Q symmetric)"""
        # Compute part of the cost related to the common state
        F, f_x, f = super()._compute_pseudo_gradient()
        # Add part related to the specific state
        F, f_x, f = F.tolil(), f_x.tolil(), f.tolil()
        f_x_s = sp.coo_array((0, 0))
        index = 0
        for v in range(self.Nv):
            # Compute cost matrices with state x^v = [x_c; x_s^v]
            A_tilde = sp.block_array([[self.A_tilde, None],
                                      [self.A_sc_v_tilde[v], self.A_ss_v_tilde[v]]])
            B_v_tilde = sp.vstack([self.B_v_tilde[v], self.B_s_v_tilde[v]])
            B_minus_v_tilde = sp.vstack([self.B_minus_v_tilde[v], self.B_s_minus_v_tilde[v]])
            C_tilde = sp.vstack([self.C_tilde, self.C_s_v_tilde[v]])
            Q_sc_v_tilde = sp.kron(np.eye(self.K+1), self.Q_sc_v[v], format='csr')
            Q_v_tilde = sp.block_array(
                [[sp.csr_array(((self.K+1)*self.nx, (self.K+1)*self.nx)), 0.5*Q_sc_v_tilde.T],
                 [0.5*Q_sc_v_tilde,
                  sp.kron(np.eye(self.K+1), self.Q_ss_v[v], format='csr')]])
            M_tilde = sp.vstack([
                sp.csr_array(((self.K+1)*self.nx, self.K*self.nu_v[v])),
                sp.kron(np.eye(self.K), self.M_s_v[v], format='csr'),
                sp.csr_array((self.nx_s_v[v], self.K*self.nu_v[v]))])
            R_tilde = sp.csr_array((self.K*self.nu_v[v], self.K*self.nu_v[v]))
            L_tilde = np.zeros((1, self.K*self.nu_v[v]))
            R_vv, R_v_minus_v, M_x_v, L_v_tilde = \
                self._aggregate_cost(
                    Q_v_tilde, R_tilde, M_tilde, L_tilde,
                    A_tilde, B_v_tilde, B_minus_v_tilde, C_tilde)
            # Decompose these matrices to extract parts meaningful for the gradient
            cross_xc_uv = M_x_v[:self.nx, :]
            cross_xs_uv = M_x_v[self.nx:, :]
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
            F[index:index+nU_v, :] += F_rows_v
            f_x[index:index+nU_v, :] += cross_xc_uv.T
            f_x_s = sp.block_diag([f_x_s, cross_xs_uv.T])
            f[index:index+nU_v, :] += L_v_tilde.T
            # increase start index
            index += nU_v
            
            # Update cost matrices (to check nash equilibrium)
            self.R_vv[v] += R_vv
            self.R_v_minus_v[v] += R_v_minus_v
            self.M_x_v[v] += cross_xc_uv
            self.M_x_v[v] = sp.vstack([self.M_x_v[v],
                                       sp.csr_array((sum(self.nx_s_v[:v]), cross_xs_uv.shape[1])),
                                       cross_xs_uv,
                                       sp.csr_array((sum(self.nx_s_v[v+1:]), cross_xs_uv.shape[1])),])
            self.L_v_tilde[v] += L_v_tilde
        
        f_x = sp.hstack([f_x, f_x_s])
        return F.tocsr(), f_x.tocsr(), f.tocsr()
    
    
    def _compute_constraints(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        # Constraints on common state
        H, h_x_0, h = super()._compute_constraints()
        # Add constraints on specific state
        if self.H_x_s_v == None:
            h_x_0 = sp.hstack([h_x_0, sp.csr_array((h_x_0.shape[0], sum(self.nx_s_v)))])
        else:
            raise NotImplementedError("Constraint on specific state not yet implemented") # TODO
        return H.tocsr(), h_x_0.tocsr(), h
    
    def _apply_commands(self, x_0, U):
        return super()._apply_commands(x_0[:self.nx,:], U)