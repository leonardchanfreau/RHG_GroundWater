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
    
    min_{x,u} ∑ ( x^T Q x + uv^T R uv + u^-v^T R^- uv + x^T M uv + L uv)
        s.t x^+ = A @ x + B^v @ u^v + B^-v @ u^-v + C
            H_x @ x < h_x
            H_u @ u < h_u
            H_xu_x @ x + H_xu_u @ u < h_xu
            x_0 = x
            TODO: for now all player have same constraints on state, where in
            fact well depth constraints only concern the players who owns it
            -> not sure how you manage coupled constraints here
    
    (/!\\ Q symmetric)
    """
    
    def __init__(self, param_file: str,
                 A: np.ndarray, B_v: List[np.ndarray], C: np.ndarray,
                 H_x: np.ndarray, h_x: np.ndarray, H_u_v: List[np.ndarray],
                 h_u_v: List[np.ndarray], H_xu_x: List[np.ndarray],
                 H_xu_u: List[np.ndarray], h_xu: List[np.ndarray],
                 Q_v: List[np.ndarray] = None, R_v: List[np.ndarray] = None,
                 R_v_minus: List[np.ndarray] = None,
                 M_v: List[np.ndarray] = None, L_v: List[np.ndarray] = None,
                 solver: str = 'fbrs',
                 overwrite_time_horizon: int = None):                       # TODO: modify to work with sparse
        
        # Initialize general game controller
        super().__init__(param_file, nu_v = [B.shape[1] for B in B_v],
                         overwrite_time_horizon=overwrite_time_horizon)
        
        # Variable dimensions
        self.nx = A.shape[0]
        self.nu_minus_v = [self.nu - nu_v for nu_v in self.nu_v]
        
        # Dynamics matrices
        self.A = A                      # [nx, nx]
        self.B_v = B_v                  # [Nv, nx, nu_v]
        self.C = C                      # [nx, 1]
        
        #Constraint matrices
        self.H_x = H_x                  # [nx, nx]
        self.h_x = h_x                  # [nx, 1]
        self.H_u_v = H_u_v              # [nu, nu]
        self.h_u_v = h_u_v              # [nu, 1]
        self.H_xu_x = H_xu_x            # [nx, nx]
        self.H_xu_u = H_xu_u            # [nx, nu]
        self.h_xu = h_xu                # [nx, 1]
        
        # Cost matrices
        self.Q_v = Q_v if Q_v is not None\
            else [np.zeros((self.nx, self.nx))]*self.Nv               # [Nv, nx, nx]
        self.R_v = R_v if R_v is not None\
            else [np.zeros((nu_v, nu_v)) for nu_v in self.nu_v]       # [Nv, nu_v, nu_v]
        self.R_v_minus = R_v_minus if R_v_minus is not None\
            else [np.zeros((nu_minus_v, nu_v)) for nu_v, nu_minus_v
                  in zip(self.nu_v, self.nu_minus_v)]                 # [Nv, nu_minus_v, nu_v]
        self.M_v = M_v if M_v is not None\
            else [np.zeros((self.nx, nu_v)) for nu_v in self.nu_v]    # [Nv, nx, nu_v]
        self.L_v = L_v if L_v is not None\
            else [np.zeros((1, nu_v)) for nu_v in self.nu_v]          # [Nv, 1, nu_v]
        
        self._compute_transition_matrices()
        self.F, self.f_x, self.f = self._compute_pseudo_gradient()
        self.F, self.f_x, self.f = self.F*1e3, self.f_x*1e3, self.f*1e3
        self.H, self.h_x_0, self.h = self._compute_constraints()
        
        self.n_cstrt = self.H.shape[0]
        
        # Solver
        self.solver = solver
        if self.solver == 'fbrs':
            # Initial guess
            self.u0 = np.zeros((self.K*self.nu,1))
            self.v0 = np.zeros((self.n_cstrt,1))     # Dual
        else:
            raise NotImplementedError(f"[RHG] Error: solver {self.solver} is not"+
                                      "yet available.")
        
    
    def _compute_transition_matrices(self):
        self.A_tilde, self.B_v_tilde, self.C_tilde = \
            self._aggregate_dynamics(self.A, self.B_v, self.C)
        self.B_minus_v_tilde =[
            sp.hstack(self.B_v_tilde[:v] + self.B_v_tilde[v+1:])
            for v in range(self.Nv)]
    
    def _aggregate_dynamics(self, A: np.ndarray, B_v: List[np.ndarray], C: np.ndarray):
        # Utils
        A_powers = np.array(
            [np.linalg.matrix_power(A, k) for k in range(self.K + 1)])
        AkC = np.einsum('kij,jv->kiv', A_powers, C)
        cum_sums_AkC = np.cumsum(AkC[:-1, :, :], axis=0)
        
        # Build substituted dynamics matrices
        A_tilde = sp.csr_array(np.vstack(A_powers))
        B_v_tilde = self._form_B_v_tilde(A_powers, B_v)
        C_tilde = np.vstack([np.zeros((1, A.shape[0], 1)), cum_sums_AkC])\
            .reshape(-1, 1)
        
        return A_tilde, B_v_tilde, C_tilde
        
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
            # Aggregated cost matrices      # TODO write in a better way
            Q_v_diag = sp.kron(np.eye(self.K+1), self.Q_v[v], format='csr')
            R_v_diag = sp.kron(np.eye(self.K), self.R_v[v], format='csr')
            M_v_diag = sp.kron(np.eye(self.K), self.M_v[v], format='csr')
            M_v_diag = sp.vstack([M_v_diag,
                                  np.zeros((self.M_v[v].shape[0], M_v_diag.shape[1]))],
                                  format='csr')
            L_v_stacked = sp.csr_array(np.tile(self.L_v[v], self.K))
            R_v_minus_diag = sp.kron(np.eye(self.K), self.R_v_minus[v], format='csr')
            transition_matrix = self._get_tcol_to_vcol(self.nu_v[:v]+self.nu_v[v+1:])
            R_v_minus_tilde = transition_matrix @ R_v_minus_diag
            R_vv, R_v_minus_v, M_x_v, L_v_tilde = \
                self._aggregate_cost(
                    Q_v_diag, R_v_diag, R_v_minus_tilde, M_v_diag, L_v_stacked,
                    self.A_tilde, self.B_v_tilde[v], self.B_minus_v_tilde[v], self.C_tilde)
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
            f = sp.vstack([f, L_v_tilde.T])
            # increase start index
            index += nU_v
            
            # Store cost matrices (to check nash equilibrium)
            self.R_vv = self.R_vv + [R_vv] \
                if hasattr(self, 'R_vv') else [R_vv]
            self.R_v_minus_v = self.R_v_minus_v + [R_v_minus_v] \
                if hasattr(self, 'R_v_minus_v') else [R_v_minus_v]
            self.M_x_v = self.M_x_v + [M_x_v] \
                if hasattr(self, 'M_x_v') else [M_x_v]
            self.L_v_tilde = self.L_v_tilde + [L_v_tilde] \
                if hasattr(self, 'L_v_tilde') else [L_v_tilde]
            
        return F.tocsr(), f_x.tocsr(), f.tocsr()
    
    def _aggregate_cost(self,
                        Q_v_tilde: np.ndarray, R_v_tilde: np.ndarray,
                        R_v_minus_tilde: np.ndarray,
                        M_v_tilde: np.ndarray, L_v_tilde: np.ndarray,
                        A_tilde: sp.csr_array, B_v_tilde: sp.csr_array,
                        B_minus_v_tilde: sp.csr_array, C_tilde: sp.csr_array) \
                            -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array]:
        # Check symmetry of Q
        assert not (Q_v_tilde != Q_v_tilde.T).sum(), "[RHG] Error: Q should be symmetric"
        # TODO: Quadratic state cost?
        # Quadratic cost of input of v (U_v)
        R_vv = B_v_tilde.T @ Q_v_tilde @ B_v_tilde +\
            B_v_tilde.T @ M_v_tilde + R_v_tilde
        # Coupled cost between inputs of v and -v
        R_v_minus_v = 2 * B_v_tilde.T @ Q_v_tilde @ B_minus_v_tilde +\
            M_v_tilde.T @ B_minus_v_tilde + R_v_minus_tilde.T
        # Cross cost between state and inputs of v
        M_x_v = 2 * A_tilde.T @ Q_v_tilde @ B_v_tilde +\
            A_tilde.T @ M_v_tilde
        # Linear term in inputs of v
        L_v = sp.csr_array(2 * C_tilde.T @ Q_v_tilde @ B_v_tilde +\
            C_tilde.T @ M_v_tilde + L_v_tilde)
        return R_vv, R_v_minus_v, M_x_v, L_v
    
    def _compute_constraints(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        """ Compute the constraints of the form: H@U ≤ h_x_0 @ x_0 + h
        H :  array of shape [n_dual, nu*K]
        h_x: array of shape [n_dual, nx  ]
        h:   array of shape [n_dual, 1   ]"""
        # Input constraint
        H = sp.block_diag([sp.kron(sp.identity(self.K), H_u_v) for H_u_v in self.H_u_v])
        h_x_0 = sp.coo_array((H.shape[0], self.nx))
        h = sp.vstack([sp.kron(np.ones((self.K, 1)), h_u_v) for h_u_v in self.h_u_v])
        if self.H_x is not None:
            # Add state constraint
            H_x_diag = sp.hstack(
                [sp.csr_array((self.H_x.shape[0]*self.K, self.nx)),
                 sp.kron(sp.identity(self.K), self.H_x)], format="csr")
            h_x_tile = sp.kron(np.ones((self.K, 1)), self.h_x).tocsr()
            H = sp.vstack([H, (H_x_diag @ sp.hstack(self.B_v_tilde))])
            # Shift due to the substitution of the dynamics
            h_x_0 = sp.vstack([h_x_0, (-H_x_diag @ self.A_tilde)])
            h = sp.vstack([h, (h_x_tile - H_x_diag @ self.C_tilde)])
        if self.H_xu_x is not None:
            # Add mixed constraint
            H_xu_x_diag = sp.hstack(
                [sp.csr_array((self.H_xu_x.shape[0]*self.K, self.nx)),
                 sp.kron(sp.identity(self.K), self.H_xu_x)], format="csr")
            H_xu_u_diag = sp.kron(sp.identity(self.K), self.H_xu_u).tocsr()
            h_xu_tile = sp.kron(np.ones((self.K, 1)), self.h_xu).tocsr()
            # Substitute the dynamics
            H = sp.vstack([H, (H_xu_x_diag @ sp.hstack(self.B_v_tilde)) + H_xu_u_diag])
            h_x_0 = sp.vstack([h_x_0, (-H_xu_x_diag @ self.A_tilde)])
            h = sp.vstack([h, (h_xu_tile - H_xu_x_diag @ self.C_tilde)])
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
        nx = B.shape[0]
        result = sp.lil_array(((self.K+1) * nx, self.K * nu_v))
        for idx, (i, j) in enumerate(zip(i_idx, j_idx)):
            result[(i+1)*nx:(i+2)*nx, j*nu_v:(j+1)*nu_v] = blocks[idx]
        
        return result.tocsr()
    
    def _get_tcol_to_vcol(self, nx_v: np.ndarray):
        """ Return the transition matrix T allowing to reorganize a vector of the
        form col_t(col_v(x_t,v)) into a vector of the form col_v(col_t(x_t,v)).
        col_v(col_t(x_t,v)) = T @ col_t(col_v(x_t,v))"""
        # Matrices allowing to individually select the (x_t,v)_v in col_v(x_t,v)
        get_v = [np.roll(
            np.hstack([np.eye(nx_v[v]), np.zeros((nx_v[v], sum(nx_v)-nx_v[v]))]),
            (0, sum(nx_v[:v]))) for v in range(len(nx_v))]
        # Reorganize those vectors
        return np.vstack([np.kron(np.eye(self.K), xv) for xv in get_v])
    
    def _solve_problem(self, x_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Form gradient and constraint based on initial state
        f = sp.csr_array(self.f_x @ x_0 + self.f)
        h = sp.csr_array(self.h_x_0 @ x_0 + self.h)
        # Call solver
        if self.solver == 'fbrs':
            U_opt, U_dual, _, out = fbrs(self.F.toarray(), f, self.H.toarray(), h.toarray(), self.u0, self.v0)      #TODO: adapt fbrs to keep sparse
            
            # Manage non-convergence cases
            if out['eflag'] == -1:
                raise RuntimeError("[RHG] Error: solver fbrs was unable to converge" +\
                                    " (Maximum number of iterations exceeded)")
            
            # Reset initial guess for future faster convergence
            self.u0 = U_opt.copy()
            self.v0 = U_dual.copy()
            
        return U_opt, U_dual

    def _apply_commands(self, x_0: np.ndarray, U: np.ndarray) -> np.ndarray:
        x_opt = self.A_tilde @ x_0 + sp.hstack(self.B_v_tilde) @ U + self.C_tilde
        return x_opt.reshape(self.K+1, self.nx).T
    
    def _check_nash(self, U: np.ndarray, x0: np.ndarray):
        # Form constraint based on initial state
        h = sp.csr_array(self.h_x_0 @ x0 + self.h)
        # Start solver
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model("matrix1", env)
        # Separate every agent's decision variable on its constraints
        U_v_splitted = np.split(U, np.cumsum(self.nu_v)[:-1], axis=1)
        H_v = np.split(self.H.toarray(), np.cumsum(self.nu_v)[:-1]*self.K, axis=1)
        for v, nu_v in enumerate(self.nu_v):
            # Define variable and parameters
            u_v = m.addMVar(shape=self.K*nu_v, lb=-np.inf,
                                 vtype=gp.GRB.CONTINUOUS, name="u_v")
            u_minus_v = np.concatenate([U_block.reshape(-1, 1)
                                        for p, U_block in enumerate(U_v_splitted)
                                        if p != v])
            # Set individual cost
            cost = u_v.T @ self.R_vv[v] @ u_v + u_v.T @ self.R_v_minus_v[v] @ u_minus_v + x0.T @ self.M_x_v[v] @ u_v + self.L_v_tilde[v] @ u_v
            m.setObjective(cost, gp.GRB.MINIMIZE)
            # Set constraints
            h_v = h - np.hstack([H_v[p] for p in range(self.Nv) if p!=v]) @ u_minus_v
            cstrt = (H_v[v] @ u_v)[:,np.newaxis] <= h_v
            m.addConstr(cstrt, name="H")
            # Compare results
            m.optimize()
            u_v_r = U_v_splitted[v].reshape(-1, 1)
            real_cost = u_v_r.T @ self.R_vv[v] @ u_v_r + u_v_r.T @ self.R_v_minus_v[v] @ u_minus_v + x0.T @ self.M_x_v[v] @ u_v_r + self.L_v_tilde[v] @ u_v_r
            if (m.ObjVal - real_cost)/abs(real_cost) <= - 1e-2:
                return False
        return True