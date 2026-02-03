from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import gurobipy as gp

from .RHG import RHG
from ..Solver.optimizer_fbrs import fbrs, fbrs_optimized

def sparse_equal(A, B):
    """Efficient CSR sparse matrix equality test."""
    if A.shape != B.shape: 
        return False
    if A.nnz != B.nnz:
        return False
    return (A != B).nnz == 0


def listlist_equal(L1, L2):
    """Efficient equality for List[List[csr_array]]."""
    if len(L1) != len(L2):
        return False
    for r1, r2 in zip(L1, L2):
        if len(r1) != len(r2):
            return False
        for A, B in zip(r1, r2):
            if not sparse_equal(A, B):
                return False
    return True

def to_csr_listlist(L):
    return [[sp.csr_array(M) for M in row] for row in L]

class TimeVaryingCost_LQRc_RHG(RHG):
    """
    Receding Horizon Game optimization problem with LTI dynamics and time-varying
    quadratic cost function with crossed terms between state and inputs.
    
    Solve the following problem for all player v:
    
    min_{x,u} ∑ ( x^T Q_t x + uv^T R_t uv + u^-v^T R^-_t uv + x^T M_t uv + L_t uv)
        s.t x^+ = A @ x + B^v @ u^v + B^-v @ u^-v + C
            H_x @ x < h_x
            H_u @ u < h_u
            H_xu_x @ x + H_xu_u @ u < h_xu
            x_0 = x
    
    (/!\\ Q symmetric)
    """
    
    def __init__(self, param_file: str,
                 A: sp.csr_array, B_v: List[sp.csr_array], C: sp.csr_array,
                 H_x: np.ndarray, h_x: np.ndarray, H_u_v: List[np.ndarray],
                 h_u_v: List[np.ndarray], H_xu_x: List[np.ndarray],
                 H_xu_u: List[np.ndarray], h_xu: List[np.ndarray],
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
        
        self._compute_transition_matrices()
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
            
    
    def update_cost_matrices(self, 
                             Q_v_t: List[List[sp.csr_array]] = None, R_v_t: List[List[sp.csr_array]] = None,
                             R_v_minus_t: List[List[sp.csr_array]] = None,
                             M_v_t: List[List[sp.csr_array]] = None, L_v_t: List[List[sp.csr_array]] = None):
        # Transform cost matrices in the correct format and check if computing the gradient is necessary
        need_recomputation = False
        # --- Q [K, Nv, nx, nx] ---
        if Q_v_t is None:
            # Reset
            new_Q = [[sp.csr_array((self.nx, self.nx)) for _ in range(self.K)]
                    for _ in range(self.Nv)]
            if (not hasattr(self, 'Q_v_t')) or (not listlist_equal(self.Q_v_t, new_Q)):
                need_recomputation = True
            self.Q_v_t = new_Q
        else:
            if not isinstance(Q_v_t[0][0], sp.csr_array):
                Q_v_t = to_csr_listlist(Q_v_t)

            if (not hasattr(self, 'Q_v_t')) or (not listlist_equal(self.Q_v_t, Q_v_t)):
                need_recomputation = True
            self.Q_v_t = Q_v_t
        # --- R_v [K, Nv, nu_v, nu_v] ---
        if R_v_t is None:
            new_R = [[sp.csr_array((nu_v, nu_v)) for _ in range(self.K)]
                    for nu_v in self.nu_v]
            if (not hasattr(self, 'R_v_t')) or (not listlist_equal(self.R_v_t, new_R)):
                need_recomputation = True
            self.R_v_t = new_R
        else:
            if not isinstance(R_v_t[0][0], sp.csr_array):
                R_v_t = to_csr_listlist(R_v_t)

            if (not hasattr(self, 'R_v_t')) or (not listlist_equal(self.R_v_t, R_v_t)):
                need_recomputation = True
            self.R_v_t = R_v_t
        # --- R_v_minus [K, Nv, nu_minus_v, nu_v] ---
        if R_v_minus_t is None:
            new_Rm = [[sp.csr_array((nu_m, nu_v)) for _ in range(self.K)]
                    for nu_v, nu_m in zip(self.nu_v, self.nu_minus_v)]
            if (not hasattr(self, 'R_v_minus_t')) or (not listlist_equal(self.R_v_minus_t, new_Rm)):
                need_recomputation = True
            self.R_v_minus_t = new_Rm
        else:
            if not isinstance(R_v_minus_t[0][0], sp.csr_array):
                R_v_minus_t = to_csr_listlist(R_v_minus_t)

            if (not hasattr(self, 'R_v_minus_t')) or (not listlist_equal(self.R_v_minus_t, R_v_minus_t)):
                need_recomputation = True
            self.R_v_minus_t = R_v_minus_t
        # --- M_v [K, Nv, nx, nu_v] ---
        if M_v_t is None:
            new_M = [[sp.csr_array((nu_m, nu_v)) for _ in range(self.K)]
                    for nu_v, nu_m in zip(self.nu_v, self.nu_minus_v)]
            if (not hasattr(self, 'M_v_t')) or (not listlist_equal(self.M_v_t, new_M)):
                need_recomputation = True
            self.M_v_t = new_M
        else:
            if not isinstance(M_v_t[0][0], sp.csr_array):
                M_v_t = to_csr_listlist(M_v_t)

            if (not hasattr(self, 'M_v_t')) or (not listlist_equal(self.M_v_t, M_v_t)):
                need_recomputation = True
            self.M_v_t = M_v_t
        # --- L_v [K, Nv, 1, nu_v] ---
        if L_v_t is None:
            new_L = [[sp.csr_array((1, nu_v)) for _ in range(self.K)]
                    for nu_v in self.nu_v]
            if (not hasattr(self, 'L_v_t')) or (not listlist_equal(self.L_v_t, new_L)):
                need_recomputation = True
            self.L_v_t = new_L
        else:
            if not isinstance(L_v_t[0][0], sp.csr_array):
                L_v_t = to_csr_listlist(L_v_t)

            if (not hasattr(self, 'L_v_t')) or (not listlist_equal(self.L_v_t, L_v_t)):
                need_recomputation = True
            self.L_v_t = L_v_t
        # Recompute gradient only if something changed
        if need_recomputation:
            self.F, self.f_x, self.f = self._compute_pseudo_gradient()
            self.F, self.f_x, self.f = self.F*1e3, self.f_x*1e3, self.f*1e3     # TODO: remove hardcoding
        
    
    def _compute_transition_matrices(self):
        self.A_tilde, self.B_v_tilde, self.C_tilde = \
            self._aggregate_dynamics(self.A, self.B_v, self.C)
        self.B_minus_v_tilde =[
            sp.hstack(self.B_v_tilde[:v] + self.B_v_tilde[v+1:])
            for v in range(self.Nv)]
    
    def _aggregate_dynamics(self, A: sp.csr_matrix, B_v: List[sp.csr_array], C: sp.csr_array):
        '''Build substituted dynamics matrices.'''
        # Utils
        nx = A.shape[0]
        K = self.K
        # Form A_tilde
        A_powers = [sp.eye(nx, format="csr")]
        for _ in range(K):
            A_powers.append(A_powers[-1] @ A)           #TODO: check if csr
        A_tilde = sp.vstack(A_powers, format="csr")
        # Form C_tilde
        cum = sp.csr_array((nx, C.shape[1]))
        C_blocks = [sp.csr_array((nx, C.shape[1]))]  # first block is zeros
        for k in range(K):
            cum = cum + A_powers[k] @ C
            C_blocks.append(cum)
        C_tilde = sp.vstack(C_blocks, format="csr")
        # Form B_v_tilde for each agent
        B_v_tilde = [self._form_B_tilde_sparse(A_powers, B) for B in B_v]
        return A_tilde, B_v_tilde, C_tilde
        
    def _compute_pseudo_gradient(self) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
        """ Compute the cost function pseudo-gradient of the form: F@U + f_x @ x_0 + f
        F :  array of shape [nu*K, nu*K]
        f_x: array of shape [nu*K, nx  ]
        f:   array of shape [nu*K, 1   ]
        (/!\\ Assumption: Q symmetric)"""
        totalU= self.nu * self.K
        F = sp.lil_matrix((totalU, totalU))
        f_x = sp.lil_matrix((totalU, self.nx))
        f = sp.lil_matrix((totalU, 1))
        index_offsets = np.cumsum(np.r_[0, self.nu_v[:-1]]) * self.K
        I = sp.eye(totalU, format="csr")
        for v in range(self.Nv):
            idx = index_offsets[v]
            # Aggregated cost matrices
            
            Q_v_diag = sp.block_diag(
                [sp.csr_array(self.Q_v_t[v][0])] + self.Q_v_t[v], format='csr')      #TODO
            R_v_diag = sp.block_diag(self.R_v_t[v], format='csr')
            M_v_diag = sp.block_diag(self.M_v_t[v], format='csr')
            M_v_diag = sp.vstack([M_v_diag,
                                  sp.csr_array((self.M_v_t[v][0].shape[0], M_v_diag.shape[1]))],
                                  format='csr')                                                     
            L_v_stacked = sp.hstack(self.L_v_t[v])
            R_v_minus_diag = sp.block_diag(self.R_v_minus_t[v], format='csr')
            transition_matrix = self._get_tcol_to_vcol_faster(self.nu_v[:v]+self.nu_v[v+1:])
            R_v_minus_tilde = transition_matrix @ R_v_minus_diag
            R_vv, R_v_minus_v, M_x_v, L_v_tilde = \
                self._aggregate_cost(
                    Q_v_diag, R_v_diag, R_v_minus_tilde, M_v_diag, L_v_stacked,
                    self.A_tilde, self.B_v_tilde[v], self.B_minus_v_tilde[v], self.C_tilde)
            # Selection matrices to select player v and players -v
            nU_v = self.nu_v[v] * self.K
            rows_v = np.arange(idx, idx + nU_v)
            S_v = I[rows_v]
            rows_minus_v = np.r_[0:idx, idx+nU_v:self.nu*self.K]
            S_minus_v = I[rows_minus_v]
            # Construct cost gradient wrt to U_v
            F_rows_v = (R_vv + R_vv.T) @ S_v + R_v_minus_v @ S_minus_v
            # Append it to the pseudo gradient
            F[rows_v, :] = F_rows_v
            f_x[rows_v, :] = M_x_v.T
            f[rows_v, :] = L_v_tilde.T
            
            # Store cost matrices (to check nash equilibrium) TODO: de-comment
            self.R_vv = self.R_vv + [R_vv] \
                if v != 0 else [R_vv]
            self.R_v_minus_v = self.R_v_minus_v + [R_v_minus_v] \
                if v != 0 else [R_v_minus_v]
            self.M_x_v = self.M_x_v + [M_x_v] \
                if v != 0 else [M_x_v]
            self.L_v_tilde = self.L_v_tilde + [L_v_tilde] \
                if v != 0 else [L_v_tilde]
            
        return F.tocsr(), f_x.tocsr(), f.tocsr()
    
    def _aggregate_cost(self,
                        Q_v_tilde: np.ndarray, R_v_tilde: np.ndarray,
                        R_v_minus_tilde: np.ndarray,
                        M_v_tilde: np.ndarray, L_v_tilde: np.ndarray,
                        A_tilde: sp.csr_array, B_v_tilde: sp.csr_array,
                        B_minus_v_tilde: sp.csr_array, C_tilde: sp.csr_array) \
                            -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array]:
        # Check symmetry of Q
        if (Q_v_tilde - Q_v_tilde.T).nnz != 0:
            raise ValueError("[RHG] Error: Q should be symmetric")
        # TODO: Quadratic state cost?
        # Precompute transposes once
        BvT = B_v_tilde.T
        AT = A_tilde.T
        CT = C_tilde.T
        # Q * X multiplications
        # Stack RHS blocks horizontally
        K = sp.hstack([B_v_tilde,
                    B_minus_v_tilde], format="csr")
        # Single expensive multiplication
        QK = Q_v_tilde @ K

        # Unpack
        nBv = B_v_tilde.shape[1]
        nBminus = B_minus_v_tilde.shape[1]
        QB_v        = QK[:, 0:nBv]
        QB_minus_v  = QK[:, nBv:nBv+nBminus]
        # Quadratic cost of input of v (U_v)
        R_vv =  BvT @ QB_v + BvT @ M_v_tilde + R_v_tilde
        # Coupled cost between inputs of v and -v
        R_v_minus_v = 2 * (BvT @ QB_minus_v) + (M_v_tilde.T @ B_minus_v_tilde) + R_v_minus_tilde.T
        # Cross cost between state and inputs of v
        M_x_v = 2 * (AT @ QB_v) + (AT @ M_v_tilde)
        # Linear term in inputs of v
        L_v =  2 * (CT @ QB_v) + (CT @ M_v_tilde) + L_v_tilde
        
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
            transition_matrix = self._get_tcol_to_vcol(self.nu_v).T     # TODO: use faster if same result
            # Substitute the dynamics
            H = sp.vstack([H, (H_xu_x_diag @ sp.hstack(self.B_v_tilde)) + H_xu_u_diag @ transition_matrix])
            h_x_0 = sp.vstack([h_x_0, (-H_xu_x_diag @ self.A_tilde)])
            h = sp.vstack([h, (h_xu_tile - H_xu_x_diag @ self.C_tilde)])
        return H.tocsr(), h_x_0.tocsr(), h
    
    def _form_B_tilde_sparse(self, A_powers, B: sp.csr_array) -> sp.csr_array:
        """
        Build lower block-triangular B̃ using sparse operations only.
        """
        nx, nu = B.shape
        K = self.K

        data = []
        rows = []
        cols = []

        # For each block row i = 1..K
        for i in range(1, K+1):
            # Blocks j = 0..i-1
            for j in range(i):
                A_power = A_powers[i - j - 1]   # corresponds to A^(i-j-1)
                block = A_power @ B             # sparse product

                coo = block.tocoo()
                rows.extend(coo.row + i*nx)
                cols.extend(coo.col + j*nu)
                data.extend(coo.data)

        B_tilde = sp.coo_matrix(
            (data, (rows, cols)),
            shape=((K+1)*nx, K*nu),
        ).tocsr()

        return B_tilde
    
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

    def _get_tcol_to_vcol_faster(self, nx_v: np.ndarray) -> sp.csr_array:
        """
        Return transition matrix T (csr_array) that maps:
            col_t(col_v(x_t,v)) → col_v(col_t(x_t,v))
        
        Produces a sparse permutation matrix without dense operations.
        """

        K = self.K
        nv = len(nx_v)

        # Total number of columns = sum(nx_v) * K
        total = int(np.sum(nx_v)) * K

        # Build (row, col) index lists for CSR
        rows = []
        cols = []

        # Row offset per v block
        v_offsets = np.cumsum(np.r_[0, nx_v])     # [0, nx_v[0], nx_v[0]+nx_v[1], ...]
        total_per_timestep = int(np.sum(nx_v))

        for v in range(nv):
            nx = int(nx_v[v])

            # column offset inside each time block
            col0 = v_offsets[v]

            # row offset in output (v-major ordering)
            row0 = int(np.sum(nx_v[:v]) * K)

            for t in range(K):
                # column block start = t * total_per_timestep + col0
                col_start = t * total_per_timestep + col0

                # row block start = row0 + t*nx
                row_start = row0 + t * nx

                # For these nx rows, place identity entries:
                rows.extend(range(row_start, row_start + nx))
                cols.extend(range(col_start, col_start + nx))

        data = np.ones(len(rows), dtype=np.float64)

        return sp.csr_array((data, (rows, cols)), shape=(total, total))
    
    def _solve_problem(self, x_0: sp.csr_array) -> Tuple[np.ndarray, np.ndarray]:
        # Form gradient and constraint based on initial state
        f = self.f_x @ x_0 + self.f
        h = self.h_x_0 @ x_0 + self.h
        # Call solver
        if self.solver == 'fbrs':
            U_opt, U_dual, _, out = fbrs(self.F.toarray(), f, self.H.toarray(), h, self.u0, self.v0)      #TODO: adapt fbrs to keep sparse
            # U_opt, U_dual, _, out = fbrs_optimized(self.F, sp.csr_array(f), self.H, sp.csr_array(h), sp.csr_array(self.u0), sp.csr_array(self.v0))      #TODO: adapt fbrs to keep sparse
            
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
    
    def _check_nash(self, U: np.ndarray, x0: np.ndarray, constrt_tol: float = 1e-5, conv_tol = 1e-2):
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
            cstrt = (H_v[v] @ u_v)[:,np.newaxis] <= h_v + constrt_tol
            m.addConstr(cstrt, name="H")
            # Compare results
            m.optimize()
            u_v_r = U_v_splitted[v].reshape(-1, 1)
            real_cost = u_v_r.T @ self.R_vv[v] @ u_v_r + u_v_r.T @ self.R_v_minus_v[v] @ u_minus_v + x0.T @ self.M_x_v[v] @ u_v_r + self.L_v_tilde[v] @ u_v_r
            if (m.ObjVal - real_cost)/(abs(real_cost) + constrt_tol + conv_tol) <= - 1e-2:
                return False
        return True