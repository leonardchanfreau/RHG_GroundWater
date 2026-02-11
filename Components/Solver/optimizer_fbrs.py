#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

  @Author: shall
  @Date:   2023-02-14 17:41:00
  @Last Modified by:   Sophie Hall
  @Last Modified time: 2023-02-14 17:41:00

  fbrs: A regularized and smoothed Fischer-Burmeister Method for 
  solving convex quadratic programs (QPs)
  [x,v,out] = fbrs(qp,x0,v0,opts) attempts to solve
  following problem

  min.  H x + f
  s.t.  Ax <= b
  
  FBRS algorithm for non symmetric H
  Implemented using numpy
  
  
  This method assumes the linear independance constraint qualification
  and strong second order sufficient conditions hold at the solution
  but often converges anyways even if they do not
  
  Inputs: 
  qp: A structure with the following fields
 	H: n x n Symmetric positive semidefinite Hessian matrix
 	f: n x 1 Gradient vector
 	A: q x n Constraint Jacobian
 	b: q x 1 Constraint vector
  x0: n x 1 Primal initial guess, use 0 if unsure
  v0: q x 1 Dual initial guess, use 0 if unsure
  opts: A structure containing any of the following fields
 	tol: Absolute tolerance
 	rtol: Relative tolerance
 	beta: backtracking linesearch parameter 
 	eta: sufficient decrease parameter
 	delta: Regularization parameter
 	lsmax: maximum number of linesearch iterations
 	max_newton_iters: maximum number of newton iterations
 	veps: smoothing parameter
 	use_nr: true to use the natural residual as a stopping criteria
  
  A default value will be used for any missing fields

  Outputs:
  x: n x 1 Primal solution
  v: q x 1 Dual solution
  out: Structure with the following fields
  prox_iters: number of proximal iterations taken
  newton_iters: total number of newton iterations
  res:  KKT residual
  eflag: Exit flag
 		 0: success
  		-1: Maximum number of iterations exceeded

"""

# import numpy as np
# from scipy import linalg

################################################################################
################################################################################

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import linalg

# Vectorized dfb: input u, v, veps are 1-d arrays of length q
def dfb_vec(u, v, veps):
    # u, v, veps are 1D arrays
    r = np.sqrt(u*u + v*v + veps*veps)
    # mask where r is large enough to divide safely
    tol = 100 * np.finfo(float).eps
    mask = (r > tol).toarray()
    gamma = sp.csr_array((u.shape[0], u.shape[1]))
    mu = sp.csr_array((u.shape[0], u.shape[1]))
    gamma[mask] = 1.0 - u[mask] / r[mask]
    mu[mask]    = 1.0 - v[mask] / r[mask]
    # generalized Jacobian value near zero
    val = 1.0 - 1.0/np.sqrt(2.0)
    gamma[~mask] = val
    mu[~mask]    = val
    return gamma, mu

def fbrs_optimized(H, f, Xi_mat, xi_vec, u0, v0,
                   tol=1e-8, rtol=1e-10, eta=1e-10, beta=0.7,
                   delta_init=np.sqrt(2.2204e-16), lsmax=10,
                   max_newton_iters=1000, use_nr=False,
                   cg_tol=1e-6, cg_maxiter=None):
    """
    Optimized FBRS:
      - Xi_mat may be dense (ndarray) or sparse (scipy.sparse)
      - H may be dense or sparse
      - solves K du = r via cg using LinearOperator for Kx = Hx + A.T (D (A x))
    Returns (u, v, E, out)
    """
    # shapes and ensure 1D arrays
    n = f.shape[0]
    q = xi_vec.size
    u = u0.copy().reshape(-1,1)
    v = v0.copy().reshape(-1,1)
    y = xi_vec - (Xi_mat @ u)   # works with sparse or dense
    f = f.reshape(-1,1)

    delta = delta_init
    veps = (tol/(2*np.sqrt(q))) * sp.csr_array(np.ones((v.shape[0], v.shape[1])))
    E = 1e10
    E0 = E
    newton_iters = 0
    out = {'eflag': -1, 'solver': 'fbrs'}

    # helper: check whether Xi_mat is sparse
    A_sparse = sp.issparse(Xi_mat)
    if A_sparse:
        A = Xi_mat.tocsr()
    else:
        A = np.asarray(Xi_mat)

    # workspace arrays avoid reallocations
    # main loop
    for j in range(max_newton_iters):
        delta = min(delta, E)

        # compute dfb (vectorized)
        gamma, mu = dfb_vec(y, v, veps)   # 1D arrays length q
        d = mu + delta*sp.csr_array(np.ones((mu.shape[0],mu.shape[1])))  # 1D

        # stationarity and complementarity residuals
        rs = -( (H @ u) + f + (Xi_mat.T @ v) )
        rc = - ( (y + v) - np.sqrt(y*y + v*v + veps*veps) )  # fb(y,v,veps)

        # combined residual
        F = sp.vstack([rs, rc])
        if use_nr:
            E = 1.0
        else:
            E = sp.linalg.norm(F)
        if j == 0:
            E0 = E
        if E <= tol + E0*rtol:
            out['eflag'] = 0
            break

        # construct operator for K = H + A.T * diag(gamma/d) * A
        w = gamma / d   # 1D length q

        # Define matvec for Kx without forming K dense:
        # Kx = H*x + A.T @ (w * (A @ x))
        def K_matvec(x):
            # x as 1D array
            Hx = H @ x
            if A_sparse:
                Ax = A.dot(x)                # length q
                WAx = w * Ax
                ATAx = A.T.dot(WAx)         # length n
            else:
                Ax = A.dot(x)
                ATAx = A.T.dot(w * Ax)
            return Hx + ATAx

        K_linop = spla.LinearOperator((n, n), matvec=K_matvec, dtype=float)

        # build right-hand side r = rs - Xi.T @ (rc/d)
        r_rhs = rs - (Xi_mat.T @ (rc / d))

        # preconditioner: Jacobi (diag of K)
        # diag(K) = diag(H) + sum_j w_j * A_j^2 where A_j are rows
        if sp.issparse(H):
            H_diag = H.diagonal()
        else:
            H_diag = np.diag(H) if H.ndim == 2 else np.asarray(H).diagonal()
        if A_sparse:
            # compute row-wise squared sums: for CSR, rows are fast
            # diag contribution from A.T D A = sum_j w_j * A[j,:]^2
            # compute s = (A.multiply(A)).T @ w  (n-vector)
            s = (A.multiply(A)).T.dot(w)
            diagK = H_diag + s
        else:
            s = (A**2).T.dot(w)
            diagK = H_diag + s

        # avoid very small diag entries
        diagK_safe = np.where(np.abs(diagK) > 0, diagK, 1.0)
        M_inv = 1.0 / diagK_safe

        def precond(r):
            return M_inv * r

        M_linop = spla.LinearOperator((n, n), matvec=precond, dtype=float)

        # solve K du = r_rhs with cg (K is symmetric if H symmetric)
        # choose cg_maxiter default:
        if cg_maxiter is None:
            cg_maxiter_local = min(n, 2000)
        else:
            cg_maxiter_local = cg_maxiter

        du, info = spla.cg(K_linop, r_rhs, tol=cg_tol, maxiter=cg_maxiter_local, M=M_linop)
        if info != 0:
            # CG didn't converge fully; still proceed with solution found or fallback to dense solve
            # fallback: form dense K if n small
            if n <= 2000:
                # assemble dense K (careful with memory)
                if A_sparse:
                    A_dense = A.toarray()
                else:
                    A_dense = A
                D = w  # 1D
                K_dense = H + A_dense.T.dot(A_dense * D[:, None])
                du = linalg.solve(K_dense, r_rhs)
            else:
                # try using sparse direct (if available):
                try:
                    # build sparse K explicitly (memory heavy)
                    if A_sparse:
                        # form A.T * diag(w) * A
                        D_s = sp.diags(w)
                        K_sp = H + A.T.dot(D_s.dot(A))
                    else:
                        K_sp = sp.csr_matrix(H) + sp.csr_matrix(A.T.dot((A * w[:, None])))
                    # use sparse LU
                    K_sp = K_sp.tocsc()
                    sol = spla.spsolve(K_sp, r_rhs)
                    du = np.asarray(sol).ravel()
                except Exception:
                    # last resort: use zeros
                    du = np.zeros_like(u)

        # dv = (rc + gamma*(A@du)) / d
        Adu = (Xi_mat @ du)
        dv = (rc + gamma * Adu) / d
        dy = -Adu

        # linesearch: avoid recomputing expensive H@up by reusing Hdu and A@du
        Hdu = H @ du
        ATdv = Xi_mat.T @ dv

        alpha = 1.0
        if lsmax > 0:
            keep_going = True
            m0 = 0.5 * linalg.norm(F)
            l = 0
            while keep_going and l < lsmax:
                # update residuals cheaply:
                rs_p = rs - alpha * (Hdu + (Xi_mat.T @ dv))
                # complementarity still needs function evaluation (nonlinear)
                yp = y + alpha * dy
                vp = v + alpha * dv
                rc_p = - ( (yp + vp) - np.sqrt(yp*yp + vp*vp + veps*veps) )
                Fp_norm = 0.5 * linalg.norm(np.concatenate([rs_p, rc_p]))
                if Fp_norm <= (1 - 2*eta*alpha) * m0:
                    keep_going = False
                else:
                    alpha *= beta
                    l += 1

        # apply step
        u += alpha * du
        v += alpha * dv
        y += alpha * dy
        newton_iters += 1

    out['newton_iters'] = newton_iters
    out['res'] = E
    return u.reshape((-1,1)), v.reshape((-1,1)), E, out

################################################################################
################################################################################




# evaluate the FB function
def fb(u,v,veps):
    
    phi = (u+v) - np.sqrt(u**2 + v**2 + veps**2)
    
    return phi

# derivative of the smoothed fb function
# taken w.r.t its first argument
# the Jacobian is diagonal so its returned as a vector
def dfb(u,v,veps):
    q = np.max(u.shape)
    gamma = np.zeros([q,1])
    mu = np.zeros([q,1])
    r = (u**2 + v**2 + veps**2)**(0.5)
    idx = r > 100* 2.2204e-16
    
    # above the tolerance compute the derivative
    gamma[idx] = (np.ones(u.shape) - u/r).reshape(-1,)
    mu[idx] = (np.ones(u.shape) - v/r).reshape(-1,)
   
    # below the division tolerance pull from the generalized Jacobian
    gamma[~idx] = 1 - 1/np.sqrt(2)
    mu[~idx] = 1- 1/np.sqrt(2)
    
    return gamma, mu


#%%
def fbrs(H,f,Xi_mat,xi_vec,u0,v0):

    # sizes
    n = np.shape(f)[0]
    q = xi_vec.size
    
	# default parameters
    tol = 1e-8 # absolute error tolerance
    rtol = 1e-10 #  relative error tolerance
    eta = 1e-10 # linesearch sufficient decrease parameter
    beta = 0.7 # backtracking parameter
    delta = np.sqrt(2.2204e-16) # regularization strength
    veps =  tol/(2*np.sqrt(q)) * np.ones(v0.shape)  # smoothing strength
    lsmax = 10; # max number of linesearch iterations
    max_newton_iters = 1000 # max number of newton iterations
    use_nr = False # use the natural residual as a stopping criteria

    
    # initialize
    u = u0.copy()
    y = xi_vec - Xi_mat @ u.copy()
    v = v0.copy()
    
    E = 1e10
    E0 = E
    newton_iters = 0
    out = {}
    out['eflag'] = -1
    
    
    d = np.zeros([q,1])
    L = np.zeros(H.shape)
    E = 1e10
    j = 0
    
    # out['rs'] = []
    # out['rc'] = []
    # out['E'] = []
    # out['du'] = []
    # out['dv'] = []

    for j in range(max_newton_iters):
        #print(j)

       # update reg strength
        delta = min(delta,E)

        
        # compute fb derivatives
        [gamma,mu] = dfb(y,v,veps)
        d = mu + delta

            
        # eval rhs
        rs = -(H @ u + f + Xi_mat.T @ v) # stationarity
        rc = -fb(y,v,veps); # complimentarity
        
        # out['rs'].append(rs)
        # out['rc'].append(rc)
                
        # convergence check
        F = np.concatenate([rs , rc ], axis=0)
        
        if use_nr:
            E = 1 #linalg.norm(np.concatenate([rs, min(y,v)]))
#TODO: Need to check this!            
        else:
            E = linalg.norm(F);
		
        if j == 0:
            E0 = E
        
        # out['E'].append(E)
        
        if E <= tol + E0*rtol:
            out['eflag'] = 0
			
            break        
        
        B = np.zeros([q,n])
        
        
        for i in range(q):
            
            B[i,:] = 1/d[i] * gamma[i] * (Xi_mat[i,:])     
       
        
        # K = H + A'*diag(c./d)*A;
        K = H + Xi_mat.T @ B
        r = rs - Xi_mat.T @ (rc/d)
                    
        # newton step, need to do LU decomposition instead of cholesky as H not symmetric
        P, L, U = linalg.lu(K)
        du = linalg.solve(L,P.T @ r)
        du = linalg.solve(U, du)
        dv = (rc + gamma*(Xi_mat @ du))/d
        dy = -Xi_mat @ du
        
        # out['du'].append(du)
        # out['dv'].append(dv)
        
        # L = linalg.cholesky(K,lower = True)
        # du = linalg.solve(L,r)
        # du = linalg.solve(L.T, du)
        # dv = (rc + gamma*(Xi_mat @ du))/d
        # dy = -Xi_mat @ du
        
		# linesearch
        alpha = 1;
        if lsmax > 0 :
            keep_going = True
            m0 = 1/2* linalg.norm(F)
            l = 0
            
            while keep_going and l < lsmax:
                up = u + alpha*du
                vp = v + alpha*dv
                yp = y + alpha*dy
        
                rs = -(H @ up+ f + Xi_mat.T @ vp)
                rc = -fb(yp,vp,veps)
                Fp = np.concatenate([rs,rc], axis=0)
        
                malpha = 1/2*linalg.norm(Fp)
                
                if malpha <= (1-2*eta*alpha)*m0:
                    keep_going = False
                else:
                    alpha = beta*alpha
                    l = l + 1
       
        u += alpha*du
        v += alpha*dv
        y += alpha*dy
        newton_iters = newton_iters + 1
        
    out['newton_iters'] = newton_iters
    out['res'] = E
    out['solver'] = 'fbrs'
    
    return u,v,E,out           

