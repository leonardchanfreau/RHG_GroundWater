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

# import time

# Vectorized dfb: input u, v, veps are 1-d arrays of length q
def dfb_vec(u, v, veps):
    # u, v, veps are 1D arrays
    r = np.sqrt(u*u + v*v + veps*veps)
    # mask where r is large enough to divide safely
    tol = 100 * np.finfo(float).eps
    mask = (r > tol).toarray()
    # above the tolerance compute the derivative
    gamma = sp.csr_array(u.shape)
    mu = sp.csr_array(u.shape)
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
    veps = (tol/(2*np.sqrt(q))) * sp.csr_array(np.ones(v.shape))
    E = 1e10
    E0 = E
    newton_iters = 0
    out = {'eflag': -1, 'solver': 'fbrs'}

    # main loop
    for j in range(max_newton_iters):
        delta = min(delta, E)

        # compute dfb (vectorized)
        # start = time.time()
        gamma, mu = dfb_vec(y, v, veps)   # 1D arrays length q
        d = mu + delta*sp.csr_array(np.ones(mu.shape))  # 1D
        # stop = time.time()
        # out["compute dfb"] = (j*out.get("compute dfb", 0) + stop - start)/(j+1)

        # stationarity and complementarity 
        # start = time.time()
        rs = -( (H @ u) + f + (Xi_mat.T @ v) )
        rc = - fb(y,v,veps)
        # stop = time.time()
        # out["eval rhs"] = (j*out.get("eval rhs", 0) + stop - start)/(j+1)

        # combined residual
        # start = time.time()
        F = sp.vstack([rs, rc])
        normF = sp.linalg.norm(F)
        if use_nr:
            E = 1.0
        else:
            E = normF
        if j == 0:
            E0 = E
        # convergence check
        if E <= tol + E0*rtol:
            out['eflag'] = 0
            break
        # stop = time.time()
        # out["cv check"] = (j*out.get("cv check", 0) + stop - start)/(j+1)

        # K = H + A'*diag(c./d)*A;
        # start = time.time()
        K = H + Xi_mat.T @ ((gamma / d) * Xi_mat)
        r = rs - Xi_mat.T @ sp.csr_array(rc/d)
        # stop = time.time()
        # out["K"] = (j*out.get("K", 0) + stop - start)/(j+1)

        # newton step, need to do LU decomposition instead of cholesky as H not symmetric
        # start = time.time()
        splu = sp.linalg.splu(K.tocsc())
        du = sp.csr_array(splu.solve(r.toarray()))
        
        Adu = (Xi_mat @ du)
        dv = sp.csr_array((rc + gamma * Adu) / d)
        dy = -Adu
        # stop = time.time()
        # out["LU"] = (j*out.get("LU", 0) + stop - start)/(j+1)

        # linesearch: avoid recomputing expensive H@up by reusing Hdu and A@du
        # start = time.time()
        Hdu = H @ du
        ATdv = Xi_mat.T @ dv

        alpha = 1.0
        if lsmax > 0:
            keep_going = True
            m0 = 0.5 * normF
            l = 0
            while keep_going and l < lsmax:
                # update residuals cheaply:
                rs_p = rs - alpha * (Hdu + ATdv)
                # complementarity still needs function evaluation (nonlinear)
                yp = y + alpha * dy
                vp = v + alpha * dv
                rc_p = - fb(yp,vp,veps)
                malpha = 0.5 * sp.linalg.norm(sp.vstack([rs_p, rc_p]))
                if malpha <= (1 - 2*eta*alpha) * m0:
                    keep_going = False
                else:
                    alpha *= beta
                    l += 1
        # stop = time.time()
        # out["line search"] = (j*out.get("line search", 0) + stop - start)/(j+1)

        # apply step
        # start = time.time()
        u += alpha * du
        v += alpha * dv
        y += alpha * dy
        newton_iters += 1
        # stop = time.time()
        # out["updt"] = (j*out.get("updt", 0) + stop - start)/(j+1)
        # out["l"] = (j*out.get("l", 0) + l)/(j+1)

    out['newton_iters'] = newton_iters
    out['res'] = E
    return u.reshape((-1,1)), v.reshape((-1,1)), E, out

################################################################################
################################################################################




# evaluate the FB function
def fb(u,v,veps):
    
    phi = (u+v) - np.sqrt(u*u + v*v + veps*veps)
    
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
        # start = time.time()
        [gamma,mu] = dfb(y,v,veps)
        d = mu + delta
        # stop = time.time()
        # out["compute dfb"] = (j*out.get("compute dfb", 0) + stop - start)/(j+1)

            
        # eval rhs
        # start = time.time()
        rs = -(H @ u + f + Xi_mat.T @ v) # stationarity
        rc = -fb(y,v,veps); # complimentarity
        # stop = time.time()
        # out["eval rhs"] = (j*out.get("eval rhs", 0) + stop - start)/(j+1)
        
        # out['rs'].append(rs)
        # out['rc'].append(rc)
                
        # convergence check
        # start = time.time()
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
        # stop = time.time()
        # out["cv check"] = (j*out.get("cv check", 0) + stop - start    )/(j+1)
        
        # start = time.time()
        B = np.zeros([q,n])
        
        
        for i in range(q):
            
            B[i,:] = 1/d[i] * gamma[i] * (Xi_mat[i,:])     
       
        
        # K = H + A'*diag(c./d)*A;
        K = H + Xi_mat.T @ B
        r = rs - Xi_mat.T @ (rc/d)
        # stop = time.time()
        # out["K"] = (j*out.get("K", 0) + stop - start)/(j+1)
                    
        # newton step, need to do LU decomposition instead of cholesky as H not symmetric
        # start = time.time()
        P, L, U = linalg.lu(K)
        du = linalg.solve(L,P.T @ r)
        du = linalg.solve(U, du)
        dv = (rc + gamma*(Xi_mat @ du))/d
        dy = -Xi_mat @ du
        # stop = time.time()
        # out["LU"] = (j*out.get("LU", 0) + stop - start)/(j+1)
        
        # out['du'].append(du)
        # out['dv'].append(dv)
        
        # L = linalg.cholesky(K,lower = True)
        # du = linalg.solve(L,r)
        # du = linalg.solve(L.T, du)
        # dv = (rc + gamma*(Xi_mat @ du))/d
        # dy = -Xi_mat @ du
        
		# linesearch
        # start = time.time()
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
        # stop = time.time()
        # out["line search"] = (j*out.get("line search", 0) + stop - start)/(j+1)
       
        # start = time.time()
        u += alpha*du
        v += alpha*dv
        y += alpha*dy
        newton_iters = newton_iters + 1
        # stop = time.time()
        # out["updt"] = (j*out.get("updt", 0) + stop - start)/(j+1)
        
    out['newton_iters'] = newton_iters
    out['res'] = E
    out['solver'] = 'fbrs'
    
    return u,v,E,out           

