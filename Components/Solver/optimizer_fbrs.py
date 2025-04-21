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

import numpy as np
from scipy import linalg


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
        
        # convergence check
        F = np.concatenate([rs , rc ], axis=0)
        
        if use_nr:
            E = 1 #linalg.norm(np.concatenate([rs, min(y,v)]))
#TODO: Need to check this!            
        else:
            E = linalg.norm(F);
		
        if j == 0:
            E0 = E
        
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

