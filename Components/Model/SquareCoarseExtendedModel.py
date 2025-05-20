from typing import Tuple, Dict, List

import math
import numpy as np
import scipy.sparse as sp

from scipy.signal import convolve2d

from .SquareCoarseModel import SquareCoarseModel

class SquareCoarseExtendedModel(SquareCoarseModel):
    """
    Model used to express interactions of present input and future state in the
    cost. To be precise, the real cost of pumping is approximated by an
    upper-bound which is the product of the pumping rate with the heads at the
    end of the time-step.
    
    The state is composed of a part x_c common to all players, and a specific
    part x_s different for each player: x = [x_c, x_s].
    Here x_s is simply the inputs from the previous time-step.
    """
    
    def __init__(self, param_file: str, boundary_condition: str="Free-flow"):
        super().__init__(param_file, boundary_condition)
        # Dimension of the specific state
        self.nx_s_v = self.nu_v.copy()
        # Dynamic matrices of the specific state
        A_sc_v = [sp.csr_array((nx_s_v,self.nx)) for nx_s_v in self.nx_s_v]
        A_ss_v = [sp.csr_array((nx_s_v,nx_s_v)) for nx_s_v in self.nx_s_v]
        self.A_s_v = [sp.hstack([A_sc_v[v], A_ss_v[v]]) for v in range(self.n_agent)]
        self.B_s_v = [sp.eye(nx_s_v) for nx_s_v in self.nx_s_v]
        self.C_s_v = [sp.csr_array((nx_s_v,1)) for nx_s_v in self.nx_s_v]
        # Adapt cost matrices to this extended state to compute upper-bound estimation
        self._extend_cost_matrices()
    
    def _extend_cost_matrices(self):
        # Input-state cross term becomes input-input
        self.Gamma_v_tilde = [
            sp.block_array(
                [[sp.csr_array((self.nx, self.nx)), self.Gamma_v[v]],
                [sp.csr_array((self.nu_v[v], self.nx)), sp.csr_array((self.nu_v[v], self.nu_v[v]))]])
            for v in range(self.n_agent)]
    
    def get_initial_state(self) -> np.ndarray:
        # Append zero command (wont play any role in the optimization pb)
        return np.vstack((self.state, np.zeros((sum(self.nx_s_v), 1))))