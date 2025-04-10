from abc import ABC, abstractmethod
from typing import List
import numpy as np

class RHG(ABC):
    """Abstract object of a Receding Horizon Game optimization problem."""
    
    def __init__(self, K: int, nx: int, nu: int, Nv: int, nu_v: List[int],):
        # Time Horizon
        self.K = K
        # Variable dimensions
        self.nx = nx
        self.nu = nu
        # Number of agent
        self.Nv = Nv
        # Control dimension for each agent
        self.nu_v = nu_v
    
    @abstractmethod
    def step(self, x_0: np.ndarray):
        pass
    
    @abstractmethod
    def _compute_pseudo_gradient(sel, x_0: np.ndarray):
        pass