from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import json

class RHG(ABC):
    """Abstract object of a Receding Horizon Game optimization problem."""
    
    def __init__(self, param_file: str):
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
        # Time Horizon
        self.K = params['controller']['time_horizon']
        # Number of agent
        self.Nv = params['wells']['n_player']
    
    @abstractmethod
    def step(self, x_0: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def _compute_pseudo_gradient(sel, x_0: np.ndarray):
        pass