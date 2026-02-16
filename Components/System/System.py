from abc import ABC, abstractmethod

import numpy as np

from ..ParameterInterface import ParameterInterface
class System(ABC):
    
    def __init__(self, param_handler: ParameterInterface, boundary_condition: str="Free-flow"):
        pass
    
    @abstractmethod
    def apply_command(self, u: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_heads_at_wells(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_well_position(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def compute_cost(self, u: np.ndarray) -> np.ndarray:
        pass