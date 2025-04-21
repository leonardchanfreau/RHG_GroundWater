from abc import ABC, abstractmethod

import numpy as np

class System(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def apply_command(self, u: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass