from abc import ABC, abstractmethod
import numpy as np

class OptimizationController(ABC):
    """
    Abstract object representing an optimization based controller.
    """
    
    def __init__(self):
        pass
    
    def step(self, x_0: np.ndarray):
        # Solve optimization problem
        U_opt, U_dual = self._solve_problem(x_0)
            
        # Compute the predicted future states
        x_opt = self._apply_commands(x_0, U_opt)
        
        # Reorganize the control vector
        U_opt = self._solution_to_command(U_opt)
        
        # Return optimal control and predicted future states
        return U_opt, x_opt
    
    @abstractmethod
    def _solve_problem(self, x_0: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def _apply_commands(self, x_0: np.ndarray, U_opt: np.ndarray):
        pass
    
    @abstractmethod
    def _solution_to_command(self, U_opt: np.ndarray):
        pass