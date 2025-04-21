import numpy as np

from .System import System
from .Model import CoarseModel
from .Controllers import RHG


class RecedingHorizon:
    
    def __init__(self, system: System, model: CoarseModel, controller: RHG):
        self.system = system
        self.model = model
        self.controller = controller
    
    def run(self, time_period: int) -> np.ndarray:
        for t in range(time_period):
            self._step()
        return self.system.get_state()
    
    def _step(self):
        heads = self.system.get_state()
        low_dim_state = self.model.translate_state(heads[0,:,:])
        inputs = self.controller.step(low_dim_state)
        pumping_rates = self.model.translate_input(inputs[[0], :].T)
        self.system.apply_command(pumping_rates)
        