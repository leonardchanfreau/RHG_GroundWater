import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import time
import os
import shutil

from .System import System
from .Model import CoarseModel
from .Controllers import RHG


class RecedingHorizon:
    
    def __init__(self, system: System, model: CoarseModel, controller: RHG,
                 param_file: str):
        self.system = system
        self.model = model
        self.controller = controller
        # Printing data
        self.h_wells = np.empty((5,0)) # TODO: Soft code dim
        self.q_rates = np.empty((5,0))
        self.profits = np.empty((3,0))
        # Store data
        self.result_folder = "run_data/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.result_file = self.result_folder + "result_data.npz"
        self.param_file = param_file
        
    def run(self, time_period: int, display_results: bool=False) -> np.ndarray:
        for t in trange(time_period, desc="Simulation progress"):
            self._step(record_results=display_results)
        if display_results:
            self._display_results()
            self._store_data()
        return self.system.get_state()
    
    def _step(self, record_results: bool=False):
        heads = self.system.get_state()
        low_dim_state = self.model.translate_state(heads[0,:,:])
        inputs = self.controller.step(low_dim_state)
        pumping_rates = self.model.translate_input(inputs[[0], :].T)
        if record_results:
            self._record_data(pumping_rates)
        self.system.apply_command(pumping_rates)
        
    def _record_data(self, q_rates: np.ndarray):
        self.h_wells = np.hstack(
            (self.h_wells, self.system.get_heads_at_wells()[:, np.newaxis]))
        self.q_rates = np.hstack((self.q_rates, q_rates))
        self.profits = np.hstack((self.profits, - self.system.compute_cost(q_rates)))
    
    def _display_results(self):
        well_coord = self.system.get_well_position()
        # Display heads evolution
        fig_h = plt.figure(figsize=(25,5))
        for ii in range(5):
            ax = fig_h.add_subplot(1, 5, ii+1)
            ttl = f"Head at cell ({well_coord[ii, 0]},{well_coord[ii, 1]},{well_coord[ii, 2]})"
            ax.set_title(ttl)
            ax.set_xlabel("time")
            ax.set_ylabel("head")
            ax.plot(self.h_wells[ii, :], "b-")
        # Display command evolution
        fig_q = plt.figure(figsize=(25,5))
        for ii in range(5):
            ax = fig_q.add_subplot(1, 5, ii+1)
            ttl = f"Command for well ({well_coord[ii, 0]},{well_coord[ii, 1]},{well_coord[ii, 2]})"
            ax.set_title(ttl)
            ax.set_xlabel("time")
            ax.set_ylabel("pumping rate")
            ax.plot(self.q_rates[ii, :], "r-")
            ax.set_ylim([-1050, 50])
        # Display profit evolution
        fig_p = plt.figure(figsize=(15,5))
        for ii in range(3):
            ax = fig_p.add_subplot(1, 3, ii+1)
            ttl = f"Marginal profit for player {ii+1}"
            ax.set_title(ttl)
            ax.set_xlabel("time")
            ax.set_ylabel("reward")
            ax.plot(self.profits[ii, :], "g-")
        # Show figures
        plt.show()
        
    def _store_data(self):
        _, param_name = os.path.split(self.param_file)
        shutil.copyfile(self.param_file, self.result_folder+param_name)
        np.savez(self.result_file, h_wells=self.h_wells, q_rates=self.q_rates, profit=self.profits)