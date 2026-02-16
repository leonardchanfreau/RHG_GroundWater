import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import trange
import time
import os
import shutil

from .ParameterInterface import ParameterInterface
from .System import System
from .Model import CoarseModel, CostModel
from .Controllers import RHG


class RecedingHorizon:
    
    def __init__(self, system: System, model: CoarseModel, controller: RHG,
                 cost_model: CostModel, param_handler: ParameterInterface,
                 K_skip : int = 1):
        self.system = system
        self.model = model
        self.controller = controller
        self.cost_model = cost_model
        self.time = 0
        self.K_skip = min(K_skip, controller.K)
        # Printing data
        n_well = model.well_to_agent.shape[0]
        self.h_wells = np.empty((n_well,0))
        self.avg_h_wells = np.empty((n_well,0))
        self.corrected_h_wells = np.empty((n_well,0))
        self.future_h_wells = np.empty((n_well, controller.K+1, 0))
        self.inputs = np.empty((controller.nu,0))
        self.states = np.empty((model.nx,0))
        self.q_rates = np.empty((n_well,0))
        self.real_profits = np.empty((model.n_agent,0))
        self.model_profits = np.empty((model.n_agent,0))
        self.last_pumping_rates = np.zeros((n_well,1))
        # Store data
        self.result_folder = "run_data/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.result_file = self.result_folder + "result_data.npz"
        self.param_file = param_handler.parameter_file
        
    def run(self, time_period: int, display_results: bool=False) -> np.ndarray:
        for t in trange(time_period, desc="Simulation progress", position=1, leave=False):
            self._step(record_results=display_results)
        if display_results:
            self._display_results()
            self._store_data()
        return self.system.get_state()
    
    def _step(self, record_results: bool=False):
        heads = self.system.get_state()
        low_dim_state = self.model.translate_state(heads[0,:,:])  # TODO: optimize
        Q, R, R_minus, M, L = self.cost_model.get_cost_matrices(self.time, self.controller.K)
        self.controller.update_cost_matrices(Q, R, R_minus, M, L)
        if self.time % self.K_skip == 0:
            self.input, self.x_pred = self.controller.step(
                sp.csr_array(self.model.get_initial_state()))
        pumping_rates = self.model.translate_input(self.input[self.time%self.K_skip:,:].T) # TODO: optimize
        if record_results:
            self._record_data(low_dim_state, self.input[self.time%self.K_skip:,:], pumping_rates, self.x_pred[:, self.time%self.K_skip:])
        self.system.apply_command(pumping_rates[:, [0]].round())
        self.time += 1
    
    
    def _record_data(self, state: np.ndarray, inputs: np.ndarray, q_rates: np.ndarray,
                     x_pred: np.ndarray):
        self.states = np.hstack((self.states, state))
        # Apply heads correction
        x_pred = self.model.select_wells(x_pred)
        x_pred_corrected = self.model.apply_head_correction(
            x_pred, np.hstack((self.last_pumping_rates, q_rates)))
        current_avg_heads = self.model.get_heads_at_wells()
        current_corrected_heads = self.model.apply_head_correction(
            current_avg_heads, self.last_pumping_rates)
        self.last_pumping_rates = q_rates[:, [0]]
        # Store data in vectors
        self.h_wells = np.hstack(
            (self.h_wells, self.system.get_heads_at_wells()[:, np.newaxis]))
        self.avg_h_wells = np.hstack(
            (self.avg_h_wells, current_avg_heads))
        self.corrected_h_wells = np.hstack(
            (self.corrected_h_wells, current_corrected_heads))
        self.future_h_wells = np.concatenate(
            (self.future_h_wells,
             np.pad(x_pred_corrected, ((0,0),(0,self.future_h_wells.shape[1]-x_pred_corrected.shape[1])), 'constant', constant_values=(None))[:,:,np.newaxis]), axis=2)
        self.inputs = np.hstack((self.inputs, inputs[[0], :].T))
        self.q_rates = np.hstack((self.q_rates, q_rates[:, [0]]))
        input_per_well = self.model.reorganize_input_per_well(inputs[[0],:].T)
        self.real_profits = np.hstack((self.real_profits,
                                       - self.system.compute_cost(q_rates[:, [0]]))) # TODO: adapt this cost (should only compute pumping cost)
        self.model_profits = np.hstack((self.model_profits,
                                        - self.cost_model.compute_cost(
                                            current_corrected_heads, input_per_well,
                                            self.time)))
    
    def _display_results(self):
        well_coord = self.system.get_well_position()
        n_well = int(sum(np.hstack(self.model.n_well_per_command)))
        # Display heads evolution
        fig_h = plt.figure(figsize=(n_well*5+3,5))
        ax_h = None
        for ii in range(n_well):
            ax_h = fig_h.add_subplot(1, n_well, ii+1, sharey=ax_h)
            ttl = f"Head at cell ({well_coord[ii, 0]},{well_coord[ii, 1]},{well_coord[ii, 2]})"
            ax_h.set_title(ttl)
            ax_h.set_xlabel("time")
            if ii == 0:
                ax_h.set_ylabel("head")
            ax_h.plot(self.h_wells[ii, :], "b-", label="Real head")
            ax_h.plot(self.corrected_h_wells[ii, :], "b--", label="Corrected\nmodel head")
            ax_h.plot(self.avg_h_wells[ii, :], "b-.", alpha=0.2, label="Coarse grid\nmodel head")
            # Display predicted evolution
            dt = np.linspace(0,self.controller.K, self.controller.K+1)
            for t0, h_pred in enumerate(np.rollaxis(self.future_h_wells, 2)):
               t = t0 + dt
               ax_h.plot(t, h_pred[ii, :], "b-", alpha=0.3,
                         label="Predicted\nheads" if t0==0 else None)
        ax_h.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
        # Display command evolution
        fig_q = plt.figure(figsize=(n_well*5,5))
        ax_q = None
        for ii in range(n_well):
            ax_q = fig_q.add_subplot(1, n_well, ii+1, sharey=ax_q)
            ttl = f"Command for well ({well_coord[ii, 0]},{well_coord[ii, 1]},{well_coord[ii, 2]})"
            ax_q.set_title(ttl)
            ax_q.set_xlabel("time")
            if ii == 0:
                ax_q.set_ylabel("pumping rate")
            ax_q.plot(self.q_rates[ii, :], "r-")
            # ax_q.set_ylim([-100050, 50])
        # Display profit evolution
        fig_p = plt.figure(figsize=(self.model.n_agent*5+3,5))
        ax_p = None
        for ii in range(self.model.n_agent):
            ax_p = fig_p.add_subplot(1, self.model.n_agent, ii+1, sharey=ax_p)
            ttl = f"Marginal profit for player {ii+1}"
            ax_p.set_title(ttl)
            ax_p.set_xlabel("time")
            if ii == 0:
                ax_p.set_ylabel("reward")
            ax_p.plot(self.real_profits[ii, :], "g-", label="Real profit")
            ax_p.plot(self.model_profits[ii, :], "g--", label="Modeled\nprofit")
        ax_p.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
        # Show figures
        fig_h.show()
        fig_q.show()
        fig_p.show()
        # Save figures
        fig_h.savefig(self.result_folder+"heads.png")
        fig_q.savefig(self.result_folder+"q_rates.png")
        fig_p.savefig(self.result_folder+"profit.png")
        plt.close(fig_h)
        plt.close(fig_q)
        plt.close(fig_p)
        
    def _store_data(self):
        _, param_name = os.path.split(self.param_file)
        shutil.copyfile(self.param_file, self.result_folder+param_name)
        np.savez(self.result_file,
                 h_wells=self.h_wells,
                 avg_h_wells=self.avg_h_wells,
                 corrected_h_wells=self.corrected_h_wells,
                 pred_avg_h_wells=self.future_h_wells,
                 states=self.states,
                 inputs=self.inputs,
                 q_rates=self.q_rates,
                 real_profit=self.real_profits,
                 model_profit=self.model_profits)