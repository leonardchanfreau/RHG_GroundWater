import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
import copy

def correct_averaging(system, model):
    # Use random inputs as training data
    n_sample = 100
    T = 100
    n_well = model.well_to_agent.shape[0]
    train_command = [(np.random.randint(-3000,0, (1,n_well))*np.ones((T, n_well))).T for _ in range(n_sample)]
    # Compute heads and costs for different commands at different wells
    system = copy.deepcopy(system)
    head_rd = np.array([np.hstack((system.get_heads_at_wells()[:,np.newaxis], np.zeros((n_well, T))))] * n_sample)
    cost_rd = np.array([np.zeros((model.n_agent, T))] * n_sample)
    for step in range(n_sample):
        system.state = system.initial_state.copy()
        for k in range(T):
            system.apply_command(train_command[step][:,[k]])
            head_rd[step, :, k+1] = system.get_heads_at_wells()
            cost_rd[step, :, k] = system.compute_cost(train_command[step][:,[k]]).flatten()
    # Compute expected heads for those commands
    model = copy.deepcopy(model)
    B = sp.hstack(model.B_v)
    system.state = system.initial_state.copy()
    model_head_rd = np.array([np.hstack((model.translate_state(system.get_state()[0,:,:]), np.zeros((model.nx, T))))] * n_sample)
    for step in range(n_sample):
        for k in range(T):
            input = np.kron(sp.vstack(model.well_to_command_v) @ train_command[step][:,[k]], np.array([[1] + [0]*(model.n_crops-1)]).T)
            model.state = model_head_rd[step, :, [k]].T
            model_head_rd[step, :, k+1] = (model.A @ model_head_rd[step, :, [k]].T + B @ input + model.C).flatten()
    # Fit linear function between pumping at well and error term in the same well.
    well_pos = np.nonzero(model.well_to_cell.T)[1]
    pumping_rates = np.array([[train_command[_][ii,0] for _ in range(n_sample)] for ii in range(n_well)])
    error = np.array([model_head_rd[:, well_pos[ii], -1] - head_rd[:, ii, -1] for ii in range(n_well)])
    res = least_squares(lambda x : (x*pumping_rates.flatten() - error.flatten()), x0 = 0)
    cx_u = res.x
    return cx_u