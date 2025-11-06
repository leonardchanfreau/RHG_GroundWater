import numpy as np
import matplotlib.pyplot as plt
from Components.Model import IrrigationSchedulingModel, IrrigationSchedulingCost

def compare_welfare_to_game(common_folder: str, parameter_filename: str,
                            crops:np.ndarray,
                            t_start:int = 0, period:int =365):
    # Get data
    data_si = np.load(common_folder+"game/result_data.npz")
    data_wf = np.load(common_folder+"welfare/result_data.npz")
    # Construct utils objects
    model = IrrigationSchedulingModel(common_folder+"game/"+parameter_filename,
                                      boundary_condition="Constant Head")
    cost_model = IrrigationSchedulingCost(common_folder+"game/"+parameter_filename,
                                          model, crops)
    # Construct utils variables
    water_req = cost_model._get_water_requirement(t_start, period)
    under_irrigation_weight = cost_model._get_under_irrigation_weight(t_start, period)
    n_well = int(sum(np.hstack(model.n_well_per_command)))
    well_to_agent = np.zeros((model.n_agent, n_well))
    indices = np.array(list(zip(model.well_to_agent.tolist(), range(n_well)))).T
    well_to_agent[indices[0, :], indices[1, :]] = 1
    
    # Compute pumping rates
    player_pumping_si = well_to_agent @ data_si['q_rates']
    player_pumping_wf = well_to_agent @ data_wf['q_rates']
    rain = cost_model._get_available_water(t_start, period)
    # Compare pumping rates
    fig = plt.figure(figsize=(15+5,5))
    ax = None
    for ii in range(3):
        ax = fig.add_subplot(2, 3, ii+1, sharey=ax)
        ttl = f"Pumping volume for player {ii+1}"
        ax.set_title(ttl)
        if ii == 0:
            ax.set_ylabel("volume")
        ax.plot((water_req+rain)[ii, :, :].sum(axis=0))
        ax.fill((water_req+rain)[ii, :, :].sum(axis=0), label='requirement')
        ax.fill(-player_pumping_si[ii, :] + rain[ii,:,:].sum(axis=0), "r-", label='pumping\n+ rain')
        ax.fill(rain[ii,:,:].sum(axis=0), "orange", label='rain')
        if ii == 2:
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
        # Welfare
        ax = fig.add_subplot(2, 3, 3+ii+1, sharey=ax)
        ax.set_xlabel("time")
        if ii == 0:
            ax.set_ylabel("volume")
        ax.plot((water_req+rain)[ii, :, :].sum(axis=0))
        ax.fill((water_req+rain)[ii, :, :].sum(axis=0))
        ax.fill(-player_pumping_wf[ii, :] + rain[ii,:,:].sum(axis=0), "r-")
        ax.fill(rain[ii,:,:].sum(axis=0), "orange")
    fig.savefig(common_folder+"pumping.png")
    plt.close(fig)
    
    # Plot cost per player
    fig = plt.figure(figsize=(15+5,5))
    ax = None
    for ii in range(3):
        ax = fig.add_subplot(1, 3, ii+1, sharey=ax)
        ttl = f"Cost for player {ii+1}"
        ax.set_title(ttl)
        ax.set_xlabel("time")
        if ii == 0:
            ax.set_ylabel("cost")
        ax.plot(-data_si['model_profit'][ii, :], 'g', label='game')
        ax.plot(-data_wf['model_profit'][ii, :], color="orange", label='welfare')
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    fig.savefig(common_folder+"costs.png")
    plt.close(fig)
    
    # Plot total cost
    fig = plt.figure(figsize=(10+5,5))
    ax = fig.add_subplot(1, 2, 1)
    ttl = f"Sum of costs for all players"
    ax.set_title(ttl)
    ax.set_xlabel("time")
    if ii == 0:
        ax.set_ylabel("cost")
    ax.plot(-data_si['model_profit'].sum(axis=0), 'g')
    ax.plot(-data_wf['model_profit'].sum(axis=0), color="orange")
    ax = fig.add_subplot(1, 2, 2)
    ttl = f"Sum of accumulated costs for all players"
    ax.set_title(ttl)
    ax.set_xlabel("time")
    if ii == 0:
        ax.set_ylabel("cost")
    ax.plot(-np.cumsum(data_si['model_profit'].sum(axis=0)), 'g', label='game')
    ax.plot(-np.cumsum(data_wf['model_profit'].sum(axis=0)), color="orange", label='welfare')
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    fig.savefig(common_folder+"total_cost.png")
    plt.close(fig)