import shutil
import time
import math
import numpy as np
import json
import itertools
from tqdm import tqdm
import os
import shutil

from Components.RecedingHorizon import RecedingHorizon
from Components.System import SimpleSquareExample
from Components.Model import SquareCoarseModel, LinearProfitCost
from Components.Controllers import LTI_LQRc_RHG
from Components.Model import IrrigationSchedulingModel, IrrigationSchedulingCostLinearYield, IrrigationSchedulingCostQuadraticYield
from Components.Controllers import TimeVaryingCost_LQRc_RHG, LQRc_welfare
from Components.ParameterInterface import PhysicalParameterInterface, DimensionlessParameterInterface
from Components.utils.plot_utils import compare_welfare_to_game
from Components.utils.grid_utils import correct_averaging


def main (parameter_file: str, use_welfare: bool = False, crops: np.ndarray = None):
    # Example crop repartition
    if crops is not None:
        dir, file = os.path.split(parameter_file)
        name, ext = file.split('.')
        new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
        shutil.copyfile(parameter_file, new_parameter_file)
        with open(new_parameter_file, 'r+') as f:
            file_data = json.load(f)
            file_data["crops_repartition"] = crops.tolist()
            f.seek(0)
            json.dump(file_data, f, indent=4)
        parameter_file = new_parameter_file
    
    # Parameter handler:
    param_handler = DimensionlessParameterInterface(parameter_file)
    # param_handler = PhysicalParameterInterface(parameter_file)
    # Initialize System and Model
    system = SimpleSquareExample(param_handler, boundary_condition="Constant Head")
    model = IrrigationSchedulingModel(param_handler, boundary_condition="Constant Head")
    # cost = IrrigationSchedulingCostQuadraticDistance(parameter_file, model, crops)
    cost = IrrigationSchedulingCostLinearYield(param_handler, model)
    # cost = IrrigationSchedulingCostQuadraticYield(param_handler, model)
    # Initialize Controller
    controller = TimeVaryingCost_LQRc_RHG(
        param_handler,
        model.A, model.B_v, model.C,
        model.H_x, model.h_x, model.H_u_v, model.h_u_v,
        model.H_xu_x, model.H_xu_u, model.h_xu,
        solver='fbrs'
    )
    
    if use_welfare:
        # Initialize welfare controller
        welfare_controller = LQRc_welfare(controller)
        # Initialize receding horizon with welfare
        receding_horizon = RecedingHorizon(system, model, welfare_controller, cost, param_handler)
    else:
        # Initialize receding horizon
        receding_horizon = RecedingHorizon(system, model, controller, cost, param_handler)#, K_skip=10)
    # Run loop
    receding_horizon.run(time_period=150, display_results=True)
    # Store type of control used
    _, param_name = os.path.split(parameter_file)
    with open(receding_horizon.result_folder + param_name, 'r+') as f:
        file_data = json.load(f)
        file_data["use_welfare"] = int(use_welfare)
        file_data["crops_repartition"] = crops.tolist()
        f.seek(0)
        json.dump(file_data, f, indent=4)
    return receding_horizon.result_folder


def test_parameters(parameter_file: str):
    wells_position = [[[0, 490, 490], [0, 260, 550], [0, 820, 210]],
                      [[0, 450, 625], [0, 250, 700], [0, 270, 430]],
                      [[0, 800, 700], [0, 550, 580], [0, 380, 810]]]
    n_player = len(wells_position)
    n_crops = 4
    n_crops_combination = 8
    n_yield_coef_variation = 10
    scaling = [1]
        
    # Compute all possible well combination
    well_combinations = []
    for wells in wells_position:
        well_combinations_v = [wells]
        for i, well in enumerate(wells):
            well_combinations_v.append([well])
            well_combinations_v.append(wells[:i]+wells[i+1:])
        well_combinations.append(well_combinations_v)
    
    # Compare all possible setups
    pbar = tqdm(
        total=(math.prod([len(comb) for comb in well_combinations])*
               (n_crops_combination+2)*
               (n_yield_coef_variation)*
               len(scaling)),
         position=0, desc="Setting batch", leave=True)
    for comb in itertools.product(*well_combinations):
        wells = [el for el_v in comb for el in el_v]
        to_player = [[i]*len(elt) for i,elt in enumerate(comb)]
        to_player = [el for el_v in to_player for el in el_v]
        for i_crops in range(n_crops_combination+2):
            rd = np.random.rand(n_player*(n_crops+1)).reshape((n_crops+1), n_player)
            crops = (rd/rd.sum(axis=0)).round(1)[:-1,:]
            if i_crops == 0:
                crops = np.zeros_like(crops)
                np.fill_diagonal(crops, 1.)
            elif i_crops == 1:
                crops = (1./n_crops)*np.ones_like(crops)
            for i_yield_coef in range(n_yield_coef_variation):
                yield_coef = np.random.randint(1, 10, (n_crops, n_player))/10
                for scale in scaling:
                    with open(parameter_file, 'r+') as f:
                        data = json.load(f)
                        # modify fields
                        data['wells']['pos'] = wells
                        data['wells']['to_player'] = to_player
                        data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                        data['constraints']['well_depth'] = [-60.0]*len(wells)
                        data['land']['surface'] = [elt*scale for elt in data['land']['surface']]
                        data['costs']['yield_sensitivity'] = yield_coef.tolist()
                        # overwrite file
                        f.seek(0)        # <--- should reset file position to the beginning.
                        json.dump(data, f, indent=4)
                        f.truncate()     # remove remaining part
                    # Welfare
                    welfare_folder = main(parameter_file, True, crops)
                    # Game
                    game_folder = main(parameter_file, False, crops)
                    # Group folders
                    common_folder = "run_data/setting_study/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
                    shutil.move(welfare_folder, common_folder+"welfare/")
                    shutil.move(game_folder, common_folder+"game/")
                    # Add some plots
                    compare_welfare_to_game(common_folder, crops, t_start=85, period=180)
                    
                    pbar.update(1)
    pbar.close()

 
def correct_heads(parameter_file):
    param_handler = DimensionlessParameterInterface(parameter_file)
    # Initialize System and Model
    system = SimpleSquareExample(param_handler, boundary_condition="Constant Head")
    model = IrrigationSchedulingModel(param_handler, boundary_condition="Constant Head")
    # Compute correction
    cx_u = correct_averaging(system, model)
    # Overwrite parameter file
    with open(parameter_file, 'r+') as f:
        data = json.load(f)
        # modify fields
        data['correction']['head'] = cx_u
        # overwrite file
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

def test_well_distance(parameter_file: str):
    well_scenarios = [
        # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 500, 450], [0, 450, 625]],
        #  "to_player": [0, 2, 1, 0, 1]},
        # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 500, 450], [0, 450, 530]],
        #  "to_player": [0, 2, 1, 0, 1]},
        # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 500, 450], [0, 490, 530]],
        #  "to_player": [0, 2, 1, 0, 1]},
        # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 450, 625]],
        #  "to_player": [0, 2, 1, 1]},
            # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 450, 530]],
            #  "to_player": [0, 2, 1, 1]},
        {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 490, 530]],
         "to_player": [0, 2, 1, 1]},
        # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 690, 290], [0, 450, 625]],
        #  "to_player": [0, 2, 1, 0, 1]},
            # {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 690, 290], [0, 450, 530]],
            #  "to_player": [0, 2, 1, 0, 1]},
        {"wells_position": [[0, 490, 490], [0, 800, 700], [0, 250, 700], [0, 690, 290], [0, 490, 530]],
         "to_player": [0, 2, 1, 0, 1]},
    ]
    demand_scenarios = [
        {"land": 1,     "yield_sensitivity": 1},
        # Vary the demand (zero crossing) without varying the under-irrigation sensitivity
        # {"land": 2,     "yield_sensitivity": 4},
        # {"land": 10,    "yield_sensitivity": 100},
        # {"land": 0.5,   "yield_sensitivity": 0.25},
        # {"land": 0.1,   "yield_sensitivity": 0.01},
        # Vary the slope without varying the demand (optimal pumping will vary)
        # {"land": 1,     "yield_sensitivity": 2},
        # {"land": 1,     "yield_sensitivity": 10},
        # {"land": 1,     "yield_sensitivity": 0.5},
        # {"land": 1,     "yield_sensitivity": 0.1},
        # TODO: vary sensibility without varying optimal pumping
        # a1/k -> a2 + a1(k - 1)/k (but a_2 cannot be controlled independently from b_2)
    ]
    
    crops = np.array([[1., 1., 0.3], [0., 0., 0.3], [0., 0., 0.3], [0., 0., 0.4]])
    dir, file = os.path.split(parameter_file)
    name, ext = file.split('.')
    new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
    
    counter = 0
    
    pbar = tqdm(total=(len(well_scenarios)*len(demand_scenarios)),
                position=0, desc="Scenario", leave=True)
    for well_scenario in well_scenarios:
        wells = well_scenario['wells_position']
        for demand_scenario in demand_scenarios:
            if counter < 0:
                counter += 1
                pbar.update(1)
                continue
            shutil.copyfile(parameter_file, new_parameter_file)
            with open(new_parameter_file, 'r+') as f:
                data = json.load(f)
                # modify fields
                data['wells']['pos'] = wells
                data['wells']['to_player'] = well_scenario['to_player']
                data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                data['constraints']['well_depth'] = [-60.0]*len(wells)
                data['land']['surface'][0] *= demand_scenario['land']
                data['costs']['yield_sensitivity'][0][0] *= demand_scenario['yield_sensitivity']
                # overwrite file
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            # Welfare
            welfare_folder = main(new_parameter_file, True, crops)
            # Game
            game_folder = main(new_parameter_file, False, crops)
            # Group folders
            common_folder = "run_data/well_distance_study/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
            shutil.move(welfare_folder, common_folder+"welfare/")
            shutil.move(game_folder, common_folder+"game/")
            # Add some plots
            parameter_filename = name + '_changed' + '.' + ext
            compare_welfare_to_game(common_folder, parameter_filename, crops, t_start=0, period=40)

            os.remove(new_parameter_file)
            pbar.update(1)
    pbar.close()


def test_three_aligned_wells(parameter_file: str):
    well_scenarios = [
        # {"wells_position": [[0, 490, 490], [0, 420, 490], [0, 580, 490]],
        #  "to_player": [0, 1, 2]},
        {"wells_position": [[0, 490, 490], [0, 460, 490], [0, 540, 490]],
         "to_player": [0, 1, 2]},
        # {"wells_position": [[0, 490, 490], [0, 460, 490], [0, 580, 490]],
        #  "to_player": [0, 1, 2]},
        {"wells_position": [[0, 490, 490], [0, 460, 490], [0, 620, 490]],
         "to_player": [0, 1, 2]}
    ]
    demand_scenarios = [
        {"land": [10000, 1000, 1000],     "yield_sensitivity": [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0]]}
    ]
    
    crops = np.array([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    dir, file = os.path.split(parameter_file)
    name, ext = file.split('.')
    new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
    
    counter = 0
    
    pbar = tqdm(total=(len(well_scenarios)*len(demand_scenarios)),
                position=0, desc="Scenario", leave=True)
    for well_scenario in well_scenarios:
        wells = well_scenario['wells_position']
        for demand_scenario in demand_scenarios:
            if counter < 0:
                counter += 1
                pbar.update(1)
                continue
            shutil.copyfile(parameter_file, new_parameter_file)
            with open(new_parameter_file, 'r+') as f:
                data = json.load(f)
                # modify fields
                data['wells']['pos'] = wells
                data['wells']['to_player'] = well_scenario['to_player']
                data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                data['constraints']['well_depth'] = [-60.0]*len(wells)
                data['land']['surface'] = demand_scenario['land']
                data['costs']['yield_sensitivity'] = demand_scenario['yield_sensitivity']
                # overwrite file
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            # Welfare
            welfare_folder = main(new_parameter_file, True, crops)
            # Game
            game_folder = main(new_parameter_file, False, crops)
            # Group folders
            common_folder = "run_data/aligned_wells_study/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
            shutil.move(welfare_folder, common_folder+"welfare/")
            shutil.move(game_folder, common_folder+"game/")
            # Add some plots
            parameter_filename = name + '_changed' + '.' + ext
            compare_welfare_to_game(common_folder, parameter_filename, crops, t_start=0, period=40)

            os.remove(new_parameter_file)
            pbar.update(1)
    pbar.close()
    
def test_boundary_condition_impact(parameter_file: str):
    well_scenarios = [
        {"wells_position": [[0, 20, 490], [0, 60, 490], [0, 680, 100]],
         "to_player": [0, 1, 2]},
        {"wells_position": [[0, 20, 490], [0, 100, 490], [0, 680, 100]],
         "to_player": [0, 1, 2]},
        {"wells_position": [[0, 20, 490], [0, 60, 490], [0, 100, 490]],
         "to_player": [0, 1, 2]},
    ]
    demand_scenarios = [
        {"land": [10000, 1000, 1000],     "yield_sensitivity": [[0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0]]}
    ]
    
    crops = np.array([[1., 1., 0.3], [0., 0., 0.3], [0., 0., 0.3], [0., 0., 0.4]])
    dir, file = os.path.split(parameter_file)
    name, ext = file.split('.')
    new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
    
    counter = 0
    
    pbar = tqdm(total=(len(well_scenarios)*len(demand_scenarios)),
                position=0, desc="Scenario", leave=True)
    for well_scenario in well_scenarios:
        wells = well_scenario['wells_position']
        for demand_scenario in demand_scenarios:
            if counter < 0:
                counter += 1
                pbar.update(1)
                continue
            shutil.copyfile(parameter_file, new_parameter_file)
            with open(new_parameter_file, 'r+') as f:
                data = json.load(f)
                # modify fields
                data['wells']['pos'] = wells
                data['wells']['to_player'] = well_scenario['to_player']
                data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                data['constraints']['well_depth'] = [-60.0]*len(wells)
                data['land']['surface'] = demand_scenario['land']
                data['costs']['yield_sensitivity'] = demand_scenario['yield_sensitivity']
                # overwrite file
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            # Welfare
            welfare_folder = main(new_parameter_file, True, crops)
            # Game
            game_folder = main(new_parameter_file, False, crops)
            # Group folders
            common_folder = "run_data/boundary_condition_study/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
            shutil.move(welfare_folder, common_folder+"welfare/")
            shutil.move(game_folder, common_folder+"game/")
            # Add some plots
            parameter_filename = name + '_changed' + '.' + ext
            compare_welfare_to_game(common_folder, parameter_filename, crops, t_start=0, period=40)

            os.remove(new_parameter_file)
            pbar.update(1)
    pbar.close()
    
    
def infinite_horizon_welfare_analysis(parameter_file: str):
    well_scenarios = [
        {"wells_position": [[0, 490, 490], [0, 530, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 450, 490], [0, 530, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 450, 490], [0, 570, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 410, 490], [0, 570, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 410, 490], [0, 610, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 370, 490], [0, 610, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 370, 490], [0, 650, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 330, 490], [0, 650, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 330, 490], [0, 690, 490]],
         "to_player": [0, 1]},
        {"wells_position": [[0, 290, 490], [0, 690, 490]],
         "to_player": [0, 1]}
    ]
    demand_scenarios = [
        {"land": [1000, 1000],     "yield_sensitivity": [[0.5, 0.5], [0, 0], [0, 0], [0, 0]]}
    ]
    horizon = 30
    
    crops = np.array([[1., 1.], [0., 0.], [0., 0.], [0., 0.]])
    dir, file = os.path.split(parameter_file)
    name, ext = file.split('.')
    new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
    
    correct_heads(new_parameter_file)
    
    counter = 0
    
    pbar = tqdm(total=(len(well_scenarios)*len(demand_scenarios)),
                position=0, desc="Scenario", leave=True)
    for well_scenario in well_scenarios:
        wells = well_scenario['wells_position']
        for demand_scenario in demand_scenarios:
            if counter < 0:
                counter += 1
                pbar.update(1)
                continue
            shutil.copyfile(parameter_file, new_parameter_file)
            with open(new_parameter_file, 'r+') as f:
                data = json.load(f)
                # modify fields
                data['wells']['pos'] = wells
                data['wells']['to_player'] = well_scenario['to_player']
                data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                data['constraints']['well_depth'] = [-60.0]*len(wells)
                data['land']['surface'] = demand_scenario['land']
                data['costs']['yield_sensitivity'] = demand_scenario['yield_sensitivity']
                data['controller']['time_horizon'] = horizon
                # overwrite file
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            # Welfare
            welfare_folder = main(new_parameter_file, True, crops)
            # Group folders
            common_folder = "run_data/two_wells_analysis/infinite_welfare/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
            shutil.move(welfare_folder, common_folder)
            
            os.remove(new_parameter_file)
            pbar.update(1)
    pbar.close()

    
def finite_horizon_welfare_analysis(parameter_file: str):
    well_scenarios = [
        {"wells_position": [[0, 780, 490], [0, 820, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 780, 490], [0, 860, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 740, 490], [0, 860, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 740, 490], [0, 900, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 700, 490], [0, 900, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 700, 490], [0, 940, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 660, 490], [0, 940, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 660, 490], [0, 980, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 620, 490], [0, 980, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 620, 490], [0, 1020, 490]],
         "to_player": [0, 1],
         "R_p": 200},
        
        # {"wells_position": [[0, 500, 490], [0, 580, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 500, 490], [0, 620, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 460, 490], [0, 620, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 4/24)},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 8/24)},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 12/24)},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 16/24)}
    ]
    demand_scenarios = [
        {"land": [1000, 1000]}
    ]
    # horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30]
    # horizons = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    horizons = [90]
    # horizons = [22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]
    
    crops = np.array([[1., 1.], [0., 0.], [0., 0.], [0., 0.]])
    dir, file = os.path.split(parameter_file)
    name, ext = file.split('.')
    new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
    
    # correct_heads(new_parameter_file)
    
    counter = 0
    
    pbar = tqdm(total=(len(well_scenarios)*len(demand_scenarios)*len(horizons)),
                position=0, desc="Welfare scenario", leave=True)
    for well_scenario in well_scenarios:
        wells = well_scenario['wells_position']
        for demand_scenario in demand_scenarios:
            for horizon in horizons:
                # if horizon != 30:
                #     counter += 1
                #     pbar.update(1)
                #     continue
                shutil.copyfile(parameter_file, new_parameter_file)
                with open(new_parameter_file, 'r+') as f:
                    data = json.load(f)
                    # modify fields
                    data['wells']['pos'] = wells
                    data['wells']['to_player'] = well_scenario['to_player']
                    data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                    data['constraints']['well_depth'] = [-60.0]*len(wells)
                    data['land']['surface'] = demand_scenario['land']
                    data['controller']['time_horizon'] = horizon
                    data['dimensions']['R_p'] = well_scenario['R_p']
                    # overwrite file
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
                # Welfare
                welfare_folder = main(new_parameter_file, True, crops)
                # Group folders
                common_folder = "run_data/two_wells_analysis_dimless_longer_larger/long_horizon/finite_welfare/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
                shutil.move(welfare_folder, common_folder)
                
                os.remove(new_parameter_file)
                pbar.update(1)
    pbar.close()
    # TODO: myopic (horizon 0)

def RHG_analysis(parameter_file: str):
    well_scenarios = [
        {"wells_position": [[0, 780, 490], [0, 820, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 780, 490], [0, 860, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 740, 490], [0, 860, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 740, 490], [0, 900, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 700, 490], [0, 900, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 700, 490], [0, 940, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 660, 490], [0, 940, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 660, 490], [0, 980, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 620, 490], [0, 980, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         {"wells_position": [[0, 620, 490], [0, 1020, 490]],
         "to_player": [0, 1],
         "R_p": 200},
         
        # {"wells_position": [[0, 500, 490], [0, 580, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 500, 490], [0, 620, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 460, 490], [0, 620, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 240},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 4/24)},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 8/24)},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 12/24)},
        # {"wells_position": [[0, 460, 490], [0, 660, 490]],
        #  "to_player": [0, 1],
        #  "R_p": 200/(1+ 16/24)}
    ]
    demand_scenarios = [
        {"land": [1000, 1000]}
    ]
    # horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30]
    horizons = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    # horizons = [22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]
    
    crops = np.array([[1., 1.], [0., 0.], [0., 0.], [0., 0.]])
    dir, file = os.path.split(parameter_file)
    name, ext = file.split('.')
    new_parameter_file = dir + '/' + name + '_changed' + '.' + ext
    
    # correct_heads(new_parameter_file)
    
    counter = 0
    
    pbar = tqdm(total=(len(well_scenarios)*len(demand_scenarios)*len(horizons)),
                position=0, desc="RHG scenario", leave=True)
    for well_scenario in well_scenarios:
        wells = well_scenario['wells_position']
        for demand_scenario in demand_scenarios:
            for horizon in horizons:
                # if counter < 18:
                #     counter += 1
                #     pbar.update(1)
                #     continue
                shutil.copyfile(parameter_file, new_parameter_file)
                with open(new_parameter_file, 'r+') as f:
                    data = json.load(f)
                    # modify fields
                    data['wells']['pos'] = wells
                    data['wells']['to_player'] = well_scenario['to_player']
                    data['constraints']['max_pumping_rates'] = [10000000]*len(wells)
                    data['constraints']['well_depth'] = [-60.0]*len(wells)
                    data['land']['surface'] = demand_scenario['land']
                    data['controller']['time_horizon'] = horizon
                    data['dimensions']['R_p'] = well_scenario['R_p']
                    # overwrite file
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
                # RHG
                rhg_folder = main(new_parameter_file, False, crops)
                # Group folders
                common_folder = "run_data/two_wells_analysis_dimless_longer_larger/long_horizon/RHG/run_" + time.strftime("%Y%m%d-%H%M%S")+"/"
                shutil.move(rhg_folder, common_folder)
                
                os.remove(new_parameter_file)
                pbar.update(1)
    pbar.close()
    # TODO: myopic (horizon 0)
 
if __name__ == '__main__':
    # parameter_file = "Components/Setup/two_wells_analysis_redefinition_parameters.json"
    parameter_file = "/Users/leonardchanfreau/Documents/ETHZ/Master Thesis/code/GroundWater_RGH/Components/Setup/two_wells_analysis_dimensionless_parameters.json"
    #correct_heads()
    # main(parameter_file = "/Users/leonardchanfreau/Documents/ETHZ/Master Thesis/code/GroundWater_RGH/run_data/run_20250801-113404/model_parameters.json",
    #      use_welfare = False,
    #      crops=np.array([[1, 1, 1], [0, 0, 0], [0., 0., 0.], [0., 0., 0]]))
    # test_parameters(parameter_file = "Components/Setup/model_parameters_combination.json")
    # test_well_distance(parameter_file = "Components/Setup/test_scenarios_parameters.json")
    # test_three_aligned_wells(parameter_file = "Components/Setup/test_scenarios_parameters.json")
    # test_boundary_condition_impact(parameter_file = "Components/Setup/test_scenarios_parameters.json")
    # infinite_horizon_welfare_analysis(parameter_file = "Components/Setup/two_wells_analysis_parameters.json")
    finite_horizon_welfare_analysis(parameter_file)
    # RHG_analysis(parameter_file)