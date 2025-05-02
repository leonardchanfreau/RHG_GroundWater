from Components.RecedingHorizon import RecedingHorizon
from Components.System import SimpleSquareExample
from Components.Model import SquareCoarseModel
from Components.Controllers import LTI_LQRc_RHG


def main (parameter_file: str):
    # Initialize System and Model
    system = SimpleSquareExample(parameter_file, boundary_condition="Constant Head")
    model = SquareCoarseModel(parameter_file, boundary_condition="Constant Head")
    # Initialize Controller
    controller = LTI_LQRc_RHG(
        parameter_file,
        model.A.toarray(), [B.toarray() for B in model.B_v], model.C,
        model.H_x, model.h_x, model.H_u_v, model.h_u_v,
        Q_v = None, R_v = None, M_v = [G.toarray() for G in model.Gamma_v], L_v = [Xi for Xi in model.Xi_v],
        solver='fbrs'
    )
    # Initialize receding horizon
    receding_horizon = RecedingHorizon(system, model, controller, parameter_file)
    # Run loop
    receding_horizon.run(time_period=200, display_results=True)
    
 
if __name__ == '__main__':
    main(parameter_file = "Components/Setup/aquifer_parameters.json")