from Components.RecedingHorizon import RecedingHorizon
from Components.System import SimpleSquareExample
from Components.Model import SquareCoarseModel
from Components.Controllers import LTI_LQRc_RHG


def main (parameter_file: str):
    # Initialize System and Model
    system = SimpleSquareExample(parameter_file)
    model = SquareCoarseModel(parameter_file)
    # Initialize Controller
    controller = LTI_LQRc_RHG(
        parameter_file,
        model.A.toarray(), [B.toarray() for B in model.B_v], [B.toarray() for B in model.B_minus_v], model.C,
        model.H_x, model.h_x, model.H_u, model.h_u,
        Q_v = None, R_v = None, M_v = [G.toarray() for G in model.Gamma_v], L_v = [Xi for Xi in model.Xi_v]
    )
    # Initialize receding horizon
    receding_horizon = RecedingHorizon(system, model, controller)
    # Run loop
    receding_horizon.run(time_period=52)
    
 
if __name__ == '__main__':
    main(parameter_file = "Components/Setup/aquifer_parameters.json")