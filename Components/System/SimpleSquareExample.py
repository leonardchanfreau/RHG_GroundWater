import os
import numpy as np
import json
import flopy

from ..ParameterInterface import ParameterInterface
from .System import System

class SimpleSquareExample(System):
    """
    Simulate a simple unconfined square aquifer using MODFLOW.
    """
    
    def __init__(self, param_handler: ParameterInterface, boundary_condition: str="Free-flow"):
        # Store some parameters used to compute cost
        self.extraction_cost, self.production_profit = param_handler.get_cost_coefficients()
        self.well_to_agent = param_handler.well_to_agent
        
        # Compute the spacing between cells
        delr = param_handler.Lx / param_handler.mf_n_col
        delc = param_handler.Ly / param_handler.mf_n_row
        delv = (param_handler.ztop - param_handler.zbot) / param_handler.mf_n_lay
        # Compute the altitude of the bottom of each layer for each cell
        botm = np.linspace(param_handler.ztop, param_handler.zbot, param_handler.mf_n_lay + 1)
        # Store boundary offset if needed
        self.offset = np.array([0, 0, 0])
        if boundary_condition == "Constant Head":
            self.offset = np.array([0, 1, 1])
        # Store the position of the wells
        self.well_pos = ((np.array(param_handler.well_coords) *\
            np.array([1, 1/delr, 1/delc])) + self.offset).tolist()
        
        # Variables for the BAS package (ibound specify the constant heads cells: value <0)     TODO: constant head boundary condition
        ibound = np.ones((param_handler.mf_n_lay + 2*self.offset[0],
                          param_handler.mf_n_row + 2*self.offset[1],
                          param_handler.mf_n_col + 2*self.offset[2]), dtype=np.int32)
        # Constant head boundary condition
        if boundary_condition == "Constant Head":
            ibound[0,:,0] = -1; ibound[0,0,:] = -1; ibound[0,:,-1] = -1; ibound[0,-1,:] = -1
        self.state = param_handler.init_heads * np.ones_like(ibound, dtype=np.float32)
        self.initial_state = self.state.copy()
        # Time step parameter
        self.dt = param_handler.dt
        
        #Create Static flopy objects
        self.workspace = os.path.join(".", "mf_data", param_handler.mf_workspace)
        self.name = f"{param_handler.mf_workspace}_mf"
        self.mf = flopy.modflow.Modflow(self.name, exe_name="mf2005", model_ws=self.workspace)
        flopy.modflow.ModflowDis(
            self.mf,
            param_handler.mf_n_lay + 2*self.offset[0],
            param_handler.mf_n_row + 2*self.offset[1],
            param_handler.mf_n_col + 2*self.offset[2],
            delr=delr,
            delc=delc,
            top=param_handler.ztop,
            botm=botm[1:],
            nper=1,
            perlen=[self.dt],
            nstp=[5],
            itmuni=param_handler.mf_time_unit,
            steady=[False],
        )
        conductivity, storativity, specific_storage = param_handler.get_physical_parameters()
        self.bas = flopy.modflow.ModflowBas(self.mf, ibound=ibound, strt=self.state)
        flopy.modflow.ModflowLpf(
            self.mf, hk=conductivity, vka=param_handler.vka,
            sy=storativity, ss=specific_storage,
            laytyp=param_handler.mf_layer_type, ipakcb=53
        )                                                                       # TODO: adapt physical param unit to the time-step units
        # flopy.modflow.ModflowPcg(self.mf)
        flopy.modflow.ModflowPcg(
            self.mf,
            hclose=1e-6,     # tighten if needed
            rclose=1e-3,     # tighten if needed
            mxiter=200,      # increase iteration budget
            iter1=50         # inner iterations
        )
        
        # Create the well package with null pumping rates
        # Zero-based layer, row, column indices
        pumping = np.hstack(
            (self.well_pos, np.zeros((len(self.well_pos),1)))).tolist()
        self.wel = flopy.modflow.ModflowWel(self.mf,
                                       stress_period_data=pumping)
        
        # Create the output control package
        flopy.modflow.ModflowOc(self.mf,
                                stress_period_data={(0, 4): ['SAVE HEAD', 'PRINT HEAD']},
                                compact=True)
    
    
    def apply_command(self, u: np.ndarray) -> np.ndarray:
        # u is of size K * nu and nu should be 1
        if u.shape[1] != 1:
            raise AttributeError("Command should be of dimension 1: pumping rate")
        
        # Set initial heads
        self.bas.strt = self.state
        
        # Form pumping command
        pumping = np.hstack((self.well_pos, u)).tolist()
        self.wel.stress_period_data = {0: pumping}
        
        # Run the model
        self.mf.write_input()
        success, _ = self.mf.run_model(silent=True, pause=False)
        
        if not success:
            # for some reason the convergence is really sensible to the pumping rates
            u += np.random.rand(u.shape[0], u.shape[1])/10
            pumping = np.hstack((self.well_pos, u)).tolist()
            self.wel.stress_period_data = {0: pumping}
            self.mf.write_input()
            success, _ = self.mf.run_model(silent=True, pause=False)
            if not success:
                raise RuntimeError("MODFLOW did not terminate normally.")
        
        # Load output
        path = os.path.join(self.workspace, f"{self.name}.hds")
        hds = flopy.utils.HeadFile(os.path.join(path))
        self.state = hds.get_data(totim=1*self.dt)
        
        return self.state
    
    
    def get_state(self) -> np.ndarray:
        slices = tuple(slice(mi, ma) for mi, ma in \
            zip(self.offset, self.state.shape - self.offset))
        return self.state[slices]
    
    def get_heads_at_wells(self) -> np.ndarray:
        return self.state[tuple(np.array(self.well_pos, dtype=int).T.tolist())]
    
    def get_well_position(self) -> np.ndarray:
        return np.array(self.well_pos, dtype=int) - self.offset
    
    def compute_cost(self, u: np.ndarray) -> np.ndarray:
        """ Depreciated: forgot to update after refactor, should be adapted to
        the new cost model interface.
        In particular, the yield profit should be multiplied bu the time-varying
        yield factor."""
        cost = np.zeros((len(self.extraction_cost), 1))
        for v in range(len(self.extraction_cost)):
            well_indices = tuple(np.array(self.well_pos, dtype=int)\
                [self.well_to_agent==v].T.tolist())
            cost[v] = (-self.extraction_cost[v] * (self.initial_state[well_indices] -\
                self.state[well_indices])[:, np.newaxis] * u[self.well_to_agent==v] + \
                    self.production_profit[v] * u[self.well_to_agent==v]).sum()
        return cost