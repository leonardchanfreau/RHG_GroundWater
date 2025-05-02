import os
import numpy as np
import json
import flopy

from .System import System

class SimpleSquareExample(System):
    """
    Simulate a simple unconfined square aquifer using MODFLOW.
    """
    
    def __init__(self, param_file: str, boundary_condition: str="Free-flow"):
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
            
        # Store some parameters used to compute cost
        self.extraction_cost = params['costs']['extraction_cost']
        self.production_profit = params['costs']['production_profit']
        self.well_to_agent = np.array(params['wells']['to_player'])
        
        # Compute the spacing between cells
        delr = params['grid']['Lx'] / params['grid']['ncol']
        delc = params['grid']['Ly'] / params['grid']['nrow']
        delv = (params['grid']['ztop'] - params['grid']['zbot']) / params['grid']['nlay']
        # Compute the altitude of the bottom of each layer for each cell
        botm = np.linspace(params['grid']['ztop'], params['grid']['zbot'], params['grid']['nlay'] + 1)
        # Store boundary offset if needed
        self.offset = np.array([0, 0, 0])
        if boundary_condition == "Constant Head":
            self.offset = np.array([0, 1, 1])
        # Store the position of the wells
        self.well_pos = ((np.array(params['wells']['pos']) *\
            np.array([1, 1/delr, 1/delc])) + self.offset).tolist()
        
        # Variables for the BAS package (ibound specify the constant heads cells: value <0)     TODO: constant head boundary condition
        ibound = np.ones((params['grid']['nlay'] + 2*self.offset[0],
                          params['grid']['nrow'] + 2*self.offset[1],
                          params['grid']['ncol'] + 2*self.offset[2]), dtype=np.int32)
        # Constant head boundary condition
        if boundary_condition == "Constant Head":
            ibound[0,:,0] = -1; ibound[0,0,:] = -1; ibound[0,:,-1] = -1; ibound[0,-1,:] = -1
        self.state = params['init']['z'] * np.ones_like(ibound, dtype=np.float32)
        self.initial_state = self.state.copy()
        
        #Create Static flopy objects
        self.workspace = os.path.join(".", "mf_data", params['workspace'])
        self.name = f"{params['workspace']}_mf"
        self.mf = flopy.modflow.Modflow(self.name, exe_name="mf2005", model_ws=self.workspace)
        flopy.modflow.ModflowDis(
            self.mf,
            params['grid']['nlay'] + 2*self.offset[0],
            params['grid']['nrow'] + 2*self.offset[1],
            params['grid']['ncol'] + 2*self.offset[2],
            delr=delr,
            delc=delc,
            top=params['grid']['ztop'],
            botm=botm[1:],
            nper=1,
            perlen=[params['time_step']['dt']],
            nstp=[1],
            itmuni=params['time_step']['unit'],
            steady=[False],
        )
        self.bas = flopy.modflow.ModflowBas(self.mf, ibound=ibound, strt=self.state)
        flopy.modflow.ModflowLpf(
            self.mf, hk=params['physical']['hk'], vka=params['physical']['vka'],
            sy=params['physical']['sy'], ss=params['physical']['ss'],
            laytyp=params['physical']['laytyp'], ipakcb=53
        )                                                                       # TODO: adapt physical param unit to the time-step units
        flopy.modflow.ModflowPcg(self.mf)
        
        # Create the well package with null pumping rates
        # Zero-based layer, row, column indices
        pumping = np.hstack(
            (self.well_pos, np.zeros((len(self.well_pos),1)))).tolist()
        self.wel = flopy.modflow.ModflowWel(self.mf,
                                       stress_period_data=pumping)
        
        # Create the output control package
        flopy.modflow.ModflowOc(self.mf,
                                stress_period_data={(0, 0): ['SAVE HEAD', 'PRINT HEAD']},
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
            raise RuntimeError("MODFLOW did not terminate normally.")
        
        # Load output
        path = os.path.join(self.workspace, f"{self.name}.hds")
        hds = flopy.utils.HeadFile(os.path.join(path))
        self.state = hds.get_data(totim=1)
        
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
        cost = np.zeros((len(self.extraction_cost), 1))
        for v in range(len(self.extraction_cost)):
            well_indices = tuple(np.array(self.well_pos, dtype=int)\
                [self.well_to_agent==v].T.tolist())
            cost[v] = (self.extraction_cost[v] * (self.initial_state[well_indices] -\
                self.state[well_indices])[:, np.newaxis] * u[self.well_to_agent==v] + \
                    self.production_profit[v] * u[self.well_to_agent==v]).sum()
        return cost