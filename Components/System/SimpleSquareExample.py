import os
import numpy as np
import json
import flopy

class SimpleSquareExample:
    """
    Simulate a simple unconfined square aquifer using MODFLOW.
    """
    
    def __init__(self, param_file: str):
        # Read parameter file
        with open(param_file, 'r') as file:
            params = json.load(file)
        
        # Compute the spacing between cells
        delr = params.grid.Lx / params.grid.ncol
        delc = params.grid.Ly / params.grid.nrow
        delv = (params.grid.ztop - params.grid.zbot) / params.grid.nlay
        # Compute the altitude of the bottom of each layer for each cell
        botm = np.linspace(params.grid.ztop, params.grid.zbot, params.grid.nlay + 1)
        # Store the position of the wells
        self.well_pos = params.well.pos
        
        # Variables for the BAS package (ibound specify the constant heads cells: value <0)     TODO: constant head boundary condition
        ibound = np.ones((params.grid.nlay, params.grid.nrow, params.grid.ncol), dtype=np.int32)
        self.state = params.init.z * \
            np.ones((params.grid.nlay, params.grid.nrow, params.grid.ncol), dtype=np.float32)
        
        #Create Static flopy objects
        self.workspace = os.path.join(".", "mf_data", params.workspace)
        self.name = f"{params.workspace}_mf"
        self.mf = flopy.modflow.Modflow(self.name, exe_name="mf2005", model_ws=self.workspace)
        flopy.modflow.ModflowDis(
            self.mf,
            params.grid.nlay,
            params.grid.nrow,
            params.grid.ncol,
            delr=delr,
            delc=delc,
            top=params.grid.ztop,
            botm=botm[1:],
            nper=1,
            perlen=[params.time_step.dt],
            nstp=[1],
            itmuni=params.time_step.unit,
            steady=[False],
        )
        self.bas = flopy.modflow.ModflowBas(self.mf, ibound=ibound, strt=self.state)
        flopy.modflow.ModflowLpf(
            self.mf, hk=params.physical.hk, vka=params.physical.vka,
            sy=params.physical.sy, ss=params.physical.ss,
            laytyp=params.physical.laytyp, ipakcb=53
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
    
    
    def apply_command(self, u: np.ndarray):
        # u is of size K * nu and nu should be 1
        if u.shape[1] != 1:
            raise AttributeError("Command should be of dimension 1: pumping rate")
        
        # Set initial heads
        self.bas.strt = self.state
        
        # Form pumping command
        pumping = np.hstack((self.well_pos, u)).tolist()
        self.wel.stress_period_data = pumping
        
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