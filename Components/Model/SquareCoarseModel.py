from typing import Tuple, Dict, List

import math
import numpy as np
import scipy.sparse as sp

from scipy.signal import convolve2d

from .CoarseModel import CoarseModel

class SquareCoarseModel(CoarseModel):
    
    def __init__(self, param_file: str, boundary_condition: str="Free-flow"):
        super().__init__(param_file, boundary_condition)

    
    def _get_grid_parameter_matrices(self, params) -> \
        Tuple[sp.csr_array, sp.csr_array, sp.csr_array, sp.csr_array]:
        # Compute adjacency matrices
        horizontal_adjacency = sp.kron(
            sp.eye(self.n_row),
            sp.diags([1, 1], offsets=[-1, 1], shape=(self.n_col, self.n_col))).tocsr()
        vertical_adjacency = sp.kron(
            sp.diags([1, 1], offsets=[-1, 1], shape=(self.n_row, self.n_row)),
            sp.eye(self.n_col)).tocsr()
        adjacency = horizontal_adjacency + vertical_adjacency
        # Compute horizontal parameters
        dlx, dly = params['grid']['Lx'] / self.n_col, params['grid']['Ly'] / self.n_row
        distance_kl = dlx * horizontal_adjacency + dly * vertical_adjacency
        surface_k = (dlx * dly) * sp.csr_array(np.ones((self.n_row, self.n_col)))
        # Compute vertical cross-sections
        b = params['grid']['ztop'] - params['grid']['zbot']
        if params['physical']['type'] == "uniform":
            cross_section_kl = b * dly * horizontal_adjacency + \
                b * dlx * vertical_adjacency
        elif params['physical']['type'] == "variable":
            # Compute average depth along horizontal and vertical links
            data = self._average_value(b).reshape((1, self.n_col*self.n_row))
            b_avg_h = (horizontal_adjacency*data + horizontal_adjacency*data.T)/2
            b_avg_v = (vertical_adjacency*data + vertical_adjacency*data.T)/2
            cross_section_kl = b_avg_h * dly + b_avg_v * dlx
        return adjacency, distance_kl, surface_k, cross_section_kl.tocsr()
    
    
    def _get_boundary_parameters(self, params) -> Tuple[sp.csr_array, sp.csr_array]:
        # Compute horizontal distance to virtual boundary cells
        dlx, dly = params['grid']['Lx'] / self.n_col, params['grid']['Ly'] / self.n_row
        distance_bound = [
            dly * np.ones((1, self.n_row)),
            dlx * np.ones((1, self.n_col)),
            dly * np.ones((1, self.n_row)),
            dlx * np.ones((1, self.n_col)),
        ]
        # Compute vertical cross sections
        b = params['grid']['ztop'] - params['grid']['zbot']     # TODO: manage case where b is an array
        cross_section_bound = [
            b * dlx * np.ones((1, self.n_row)),
            b * dly * np.ones((1, self.n_col)),
            b * dlx * np.ones((1, self.n_row)),
            b * dly * np.ones((1, self.n_col)),
        ]
        return distance_bound, cross_section_bound
    
    def _average_value(self, var_HD: np.ndarray) -> np.ndarray:
        # TODO: adapt method to manage other original geometries
        # Compute discretization ratio
        n_x, n_y = var_HD.shape
        x_ratio = n_x / self.n_col
        y_ratio = n_y / self.n_row
        # Construct averaging filter
        avg_filter = np.ones((math.floor(x_ratio), math.floor(y_ratio)))
        avg_filter /= np.sum(avg_filter)
        # Apply the filter
        var = convolve2d(var_HD, avg_filter, mode="valid")[::math.ceil(x_ratio), ::math.ceil(y_ratio)]
        return var
    
    
    def _compute_recharge(self,
                          recharge_coord: Dict[Tuple[int, int], float],
                          Lx: float, Ly: float) -> sp.csr_array:
        recharge = sp.lil_array((self.n_row, self.n_col))
        for coord, r in recharge_coord.items():
            row = int(coord[0]*self.n_row/Lx)
            col = int(coord[1]*self.n_row/Ly)
            recharge[row, col] += r
        return recharge.tocsr()
    
    
    def _compute_well_to_cell(self, well_pos: List[List[int]],
                              Lx: float, Ly: float) -> sp.csr_array:
        # Construct well to cell mapping
        mapping = sp.lil_array((self.n_row * self.n_col, len(well_pos)))
        for k, coord in enumerate(well_pos):
            row = int(coord[1]*self.n_row/Lx)
            col = int(coord[2]*self.n_col/Ly)
            mapping[row * self.n_col + col, k] += 1
        return mapping.tocsr()
    
    
    def _compute_input_to_state(self,
                                n_player: int, to_player: List[int]) -> \
                                    List[sp.csr_array]:
        # Aggregate wells on the same cell
        unique, index, count =\
            np.unique(self.well_to_cell.toarray(), axis=1, return_counts=True, return_index=True)
        mapping_agg = unique*count
        to_player_agg = np.array(to_player)[index]
        # Construct mapping for every player
        mapping_v = []
        for i in range(n_player):
            mapping_v.append(
                sp.csr_array(mapping_agg[:, to_player_agg==i]).minimum(1)
            )
        return mapping_v
    
    
    def _compute_input_to_well(self) -> sp.csr_array:
        # Compute input to well mapping
        input_to_well = self.well_to_cell.T @ sp.hstack(self.input_to_state_v, dtype=np.float64)
        return input_to_well