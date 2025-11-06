import os
import numpy as np
import pandas as pd

def get_rain_data(data_file: str, year: int, state: str, county: str):
    if data_file is None:
        return np.zeros((365,))
    # Check is query has already been made
    file_name, ext = os.path.splitext(data_file)
    query_file =( file_name+f'_{state}_{county}_{year}'+ext).replace(" ", "_")
    if os.path.exists(query_file):
        return np.loadtxt(query_file)
    df = pd.read_csv(data_file)
    query = df.query(f'ST == "{state}" and ct_wb == "{county}" and yr == {year}')["prcp_wb"].values
    # Save query to win time in future runs
    np.savetxt(query_file, query)
    return query

def get_available_water(data_file: str, year: int, state: str, county: str):
    rain = get_rain_data(data_file, year, state, county)
    return np.maximum((rain/1.25) - 25/30, 0.)

def spread_rain(water_requirement: np.ndarray, storage_coef: float) -> np.ndarray:
    """Time dimension must be the last."""
    res = water_requirement.copy()
    if res.shape[-1] > 1: 
        while res[..., :-1].min() < 0:
            shift = [(0,0)] * (water_requirement.ndim - 1) + [(1,0)]
            shifted_x = np.pad(res, shift, mode='constant')[..., :-1]
            tmp = np.where(shifted_x>=0, res, res + storage_coef*shifted_x)
            res = np.where(res >= 0, tmp, tmp-res)
    return np.maximum(res, 0.)