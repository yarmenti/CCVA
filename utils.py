import numpy as np


def time_offseter(time, time_grid_ref, left=False, max_value=1000):
    offset = -1 if left else 0
    idx = np.searchsorted(time_grid_ref, time, side='right') + offset

    if idx == len(time_grid_ref):
        return max_value

    return time_grid_ref[idx]

v_time_offseter = np.vectorize(time_offseter, otypes=[np.float], excluded=set((1, "left", "max_value")))