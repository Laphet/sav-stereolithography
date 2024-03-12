import numpy as np
from itertools import product
from sav_solver import Solver
from datetime import datetime

w_0 = 0.01
I_max = 10.0
t_check_pnts = [0.25, 0.5, 0.75]
coord_pnts = np.array(
    [[0.25, 0.5 + 1.0 / 3], [0.5, 0.5], [0.5, 0.5 - 1.0 / 3], [0.5, 0.5 + 1.0 / 3]]
)


# Moving center
def heat_source(x, y, t):
    if t <= t_check_pnts[0]:
        delta_t = t / t_check_pnts[0]
        center = (1.0 - delta_t) * coord_pnts[0] + delta_t * coord_pnts[1]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0)
    elif t_check_pnts[0] < t <= t_check_pnts[1]:
        delta_t = (t - t_check_pnts[0]) / (t_check_pnts[1] - t_check_pnts[0])
        center = (1.0 - delta_t) * coord_pnts[1] + delta_t * coord_pnts[2]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0)
    elif t_check_pnts[1] < t <= t_check_pnts[2]:
        delta_t = (t - t_check_pnts[1]) / (t_check_pnts[2] - t_check_pnts[1])
        center = (1.0 - delta_t) * coord_pnts[2] + delta_t * coord_pnts[1]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0)
    elif t > t_check_pnts[2]:
        delta_t = (t - t_check_pnts[2]) / (1.0 - t_check_pnts[2])
        center = (1.0 - delta_t) * coord_pnts[1] + delta_t * coord_pnts[3]
        return I_max * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / w_0)
