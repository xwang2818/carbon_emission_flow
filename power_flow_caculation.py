"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
潮流分布矩阵计算部分
"""

import numpy as np
from pypower.api import runpf, ppoption

def mpc_to_power_flow_analysis(mpc):
    # ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)  # 牛顿法
    ppopt = ppoption(PF_ALG=1, VERBOSE=0)  # 牛顿法
    # 计算潮流
    power_flow_analysis_res, success = runpf(mpc, ppopt)
    return power_flow_analysis_res


def compute_pk(results, pg_arr):
    """
    pk = 绝对值（节点流入功率） + 节点发动机的注入功率
    """
    # 2) 计算 pk_arr
    pk_arr = pg_arr
    pji_arr = results['branch'][:, [0, 1, 15]]

    for idx, bus_id in enumerate(pk_arr[:, 0]):
        mask = pji_arr[:, 1] == bus_id
        if mask.any():
            pij_abs_sum = np.sum(np.abs(pji_arr[mask, -1]))
            pk_arr[idx, 1] += pij_abs_sum

    return pk_arr



    # 2-b 再按支路累加
def compute_au(results, pk_arr):
    shape = results['bus'].shape[0]
    pji_arr = results['branch'][:, [0, 1, 15]]
    au = np.eye(shape)
    # 3) 填 Au
    for i in pji_arr:
        bus_from = np.int32(i[0])
        bus_to = np.int32(i[1])
        au[bus_to - 1, bus_from - 1] = - np.abs(i[-1]) / pk_arr[bus_from - 1, 1]
    return au

def get_pg_arr(results):
    shape = results['bus'].shape[0]
    pg = np.zeros((shape, 2))
    pg[:, 0] = results['bus'][:, 0]
    _gen = results['gen']
    _gen_idx = np.int32(_gen[:, 0] - 1)
    pg[_gen_idx, 1] = _gen[:, 1]
    return pg

def get_plk_arr(results):
    return results['bus'][:, 2]




if __name__ == '__main__':
    mpc = {
    'version': '2',
    'baseMVA': 100,
    'bus': np.array([
        # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
        [1, 3, 0, 0, 0, 0, 1, 1.00, 0, 230, 1, 1.1, 0.9],
        [2, 2, 0, 0, 0, 0, 1, 1.00, 0, 230, 1, 1.1, 0.9],
        [3, 1, 200, 80, 0, 0, 1, 1.00, 0, 230, 1, 1.1, 0.9],
        [4, 1, 300, 120, 0, 0, 1, 1.00, 0, 230, 1, 1.1, 0.9],
    ]),
    'gen': np.array([
        # bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin
        [1, 0, 0, 0, 0, 1.025, 100, 1, 600, 0],
        [2, 114.0000, 0, 0, 0, 1.00, 100, 1, 600, 0],
    ]),
    'branch': np.array([
        # fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
        [1, 2, 0.01938, 0.05917, 0.0528, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 3, 0.01938, 0.05917, 0.0528, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 4, 0.01938, 0.05917, 0.0528, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 3, 0.01938, 0.05917, 0.0528, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 4, 0.01938, 0.05917, 0.0528, 250, 250, 250, 0, 0, 1, -360, 360],
    ])
}
    p_a_res = mpc_to_power_flow_analysis(mpc)
    # shape, bus_idx = bus_idx_shape(p_a_res)
    pg_arr = get_pg_arr(p_a_res)
    # plk_arr = get_plk_arr(p_a_res)
    pk_arr = compute_pk(p_a_res, pg_arr)
    au = compute_au(p_a_res, pk_arr)
    print()