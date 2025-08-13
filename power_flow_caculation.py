"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
潮流分布矩阵计算部分
"""

import numpy as np
from pypower.api import runpf, ppoption

def mpc_to_power_flow_analysis(mpc):
    ppopt = ppoption(PF_ALG=1, VERBOSE=0)  # 牛顿法
    # 计算潮流
    power_flow_analysis_res, success = runpf(mpc, ppopt)
    return power_flow_analysis_res

def bus_idx_shape(results):
    n = results['bus'].shape[0]
    bus_idx = {int(results['bus'][i, 0]): i for i in range(n)}
    return n, bus_idx

def compute_au_pi(results, n, bus_idx):
    """
    通用 Au 计算：
    Au[j,i] = -|p_ji| / Pi
    Pi 规则：
        负荷节点 i → Pi = Σ|k→i| + Pd_i
        发电机节点 → Pi = Σ|i→k|
    p_ji = 下游节点实际得到功率（|To Bus Injection|）
    """
    # n = results['bus'].shape[0]
    # bus_idx = {int(results['bus'][i, 0]): i for i in range(n)}
    Au = np.eye(n)

    # 1) 标记发电机节点
    gen_buses = {int(g[0]) for g in results['gen']}

    # 2) 计算 Pi
    Pi = np.zeros(n)

    # 2-a 先加负荷
    for b in range(n):
        Pi[b] += results['bus'][b, 2]  # Pd

    # 2-b 再按支路累加
    for br in results['branch']:
        f, t = int(br[0]), int(br[1])
        P_abs = abs(br[13])

        # if f in gen_buses:
        #     # 发电机：加到出端
        Pi[bus_idx[f]] += P_abs
        # else:
            # 负荷：加到入端
            # Pi[bus_idx[t]] += P_abs

    # 3) 填 Au
    for br in results['branch']:
        f, t = int(br[0]), int(br[1])
        Pft   = br[13]
        P_abs = abs(br[15])

        if abs(Pft) < 1e-6:
            continue

        # 上游节点判定
        if Pft > 0:  # f→t ⇒ f 是 t 的上游
            upstream, downstream = bus_idx[f], bus_idx[t]
        else:        # t→f ⇒ t 是 f 的上游
            upstream, downstream = bus_idx[t], bus_idx[f]

        if Pi[upstream] > 1e-6:
            Au[downstream, upstream] = -P_abs / Pi[upstream]

    return Au, Pi

def pg(results, n):
    n = results['bus'].shape[0]
    pg = np.zeros(n)
    _gen = results['gen']
    _gen_idx = np.int32(_gen[:, 0] - 1)
    pg[_gen_idx] = _gen[:, 1]
    return pg

def plk(results):
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
    shape, bus_idx = bus_idx_shape(p_a_res)
    pg = pg(p_a_res, shape)
    plk = plk(p_a_res)
    print()