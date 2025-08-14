"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
潮流分布矩阵计算部分
"""

import numpy as np
from pypower.api import runpf, ppoption

def mpc_to_power_flow_analysis(mpc):
    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)  # 牛顿法
    # ppopt = ppoption(PF_ALG=1, VERBOSE=0)  # 牛顿法
    # 计算潮流
    power_flow_analysis_res, success = runpf(mpc, ppopt)
    return power_flow_analysis_res


def get_pk_arr(results, pg_arr):
    """
    pk = 绝对值（节点流入功率） + 节点发动机的注入功率
    """
    # 2) 计算 pk_arr
    pk_arr = pg_arr.copy()
    pji_arr = results['branch'][:, [0, 1, 15]]

    for idx, bus_id in enumerate(pk_arr[:, 0]):
        mask = pji_arr[:, 1] == bus_id
        if mask.any():
            pij_abs_sum = np.sum(np.abs(pji_arr[mask, -1]))
            pk_arr[idx, 1] += pij_abs_sum

    return pk_arr



    # 2-b 再按支路累加
def get_au(results, pk_arr):
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
    return results['bus'][:, [0, 2]]

def get_pkj_and_loss(results):
    _pkj_and_loss = results['branch'][:, [0, 1, 13, 15]]
    pkj_and_loss = _pkj_and_loss
    pkj_and_loss[:, -1] = _pkj_and_loss[:, -2] + _pkj_and_loss[:, -1]
    return pkj_and_loss

def get_power_flow_paras(mpc):
    p_a_res = mpc_to_power_flow_analysis(mpc)
    power_flow_paras = {}
    power_flow_paras['pg_arr'] = get_pg_arr(p_a_res)
    power_flow_paras['plk_arr'] = get_plk_arr(p_a_res)
    power_flow_paras['pkj_and_loss'] = get_pkj_and_loss(p_a_res)
    power_flow_paras['pk_arr'] = get_pk_arr(p_a_res, power_flow_paras['pg_arr'])
    power_flow_paras['au'] = get_au(p_a_res, power_flow_paras['pk_arr'])
    power_flow_paras['original_power_anlysis_res'] = p_a_res
    return power_flow_paras


