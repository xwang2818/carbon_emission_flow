"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
潮流分布矩阵计算部分
"""

import numpy as np
from pypower.api import runpf, ppoption

def mpc_to_power_flow_analysis(mpc: dict) -> dict:
    """
    用于将 mpc 系统数据输入潮流计算器并返回一次潮流计算的完整结果
    :param mpc: 系统数据字典，包含 bus、branch、gen、baseMVA 等字段
    :return: 潮流计算结果字典（与 pypower.runpf 的结果结构一致）
    """
    ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)  # 设定潮流计算选项：牛顿法、静默输出、关闭详细打印
    # ppopt = ppoption(PF_ALG=1, VERBOSE=0)          # 如需打印可改为此行
    power_flow_analysis_res, success = runpf(mpc, ppopt)  # 执行一次潮流计算，返回结果与是否成功标志
    return power_flow_analysis_res                         # 仅返回完整结果结构体（保持原有行为）



def get_pk_arr(results: dict, pg_arr: np.array) -> np.array:
    """
    用于计算各节点的流过功率 P_k，并在 pg_arr 的基础上回填到第二列
    :param results: 潮流计算结果字典（含 'branch' 等关键数组）
    :param pg_arr: 发电机注入功率数组（第一列为节点编号，第二列为该节点发电注入功率）
    :return: 在第二列写入 P_k 的数组（与 pg_arr 形状一致）
    """
    # 计算规则：P_k = |节点流入功率之和| + 节点发电注入功率
    pk_arr = pg_arr.copy()                                  # 复制一份作为结果承载（避免覆盖原始输入）
    pji_arr = results['branch'][:, [0, 1, 15]]              # 取出支路表中的 from、to、P_ji 列（假定第 15 列为 P_ji）

    for idx, bus_id in enumerate(pk_arr[:, 0]):             # 遍历每个节点（以 pk_arr 的行定义为准）
        mask = pji_arr[:, 1] == bus_id                      # 选择所有“流入该节点”的支路（to == bus_id）
        if mask.any():                                      # 若存在流入支路
            pij_abs_sum = np.sum(np.abs(pji_arr[mask, -1])) # 对这些支路的 P_ji 求绝对值并求和
            pk_arr[idx, 1] += pij_abs_sum                   # P_k = 发电注入功率（已有）+ 流入功率绝对值之和

    return pk_arr                                           # 返回写入了 P_k 的结果数组



def get_au(results: dict, pk_arr: np.array) -> np.array:
    """
    本函数用于依据潮流结果与各节点流过功率构建潮流分布矩阵 A_u
    :param results: 潮流计算结果字典（需包含 'bus' 与 'branch' 等键）
    :param pk_arr: 各节点流过功率数组（第二列为 P_k，行顺序与节点编号一致）
    :return: 潮流分布矩阵 A_u
    """
    shape = results['bus'].shape[0]                   # 节点数量（用于确定 A_u 的规模）
    pji_arr = results['branch'][:, [0, 1, 15]]        # 取出支路表的 from、to、P_ji 列（假定第 15 列为 P_ji）
    au = np.eye(shape)                                # 初始化为单位阵（主对角线为 1）

    # 逐支路填充 A_u 的非对角元素
    for i in pji_arr:
        bus_from = np.int32(i[0])                     # from 节点编号（1 基）
        bus_to = np.int32(i[1])                       # to 节点编号（1 基）
        au[bus_to - 1, bus_from - 1] = - np.abs(i[-1]) / pk_arr[bus_from - 1, 1]  # A_u[to, from] = -|P_ji| / P_k(from)

    return au                                         # 返回构造完成的 A_u

def get_pg_arr(results: dict) -> np.array:
    """
    本函数用于从潮流计算结果中提取各节点的发电机注入功率数组
    :param results: 潮流计算结果字典（含 'bus'、'gen' 等数组）
    :return: 两列数组：[节点编号, 发电机注入功率]
    """
    shape = results['bus'].shape[0]          # 节点数量（用于确定结果数组规模）
    pg = np.zeros((shape, 2))                # 预分配结果数组：第一列节点编号，第二列发电机注入功率
    pg[:, 0] = results['bus'][:, 0]          # 写入节点编号（与 bus 表一致）
    _gen = results['gen']                    # 取出发电机表
    _gen_idx = np.int32(_gen[:, 0] - 1)      # 发电机所在节点的 0 基索引
    pg[_gen_idx, 1] = _gen[:, 1]             # 将各发电机出力写入对应节点的第二列
    return pg                                # 返回节点-发电出力数组


def get_plk_arr(results: dict) -> np.array:
    """
    本函数用于从潮流计算结果中提取各节点的有功负荷数组
    :param results: 潮流计算结果字典（含 'bus' 数组）
    :return: 两列数组：[节点编号, 节点有功负荷 P_Lk]
    """
    return results['bus'][:, [0, 2]]         # 直接从 bus 表抽取节点编号与有功负荷列


def get_pkj_and_loss(results: dict) -> np.array:
    """
    本函数用于从潮流计算结果中提取支路有功功率与网损功率，并合成为结果数组
    :param results: 潮流计算结果字典（含 'branch' 数组）
    :return: 四列数组：[from 节点, to 节点, P_kj, P_kj_loss]
    """
    _pkj_and_loss = results['branch'][:, [0, 1, 13, 15]]  # 取出支路的 from、to、P_kj、P_ji（或损耗相关）列
    pkj_and_loss = _pkj_and_loss                          # 建立视图（保持原列顺序）
    pkj_and_loss[:, -1] = _pkj_and_loss[:, -2] + _pkj_and_loss[:, -1]  # 最后一列作为“网损功率”= 第三列 + 第四列
    return pkj_and_loss                                   # 返回支路有功功率与网损功率数组


def get_power_flow_paras(mpc: dict) -> dict:
    """
    本函数用于基于一次潮流计算，整理后续碳流计算所需的关键数组与矩阵
    :param mpc: 系统数据结构（包含 bus、branch、gen、baseMVA 等）
    :return: 参数字典，含键：
        - 'pg_arr'：发电机注入功率数组
        - 'plk_arr'：节点有功负荷数组
        - 'pkj_and_loss'：支路有功与网损功率数组
        - 'pk_arr'：节点流过功率数组
        - 'au'：潮流分布矩阵 A_u
        - 'original_power_anlysis_res'：原始潮流计算结果
    """
    p_a_res = mpc_to_power_flow_analysis(mpc)        # 执行潮流计算，获得完整结果
    power_flow_paras = {}                            # 汇总容器
    power_flow_paras['pg_arr'] = get_pg_arr(p_a_res)                 # 节点-发电出力
    power_flow_paras['plk_arr'] = get_plk_arr(p_a_res)               # 节点-有功负荷
    power_flow_paras['pkj_and_loss'] = get_pkj_and_loss(p_a_res)     # 支路有功与网损
    power_flow_paras['pk_arr'] = get_pk_arr(p_a_res, power_flow_paras['pg_arr'])  # 节点流过功率
    power_flow_paras['au'] = get_au(p_a_res, power_flow_paras['pk_arr'])          # 潮流分布矩阵
    power_flow_paras['original_power_anlysis_res'] = p_a_res         # 保存原始潮流计算结果
    return power_flow_paras                                          # 返回参数集合



