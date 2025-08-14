"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
碳流计算部分
"""


# TODO 计算eg

from power_flow_caculation import get_power_flow_paras
import numpy as np
"""
先使用numpy编程，后期如果需要使用torch联动，修改比较快
"""

def get_eg(pg_arr, unit_paras):
    mask = pg_arr[:, 1] != 0
    _pg = pg_arr[mask]
    pg_value = _pg[:, 1]
    pg_2 = np.diag(pg_value)
    pg_1 = np.eye(2)
    pg_0 = np.diag(1 / pg_value)
    a = unit_paras[:, 0].reshape(2, -1)
    b = unit_paras[:, 1].reshape(2, -1)
    c = unit_paras[:, 2].reshape(2, -1)
    zeta = unit_paras[:, 3]
    eta = unit_paras[:, 4]
    xi = unit_paras[:, 5]
    mu = unit_paras[:, 6]
    _w = pg_2 @ a + pg_1 @ b + pg_0 @ c
    w = _w.reshape(-1) * zeta
    eg = eta * xi * (1 - mu) * (44 / 12) * w
    eg_arr = pg_arr.copy()
    eg_arr[mask, -1] = eg
    return eg_arr


def _r_l(k: int, eg: np.array, au: np.array, pg: np.array):
    """
    本函数用于计算所有碳流计算均需要的矩阵乘法部分
    k: 节点编号，从0开始
    eg: 发电机的碳排放强度，向量
    au: 潮流分布矩阵
    pg: 发电机有功功率，除发电机节点外取0，向量
    """
    n = pg.shape[0]
    au_inverse = np.linalg.inv(au)
    pg_diag = np.diag(pg)
    ek_t = np.zeros((1,n))
    ek_t[0, k] = 1
    return float(ek_t @ au_inverse @ pg_diag @ eg)


def branch_r_l(k: int, j: int, pkj: float, pk: float, _r_l_):
    """
    本函数用于计算支路碳流率
    k: form节点编号，从0开始，即0对应节点编号1
    j: to节点编号，空置变量
    pkj: 支路传输功率，MW
    pk: 节点流过功率，MW
    """
    print(f'支路{k}-{j}的碳排放率计算完毕')
    return float(pkj / pk * _r_l_)

def bus_r_l(k: int, plk: float, pk: float, _r_l_):
    """
    本函数用于计算节点碳流率
    k: form节点编号，从0开始，即0对应节点编号1
    j: to节点编号
    plk: 节点负荷，MW
    pk: 节点流过功率，MW
    """
    print(f'节点{k+1}的碳排放率计算完毕')
    return plk / pk * _r_l_

def loss_r_l(k: int, j: int, pkj_loss: float, pk: float, _r_l_=None):
    """
    本函数用于计算网损碳流率
    k: form节点编号，从0开始，即0对应节点编号1
    j: to节点编号
    plk_loss: 支路的损耗功率，MW
    pk: 节点流过功率，MW
    """
    print(f'支路{k}-{j}的损耗碳排放率计算完毕')    
    return pkj_loss / pk * _r_l_


# def r_l_2g4b(eg: np.array, au: np.array, pg: np.array,
#               pk_lis: list, plk_lis: list, pkj_loss_dict: dict):
#     """
#     批量计算文中2机4节点的节点、支路、网损碳流率，后续看一下如何改成通用的批量计算函数
#     pk, plk分别为所有节点流过功率、节点负荷的数组
#     pkj_loss_dict: 支路流过、损耗功率组成的字典，字典中value第一位为流过功率，第二位为损耗功率
#     """
#     bus_num = 4
#     bus_res = np.zeros(bus_num)
#     branch_res = {}
#     loss_res = {}
#     for k in range(bus_num):
#         _r_l_ = _r_l(k, eg, au, pg)[0, 0]
#         bus_res[k] = bus_r_l(k, plk = plk_lis[k], pk = pk_lis[k], _r_l_= _r_l_)
#         for key, value in pkj_loss_dict.items():
#             if key[0] == str(k+1):
#                 branch_res[key] = float(branch_r_l(k, eval(key[-1]), pkj=value[0], pk=pk_lis[k], _r_l_= _r_l_))
#                 loss_res[key] = float(loss_r_l(k, eval(key[-1]), pkj_loss=value[1], pk=pk_lis[k], _r_l_= _r_l_))
#     return bus_res, branch_res, loss_res

def carbon_flow_rate(all_paras):
    carbon_flow_rate_res = all_paras.copy()
    bus_flow_res = all_paras['pg_arr'].copy()
    branch_flow_res = all_paras['pkj_and_loss'].copy()
    for i in range(bus_flow_res.shape[0]):
        _r_l_ = _r_l(i, all_paras['eg_arr'][:, 1], all_paras['au'], all_paras['pg_arr'][:, 1])
        pk =  all_paras['pk_arr'][i, 1]
        plk = all_paras['plk_arr'][i, 1]
        bus_flow_res[i, 1] = bus_r_l(i, pk=pk, plk=plk, _r_l_=_r_l_)

        bus_from = i + 1
        mask = branch_flow_res[:, 0] == bus_from
        idx = np.where(mask)[0]
        for j in range(branch_flow_res[idx].shape[0]):
            bus_to = int(branch_flow_res[idx[j], 1])
            pkj = all_paras['pkj_and_loss'][idx[j], -2]
            pkj_loss = all_paras['pkj_and_loss'][idx[j], -1]
            branch_flow_res[idx[j], -2] = branch_r_l(
                bus_from, bus_to, 
                pkj, pk, _r_l_
                )
            branch_flow_res[idx[j], -1] = loss_r_l(
                bus_from, bus_to, 
                pkj_loss, pk, _r_l_
                )
    carbon_flow_rate_res['bus_carbon_flow_res'] = bus_flow_res
    carbon_flow_rate_res['branch_carbon_flow_res'] = branch_flow_res
    return carbon_flow_rate_res

def branch_carbon_flow_density(carbon_flow_rate_res):
    branch_flow_res = carbon_flow_rate_res['branch_carbon_flow_res'].copy()
    branch_flow_arr = branch_flow_res[:, 2]
    pkj_arr = carbon_flow_rate_res['pkj_and_loss'][:, 2]
    density = branch_flow_arr / pkj_arr
    branch_flow_res = np.column_stack([branch_flow_res, density])
    carbon_flow_rate_res['branch_carbon_flow_res'] = branch_flow_res
    return carbon_flow_rate_res

def bus_carbon_potential(carbon_flow_rate_res):
    branch_flow_res = carbon_flow_rate_res['branch_carbon_flow_res'].copy()
    branch_enj = np.zeros((branch_flow_res.shape[0], 3))
    bus_enj = carbon_flow_rate_res['eg_arr'].copy()
    branch_enj[:, [0, 1]] = branch_flow_res[:, [0, 1]]
    branch_enj[:, 2] = branch_flow_res[:, 2] - branch_flow_res[:, 3]
    pk_arr = carbon_flow_rate_res['pk_arr'].copy()
    for i in pk_arr[:, 0]:
        mask_branch = branch_enj[:, 1] == i
        if mask_branch.any():
            mask_bus = pk_arr[:, 0] == i
            enj = np.sum(branch_enj[mask_branch, -1]) / pk_arr[mask_bus, 1]
            bus_enj[mask_bus, -1] += enj 
    carbon_flow_rate_res['bus_carbon_flow_res'] = np.column_stack([carbon_flow_rate_res['bus_carbon_flow_res'], bus_enj[:, -1]])
    return carbon_flow_rate_res

def get_all_paras(power_flow_paras, unit_carbon_paras):
    eg = get_eg(power_flow_paras['pg_arr'], unit_carbon_paras)
    all_paras = power_flow_paras.copy()
    all_paras['eg_arr'] = eg
    return all_paras

def carbon_flow_caculation(mpc, unit_carbon_paras, print_=0):
    power_flow_paras = get_power_flow_paras(mpc)
    all_paras = power_flow_paras.copy()
    eg = get_eg(power_flow_paras['pg_arr'], unit_carbon_paras)
    all_paras['eg_arr'] = eg
    carbon_flow_res = carbon_flow_rate(all_paras)
    carbon_flow_res = branch_carbon_flow_density(carbon_flow_res)
    carbon_flow_res = bus_carbon_potential(carbon_flow_res)
    if print_:
        print('-' * 16)
        print('*本算例碳流相关计算结果如下*')
        print('1.节点相关的碳流计算结果如下，从左到右每列分别为节点标号、节点碳流率、节点碳势')
        print(np.array2string(carbon_flow_res['bus_carbon_flow_res'], suppress_small=True, precision=4))
        print('2.支路相关的碳流计算结果如下，从左到右每列分别为from节点标号、to节点标号、支路碳流率、支路损耗碳流率、支路碳流密度')
        print(np.array2string(carbon_flow_res['branch_carbon_flow_res'], suppress_small=True, precision=4))
    return carbon_flow_res

        
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
    unit_carbon_paras = np.array([
        [0.0004, 0.12, 2.5, 1.01, 0.80, 0.98, 0.8],
        [0.0010, 0.14, 4.0, 1.00, 0.80, 0.98, 0]
    ])
    carbon_flow_res = carbon_flow_caculation(mpc, unit_carbon_paras)
    breakpoint