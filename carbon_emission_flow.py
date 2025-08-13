"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
碳流计算部分
"""

# TODO 计算eg

import numpy as np
"""
先使用numpy编程，后期如果需要使用torch联动，修改比较快
"""

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
    return ek_t @ au_inverse @ pg_diag @ eg


def branch_r_l(k: int, j: int, pkj: float, pk: float, _r_l_):
    """
    本函数用于计算支路碳流率
    k: form节点编号，从0开始，即0对应节点编号1
    j: to节点编号，空置变量
    pkj: 支路传输功率，MW
    pk: 节点流过功率，MW
    """
    print(f'支路{k+1}-{j+1}的碳排放率计算完毕')
    return pkj / pk * _r_l_

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
    print(f'支路{k+1}-{j+1}的损耗碳排放率计算完毕')    
    return pkj_loss / pk * _r_l_

def eg_caculate

def r_l_2g4b(eg: np.array, au: np.array, pg: np.array,
              pk_lis: list, plk_lis: list, pkj_loss_dict: dict):
    """
    批量计算文中2机4节点的节点、支路、网损碳流率，后续看一下如何改成通用的批量计算函数
    pk, plk分别为所有节点流过功率、节点负荷的数组
    pkj_loss_dict: 支路流过、损耗功率组成的字典，字典中value第一位为流过功率，第二位为损耗功率
    """
    bus_num = 4
    bus_res = np.zeros(bus_num)
    branch_res = {}
    loss_res = {}
    for k in range(bus_num):
        _r_l_ = _r_l(k, eg, au, pg)[0, 0]
        bus_res[k] = bus_r_l(k, plk = plk_lis[k], pk = pk_lis[k], _r_l_= _r_l_)
        for key, value in pkj_loss_dict.items():
            if key[0] == str(k+1):
                branch_res[key] = float(branch_r_l(k, eval(key[-1]), pkj=value[0], pk=pk_lis[k], _r_l_= _r_l_))
                loss_res[key] = float(loss_r_l(k, eval(key[-1]), pkj_loss=value[1], pk=pk_lis[k], _r_l_= _r_l_))
    return bus_res, branch_res, loss_res


        
if __name__ == '__main__':
    eg = np.array([[0.1682], [0.8310], [0], [0]])
    # au = np.array([[1, 0, 0, 0],
    #                [-0.0489, 1, 0, 0],
    #                [-0.3642, -0.9702, 1, 0],
    #                [-0.5446, 0, -0.2773, 1]])
    au = np.load('./Au_sy.npz')['arr_0']

    pg = np.array([408.8254, 133.9775, 0, 0])
    pkj_loss_dict = {
        '1-2': (20.04914, 0.0717),
        '1-3': (153.9593, 5.0603),
        '1-4': (234.8170, 12.1646),
        '2-3': (133.9775, 3.9865),
        '3-4': (78.8899, 1.5423)
    }
    pk_lis = [408.8254, 133.9775, 278.8899, 300.00]
    plk_lis = [408.8254, 133.9775, 200, 300]
    res = r_l_2g4b(eg, au, pg, pk_lis, plk_lis, pkj_loss_dict)
    print(f"""2机4节点算例中
    节点的碳流率计算结果为{res[0]}
    支路的碳流率计算结果为{res[1]}
    支路损耗的碳硫率计算结果为{res[2]}""")