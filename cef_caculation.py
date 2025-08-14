"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
①结合pypower的场景定义字典即mpc字典提取后续计算所需的后续参数的过程
②调用潮流分布计算和碳流分布计算进行完整计算
"""

from carbon_emission_flow import carbon_flow_caculation
import numpy as np


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