"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
本文件是项目的“最小可复现场景/示例脚本”。它在脚本内：
1) 构造一个 4 节点的 MATPOWER 风格算例 mpc；
2) 定义火电机组的单位碳排参数 unit_carbon_paras；
3) 调用主流程函数 carbon_flow_caculation(mpc, unit_carbon_paras, print_=1)，
   完成从潮流计算到碳流计算的一键式执行，并美观打印结果。

数据结构
--------
1) mpc（MATPOWER/PPower 风格系统数据，字典）
   - version: 字符串（如 '2'）
   - baseMVA: 数值（如 100）
   - bus: 二维数组，形状 (N, 13)
       列顺序：[bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
   - gen: 二维数组，形状 (G, 10)
       列顺序：[bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin]
   - branch: 二维数组，形状 (L, 13)
       列顺序：[fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]

2) unit_carbon_paras（火电机组单位碳排相关参数，二维数组）
   - 形状：(G, 7)，列从左到右依次为：
     [a, b, c, zeta, eta, xi, mu]
   - 说明：与机组出力 pg_arr 共同用于计算发电机碳排放强度向量 eg_arr。

3) carbon_flow_res（碳流一键计算的综合结果，字典）
   —— 由 carbon_flow_caculation 返回，包含中间量与最终指标：
   - pg_arr: 二维数组 (N, 2) → [bus, Pg]（节点编号、发电注入功率）
   - plk_arr: 二维数组 (N, 2) → [bus, Pd]（节点编号、有功负荷）
   - pkj_and_loss: 二维数组 (L, 4) → [from, to, P_kj, P_kj_loss]（支路有功与网损功率）
   - pk_arr: 二维数组 (N, 2) → [bus, P_k]（节点流过功率）
   - au: 二维数组 (N, N) → 潮流分布矩阵 A_u
   - eg_arr: 二维数组 (N, 2) → [bus, eg]（节点编号、单位碳排强度）
   - original_power_anlysis_res: 字典 → pypower.runpf 的原始潮流结果
   - bus_carbon_flow_res: 二维数组 (N, 3) → [bus, r_l, potential]
       * r_l：节点碳流率；potential：节点碳势
   - branch_carbon_flow_res: 二维数组 (L, 5) → [from, to, r_l_branch, r_l_loss, density]
       * r_l_branch：支路碳流率
       * r_l_loss：网损碳流率
       * density：碳流密度 = r_l_branch / P_kj（分母来自 pkj_and_loss 中的 P_kj）

备注
----
- 以上 N/G/L 分别表示节点/机组/支路数量，示例脚本中 N=4、G=2、L=5。
- 本文件仅负责组装输入并调用主流程；计算细节在 power_flow_caculation.py 与
  carbon_emission_flow.py 中实现。
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
carbon_flow_res = carbon_flow_caculation(mpc, unit_carbon_paras, 1)
breakpoint