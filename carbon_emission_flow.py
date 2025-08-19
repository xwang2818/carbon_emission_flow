"""
汪超群, 陈懿, 迟长云, 等. 基于潮流分布矩阵的电力系统碳排放流计算方法[J]. 科学技术与工程(12): 4835-4842.
碳流计算部分
"""


from power_flow_caculation import get_power_flow_paras
import numpy as np


def get_eg(pg_arr: np.array, unit_carbon_paras: np.array) -> np.array:
    """
    用于计算文章中的发电机碳排放强度向量(除了发动机节点之外其他置为0)
    :param pg_arr: 发电机注入功率组成的向量，除了发动机节点之外其他置为0
    :param unit_carbon_paras: 火电机组碳排放相关的参数，矩阵从左到右每列分别为a, b, c, zeta, xi, mu
    :return: eg_arr: 发电机碳排放强度向量(除了发动机节点之外其他置为0)
    """
    mask = pg_arr[:, 1] != 0  # 返回一个bool向量，用于从pg筛选出发电机节点
    _pg = pg_arr[mask]  # 取出发电机节点
    pg_value = _pg[:, 1]  # 取出发电机节点的出力值
    pg_num = pg_value.shape[0]
    pg_2 = np.diag(pg_value)  # 对角化发电机节点的出力值，本质上是pgi^2 / pgi
    pg_1 = np.eye(pg_num)  # 创建一个单位矩阵，本质上是pgi / pgi
    pg_0 = np.diag(1 / pg_value)  # 对角化发电机节点出力的倒数，本质上是1 / pgi
    a = unit_carbon_paras[:, 0].reshape(pg_num, -1)  # 从火电机组碳排放相关的参数取出a
    b = unit_carbon_paras[:, 1].reshape(pg_num, -1)  # 从火电机组碳排放相关的参数取出b
    c = unit_carbon_paras[:, 2].reshape(pg_num, -1)  # 从火电机组碳排放相关的参数取出c
    zeta = unit_carbon_paras[:, 3]  # 从火电机组碳排放相关的参数取出zeta
    eta = unit_carbon_paras[:, 4]  # 从火电机组碳排放相关的参数取出eta
    xi = unit_carbon_paras[:, 5]  # 从火电机组碳排放相关的参数取出xi
    mu = unit_carbon_paras[:, 6]  # 从火电机组碳排放相关的参数取出mu
    _w = pg_2 @ a + pg_1 @ b + pg_0 @ c  # 计算代表火电机组经济性的二次函数，用矩阵化的表达可实现批量计算而不使用循环
    w = _w.reshape(-1) * zeta  # 将前一步的结果变为一个一维的向量，乘上zeta得到w
    eg = eta * xi * (1 - mu) * (44 / 12) * w  # 按照公式计算eg
    eg_arr = pg_arr.copy()  # 复制pg_arr的样式用于储存这一函数计算的eg
    eg_arr[mask, -1] = eg  # 储存eg的计算结果
    return eg_arr


def _r_l(k: int, eg: np.array, au: np.array, pg: np.array) -> float:
    """
    用于计算文章中的 r_l（单位碳流率）标量结果
    :param k: 节点编号（从 0 开始，对应论文中的第 k+1 个节点）
    :param eg: 发电机碳排放强度向量（非发电机节点处为 0）
    :param au: 潮流分布矩阵 A_u
    :param pg: 发电机有功功率向量（与 eg 对齐）
    :return: r_l 的数值
    """
    n = pg.shape[0]                       # 取发电机向量长度，建立同维度选择向量
    au_inverse = np.linalg.inv(au)        # A_u 求逆，便于后续矩阵乘法
    pg_diag = np.diag(pg)                  # 将发电机有功功率向量对角化
    ek_t = np.zeros((1, n))               # 构造 1×n 的选择行向量 e_k^T
    ek_t[0, k] = 1                        # 在第 k 位设为 1，实现对第 k 个节点的选取
    return float(ek_t @ au_inverse @ pg_diag @ eg)


def branch_r_l(k: int, j: int, pkj: float, pk: float, _r_l_: float) -> float:
    """
    本函数用于计算支路碳流率
    :param k: from 节点编号（从 0 开始）
    :param j: to 节点编号（从 0 开始）
    :param pkj: 支路传输功率（MW）
    :param pk: 节点流过功率（MW）
    :param _r_l_: 该节点对应的 r_l（单位碳流率）
    :return: 支路碳流率
    """
    print(f'支路{k}-{j}的碳排放率计算完毕')   # 输出当前支路的计算完成信息
    return float(pkj / pk * _r_l_)        # 按公式 r_l^branch = (P_kj / P_k) * r_l 计算并返回

def bus_r_l(k: int, plk: float, pk: float, _r_l_: float) -> float:
    """
    本函数用于计算节点碳流率
    :param k: 节点编号（从 0 开始）
    :param plk: 节点负荷（MW）
    :param pk: 节点流过功率（MW）
    :param _r_l_: 该节点对应的 r_l（单位碳流率）
    :return: 节点碳流率
    """
    print(f'节点{k+1}的碳排放率计算完毕')     # 输出当前节点的计算完成信息（展示为 1 基）
    return plk / pk * _r_l_               # 按公式 r_l^bus = (P_Lk / P_k) * r_l 计算并返回

def loss_r_l(k: int, j: int, pkj_loss: float, pk: float, _r_l_: float = None) -> float:
    """
    本函数用于计算网损碳流率
    :param k: from 节点编号（从 0 开始）
    :param j: to 节点编号（从 0 开始）
    :param pkj_loss: 支路的损耗功率（MW）
    :param pk: 节点流过功率（MW）
    :param _r_l_: 该节点对应的 r_l（单位碳流率），可选参数
    :return: 网损碳流率
    """
    print(f'支路{k}-{j}的损耗碳排放率计算完毕')     # 输出当前支路损耗碳流率的计算完成信息
    return pkj_loss / pk * _r_l_                   # 按公式 r_l^loss = (P_kj_loss / P_k) * r_l 计算并返回


def carbon_flow_rate(all_paras: dict) -> dict:
    """
    本函数用于批量计算节点与支路的碳流率结果，并将结果回填到对应数组中
    :param all_paras: 汇总了碳流与潮流计算所需的全部参数字典，含键：
        - 'eg_arr'：发电机碳排放强度数组（第二列为 eg）
        - 'au'：潮流分布矩阵 A_u
        - 'pg_arr'：发电机注入功率数组（第二列为 Pg）
        - 'pk_arr'：各节点流过功率数组（第二列为 Pk）
        - 'plk_arr'：各节点负荷数组（第二列为 P_Lk）
        - 'pkj_and_loss'：支路功率与网损数组（倒数第二列为 P_kj，最后一列为 P_kj_loss）
    :return: 返回包含新增两项结果的字典：
        - 'bus_carbon_flow_res'：节点碳流率结果数组
        - 'branch_carbon_flow_res'：支路碳流率与网损碳流率结果数组
    """
    carbon_flow_rate_res = all_paras.copy()             # 复制整体参数字典，存放结果
    bus_flow_res = all_paras['pg_arr'].copy()           # 基于 pg_arr 的形状来承载“节点碳流率”结果
    branch_flow_res = all_paras['pkj_and_loss'].copy()  # 基于支路功率与损耗数组来承载“支路/网损碳流率”结果

    for i in range(bus_flow_res.shape[0]):              # 遍历每个节点（以 pg_arr 的行数为准）
        _r_l_ = _r_l(i, all_paras['eg_arr'][:, 1], all_paras['au'], all_paras['pg_arr'][:, 1])  # 计算第 i 个节点的 r_l
        pk =  all_paras['pk_arr'][i, 1]                 # 取该节点的流过功率 P_k
        plk = all_paras['plk_arr'][i, 1]                # 取该节点的负荷 P_Lk
        bus_flow_res[i, 1] = bus_r_l(i, pk=pk, plk=plk, _r_l_=_r_l_)  # 节点碳流率：回填到结果数组第二列
        

        bus_from = i + 1                                # 将 0 基节点索引转换为数据中的 1 基节点编号
        mask = branch_flow_res[:, 0] == bus_from        # 找到以该节点为 from 端的支路
        idx = np.where(mask)[0]                         # 提取这些支路的行索引

        for j in range(branch_flow_res[idx].shape[0]):  # 遍历与该节点相连的“从该节点发出”的每一条支路
            bus_to = int(branch_flow_res[idx[j], 1])    # 读取支路的 to 节点编号
            pkj = all_paras['pkj_and_loss'][idx[j], -2] # 该支路的有功功率 P_kj（倒数第二列）
            pkj_loss = all_paras['pkj_and_loss'][idx[j], -1]  # 该支路的网损功率 P_kj_loss（最后一列）

            branch_flow_res[idx[j], -2] = branch_r_l(   # 计算并写回“支路碳流率”（写入倒数第二列）
                bus_from, bus_to, 
                pkj, pk, _r_l_
            )
            branch_flow_res[idx[j], -1] = loss_r_l(     # 计算并写回“网损碳流率”（写入最后一列）
                bus_from, bus_to, 
                pkj_loss, pk, _r_l_
            )
    # ===== 追加“机组注入碳流率”列：gen_carbon = Pg * eg =====
    generator_r_l = all_paras['pg_arr'].copy()                          # 复制 Pg 数组（保留 [bus, Pg] 结构）
    generator_r_l[:, 1] = generator_r_l[:, 1] * all_paras['eg_arr'][:, 1]  # 逐节点计算 Pg * eg（发电侧碳排量）
    bus_flow_res = np.column_stack([bus_flow_res, generator_r_l[:, 1]]) # 将机组注入碳流率作为第3列追加到节点结果

    carbon_flow_rate_res['bus_carbon_flow_res'] = bus_flow_res         # 将节点碳流率结果加入返回字典
    carbon_flow_rate_res['branch_carbon_flow_res'] = branch_flow_res   # 将支路/网损碳流率结果加入返回字典
    return carbon_flow_rate_res                                        # 返回汇总结果字典

def branch_carbon_flow_density(carbon_flow_rate_res: dict) -> dict:
    """
    本函数用于计算支路的“碳流密度”（即支路碳流率与支路有功功率之比），并将其作为新列追加到支路结果数组中
    :param carbon_flow_rate_res: 含有 'branch_carbon_flow_res' 与 'pkj_and_loss' 等键的结果字典
    :return: 在 'branch_carbon_flow_res' 中追加一列“碳流密度”后的结果字典
    """
    branch_flow_res = carbon_flow_rate_res['branch_carbon_flow_res'].copy()  # 复制支路碳流率结果数组
    branch_flow_arr = branch_flow_res[:, 2]                                  # 取出用于计算密度的支路碳流率列
    pkj_arr = carbon_flow_rate_res['pkj_and_loss'][:, 2]                     # 取出对应支路的有功功率列 P_kj
    density = branch_flow_arr / pkj_arr                                      # 计算碳流密度 = 碳流率 / P_kj
    branch_flow_res = np.column_stack([branch_flow_res, density])            # 将密度作为新列拼接到结果数组
    carbon_flow_rate_res['branch_carbon_flow_res'] = branch_flow_res         # 写回结果字典
    return carbon_flow_rate_res                                              # 返回更新后的结果字典


def bus_carbon_potential(carbon_flow_rate_res: dict) -> dict:
    """
    本函数用于计算节点“碳势”，以支路碳流结果汇聚到各节点并归一到节点流过功率，最后将其追加到节点结果数组中
    :param carbon_flow_rate_res: 含有 'branch_carbon_flow_res'、'eg_arr'、'pk_arr' 等键的结果字典
    :return: 在 'bus_carbon_flow_res' 中追加一列“碳势”后的结果字典
    """
    branch_flow_res = carbon_flow_rate_res['branch_carbon_flow_res'].copy()  # 复制支路碳流结果（含支路与网损碳流率列）
    branch_enj = np.zeros((branch_flow_res.shape[0], 3))                     # 预分配支路“净碳流率”承载数组：[from, to, net]
    bus_enj = carbon_flow_rate_res['eg_arr'].copy()                          # 以 eg_arr 的形状承载节点碳势结果（在最后一列累加）
    branch_enj[:, [0, 1]] = branch_flow_res[:, [0, 1]]                       # 复制支路的 from/to 节点编号
    branch_enj[:, 2] = branch_flow_res[:, 2] - branch_flow_res[:, 3]         # 计算支路“净碳流率” = 支路碳流率 - 网损碳流率
    pk_arr = carbon_flow_rate_res['pk_arr'].copy()                            # 复制各节点流过功率数组 [node_id, P_k]


    for i in pk_arr[:, 0]:                                                   # 遍历每个节点编号（与数据中的编号一致）
        mask_branch = branch_enj[:, 1] == i                                  # 选取以该节点为 to 端的支路（汇入该节点）
        if mask_branch.any():                                                # 若存在汇入该节点的支路
            mask_bus = pk_arr[:, 0] == i                                     # 找到该节点在 pk_arr 中的行
            enj = np.sum(branch_enj[mask_branch, -1]) / pk_arr[mask_bus, 1]  # 节点碳势 = 汇入净碳流率之和 / 节点流过功率
            bus_enj[mask_bus, -1] += enj                                     # 将该节点碳势累加到 bus_enj 的最后一列
    
    mask_generator = carbon_flow_rate_res['plk_arr'][:, 1] == 0              # 找到无负荷（P_Lk==0）的节点（通常为纯发电节点）
    bus_enj[mask_generator, 1] = 0                                           # 将此类节点的碳势强制置 0（避免误解读）


    carbon_flow_rate_res['bus_carbon_flow_res'] = np.column_stack(           # 将节点碳势追加到节点结果数组
        [carbon_flow_rate_res['bus_carbon_flow_res'],  bus_enj[:, -1], carbon_flow_rate_res['eg_arr'][:, 1]]
    )
    return carbon_flow_rate_res                                              # 返回更新后的结果字典


def get_all_paras(power_flow_paras: dict, unit_carbon_paras: np.array) -> dict:
    """
    本函数用于组装碳流计算所需的完整参数字典：在潮流结果参数的基础上补充 eg 向量
    :param power_flow_paras: 由潮流计算整理得到的参数字典（含 pg_arr、plk_arr、pkj_and_loss、pk_arr、au 等）
    :param unit_carbon_paras: 火电机组碳排放相关参数矩阵（列依次为 a、b、c、zeta、eta、xi、mu）
    :return: 含新增键 'eg_arr' 的完整参数字典
    """
    eg = get_eg(power_flow_paras['pg_arr'], unit_carbon_paras)  # 基于发电机出力与碳排参数计算 eg 向量
    all_paras = power_flow_paras.copy()                          # 复制潮流参数字典，避免原数据被覆盖
    all_paras['eg_arr'] = eg                                     # 增加 eg 向量到参数集合
    return all_paras                                             # 返回完整参数字典


def carbon_flow_caculation(mpc: dict, unit_carbon_paras: np.array, print_: int = 0) -> dict:
    power_flow_paras = get_power_flow_paras(mpc)                 # 执行潮流计算并整理为参数字典
    all_paras = power_flow_paras.copy()                          # 复制一份作为后续碳流计算的输入
    eg = get_eg(power_flow_paras['pg_arr'], unit_carbon_paras)   # 计算发电机碳排放强度向量 eg
    all_paras['eg_arr'] = eg                                     # 将 eg 写入参数集合

    carbon_flow_res = carbon_flow_rate(all_paras)                # 计算节点与支路的碳流率
    carbon_flow_res = branch_carbon_flow_density(carbon_flow_res)  # 计算并追加支路碳流密度
    carbon_flow_res = bus_carbon_potential(carbon_flow_res)        # 计算并追加节点碳势

    if print_:                                                   # 若需要，打印结果摘要
        print_carbon_results(carbon_flow_res, precision=4)       # 调用独立打印函数（更明显分割线 + 等宽表格）

    return carbon_flow_res                                       # 返回结果

def system_balance_check(carbon_flow_res: dict) -> tuple[float, float, float, float]:
    """
    本函数用于进行系统“碳平衡”检验：分别统计网损碳流率之和、发电侧碳排量之和、负荷侧碳流率之和，
    并给出用于平衡校验的组合量（loss_sum + load_r_l_sum），以便与发电侧总量对比。
    :param carbon_flow_res: 结果字典，需包含：
        - 'branch_carbon_flow_res'：支路结果数组，其中第 4 列（索引 3）为 r_l_loss（网损碳流率）
        - 'bus_carbon_flow_res'：节点结果数组，其中第 3 列（索引 2）为 gen_carbon（Pg*eg），
                                  第 2 列（索引 1）为 r_l_bus（负荷侧碳流率）
    :return: 四元组 (loss_sum, g_r_l_sum, load_r_l_sum, balance_sum)
        - loss_sum：网损碳流率求和（按列 r_l_loss 汇总）
        - g_r_l_sum：发电侧碳排量求和（按列 gen_carbon 汇总）
        - load_r_l_sum：负荷侧碳流率求和（按列 r_l_bus 汇总）
        - balance_sum：loss_sum + load_r_l_sum（用于与 g_r_l_sum 对比做平衡校验）
    """
    loss_r_l_sum = np.sum(carbon_flow_res['branch_carbon_flow_res'][:, 3])  # 网损碳流率之和（支路结果第 4 列）
    g_r_l_sum = np.sum(carbon_flow_res['bus_carbon_flow_res'][:, 2])        # 发电侧碳排量之和（节点结果第 3 列）
    load_r_l_sum = np.sum(carbon_flow_res['bus_carbon_flow_res'][:, 1])     # 负荷侧碳流率之和（节点结果第 2 列）
    return loss_r_l_sum, g_r_l_sum, load_r_l_sum, loss_r_l_sum + load_r_l_sum   # 注意：此处的 loss_sum 名称应与上方 loss_r_l_sum 对应




def print_carbon_results(carbon_flow_res: dict, precision: int = 4) -> None:
    """
    用于美观地打印碳流相关计算结果，并统一所有分割线与表格宽度；同时追加“系统碳平衡检验”。
    术语说明：本函数中将“发电侧碳排量”统一称为“发电机注入碳流率”。
    :param carbon_flow_res: 碳流计算结果字典，需包含
        - 'bus_carbon_flow_res'：节点结果数组（列：bus, r_l, gen_inj_r_l, potential, gen_eg）
        - 'branch_carbon_flow_res'：支路结果数组（列：from, to, r_l, loss_r_l, density）
    :param precision: 浮点数打印的小数位数
    :return: 无
    """
    def _format_table(arr, headers=None, precision=4, target_width=None):
        """
        将二维数组格式化为等宽对齐表格；若提供 target_width，则自动加宽最后一列以匹配目标宽度
        :param arr: 待格式化的二维数组
        :param headers: 可选的表头列表，与列数一致
        :param precision: 浮点数打印的小数位数
        :param target_width: 期望整张表（含边框）的总宽度；None 表示自适应
        :return: (格式化字符串, 表格总宽度)
        """
        arr = np.asarray(arr)
        if arr.ndim != 2:
            s = np.array2string(arr, suppress_small=True, precision=precision)
            return s, max(len(line) for line in s.splitlines())

        rows, cols = arr.shape
        if headers is None:
            headers = [f'col{c+1}' for c in range(cols)]

        # 判断每列是否“整数列”
        int_col = []
        for c in range(cols):
            col = arr[:, c]
            ok = np.all(np.isfinite(col)) and np.all(np.isclose(col, np.round(col)))
            int_col.append(ok)

        # 数值转字符串
        str_mat = []
        for r in range(rows):
            row_str = []
            for c in range(cols):
                v = arr[r, c]
                s = f"{int(round(v))}" if int_col[c] else f"{float(v):.{precision}f}"
                row_str.append(s)
            str_mat.append(row_str)

        # 列宽（含表头）
        widths = []
        for c in range(cols):
            col_cells = [str_mat[r][c] for r in range(rows)]
            w = max(len(x) for x in col_cells)
            w = max(w, len(headers[c]))
            widths.append(w)

        # 构造一次表格
        def _build(widths_local):
            def _hbar(left, mid, right):
                return left + mid.join('─' * (w + 2) for w in widths_local) + right  # 每列左右各加 1 空格
            top = _hbar('┌', '┬', '┐')
            sep = _hbar('├', '┼', '┤')
            bot = _hbar('└', '┴', '┘')
            lines = [top]
            head = '│' + '│'.join(f" {headers[c]:<{widths_local[c]}} " for c in range(cols)) + '│'
            lines += [head, sep]
            for r in range(rows):
                line = '│' + '│'.join(f" {str_mat[r][c]:>{widths_local[c]}} " for c in range(cols)) + '│'
                lines.append(line)
            lines.append(bot)
            table_str = "\n".join(lines)
            width_len = len(top)
            return table_str, width_len

        table_str, width_len = _build(widths)

        # 若需要统一宽度则加宽最后一列
        if target_width is not None and target_width > width_len:
            delta = target_width - width_len
            widths[-1] += delta
            table_str, width_len = _build(widths)

        return table_str, width_len

    # 1) 先渲染节点表与支路表，获取宽度
    bus_headers = ['bus', 'r_l', 'gen_inj_r_l', 'potential', 'gen_eg'] \
        if carbon_flow_res['bus_carbon_flow_res'].shape[1] >= 4 else ['bus', 'r_l', 'gen_inj_r_l']
    bus_tbl_str, bus_w = _format_table(
        carbon_flow_res['bus_carbon_flow_res'],
        headers=bus_headers,
        precision=precision,
        target_width=None
    )
    br_tbl_str, br_w = _format_table(
        carbon_flow_res['branch_carbon_flow_res'],
        headers=['from', 'to', 'r_l', 'loss_r_l', 'density'],
        precision=precision,
        target_width=None
    )

    # 2) 系统碳平衡检验
    loss_sum, gen_inj_sum, load_sum, balance_sum = system_balance_check(carbon_flow_res)  # 调用你前面写的函数

    # 碳平衡表：单行 4 列（仅数值），表头为术语化列名
    balance_arr = np.array([[gen_inj_sum, loss_sum,  load_sum, balance_sum]], dtype=float)
    balance_headers = ['gen_inj_r_l_sum', 'loss_sum',  'load_r_l_sum', 'load_loss_sum']
    balance_tbl_str, bal_w = _format_table(
        balance_arr,
        headers=balance_headers,
        precision=precision,
        target_width=None
    )

    # 3) 统一宽度 W，并按 W 重渲染三张表
    W = max(bus_w, br_w, bal_w)
    bus_tbl_str, _ = _format_table(
        carbon_flow_res['bus_carbon_flow_res'], headers=bus_headers, precision=precision, target_width=W
    )
    br_tbl_str, _ = _format_table(
        carbon_flow_res['branch_carbon_flow_res'],
        headers=['from', 'to', 'r_l', 'loss_r_l', 'density'],
        precision=precision, target_width=W
    )
    balance_tbl_str, _ = _format_table(
        balance_arr, headers=balance_headers, precision=precision, target_width=W
    )

    divider_main = '═' * W
    divider_section = '─' * W

    # 4) 打印：标题 + 节点表 + 支路表 + 平衡检验表
    print('\n' + divider_main)
    print('★ 本算例碳流相关计算结果')
    print(divider_main + '\n')

    print('1) 节点相关结果（列：' + ', '.join(bus_headers) + '）')
    print(bus_tbl_str)
    print('\n' + divider_section + '\n')

    print('2) 支路相关结果（列：from, to, r_l, loss_r_l, density）')
    print(br_tbl_str)
    print('\n' + divider_section + '\n')

    print('3) 系统碳平衡检验（列：loss_sum, gen_inj_r_l_sum, load_r_l_sum, balance_sum）')
    print(balance_tbl_str)

    # 5) 平衡状态提示（PASS / FAIL）
    ok = np.isclose(gen_inj_sum, balance_sum, rtol=1e-6, atol=1e-8)
    status = 'PASS' if ok else 'FAIL'
    delta = gen_inj_sum - balance_sum
    tip = f"Check: {status}  (gen_inj_r_l_sum ?≈ balance_sum)  Δ = {delta:.{precision}g}"
    print(tip.center(W))
    print('\n' + divider_main + '\n')
