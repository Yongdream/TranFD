import numpy as np
import pandas as pd
import os
import math
import random

win = 300  # 计算相关系数的时间窗长度


def calc_corr(a, b):  # 计算相关系数
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return corr_factor


def calc_corr_group(a, AVE, win):  # 以时间窗截取的方式进行循环计算
    b = np.zeros_like(a[win:])  # 构造目标形状的0矩阵
    for i in range(win, len(a[:])):  # 时间戳从win开始进行截取
        b[i - win] = calc_corr(a[i - win:i], AVE[i - win:i])  # 对该时间戳及其之前长度为win的数据进行截取
    return b


def Normalize(data, U_max, U_min, corr_min, diff_max, pict_mode=False):
    if corr_min < 0:
        corr_min = -1
    else:
        corr_min = 0
    if not pict_mode:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                data[0, i, j] = ((data[0, i, j] - U_min) / (U_max - U_min))
                data[1, i, j] = ((data[1, i, j] - corr_min) / (1 - corr_min))
                data[2, i, j] = ((data[2, i, j] - 0) / (diff_max - 0))
    else:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                data[0, i, j] = ((data[0, i, j] - U_min) / (U_max - U_min)) * 255
                data[1, i, j] = ((data[1, i, j] - corr_min) / (1 - corr_min)) * 255
                data[2, i, j] = ((data[2, i, j] - 0) / (diff_max - 0)) * 255
    return data


def Add_Noise(data, mu, sigma):
    for i in range(len(data)):
        data[i] += random.gauss(mu, sigma)
    return data


def signal_stick(data, sti_step=200, con_step=20, start=1):
    step = sti_step + con_step
    rounds = len(data) // step + 1
    for j in range(1, rounds):
        data[start:start + sti_step] = np.repeat(data[start - 1], sti_step)
        start += step
    return np.array(data)


def Get_Data_Len(path):
    file = os.path.join(path)
    data_all = pd.read_csv(file).dropna()  # 读取csv文件，丢弃NaN数据
    len_d = len(np.array(data_all['CH5001'])[:])
    return len_d


def Get_Data(path, T_begin, T_end):  # 获取指定路径的原始数据、相关系数及压差绝对值矩阵，并输出相关系数最小值及压差绝对值最大值
    file = os.path.join(path)
    data_all = pd.read_csv(file).dropna()  # 读取csv文件，丢弃NaN数据
    U_5001 = np.array(data_all['CH5001'])[T_begin:T_end]

    # 使用 signal_stick 处理粘滞
    U_5001 = signal_stick(U_5001)
    U_5002 = np.array(data_all['CH5002'])[T_begin:T_end]
    U_5003 = np.array(data_all['CH5003'])[T_begin:T_end]
    U_5004 = np.array(data_all['CH5004'])[T_begin:T_end]
    U_5005 = np.array(data_all['CH5005'])[T_begin:T_end]
    U_5006 = np.array(data_all['CH5006'])[T_begin:T_end]
    U_5007 = np.array(data_all['CH5007'])[T_begin:T_end]
    U_5008 = np.array(data_all['CH5008'])[T_begin:T_end]
    U_5009 = np.array(data_all['CH5009'])[T_begin:T_end]
    U_5010 = np.array(data_all['CH5010'])[T_begin:T_end]
    U_5011 = np.array(data_all['CH5011'])[T_begin:T_end]
    U_5012 = np.array(data_all['CH5012'])[T_begin:T_end]
    U_5013 = np.array(data_all['CH5013'])[T_begin:T_end]
    U_5014 = np.array(data_all['CH5014'])[T_begin:T_end]
    U_5015 = np.array(data_all['CH5015'])[T_begin:T_end]
    U_5016 = np.array(data_all['CH5016'])[T_begin:T_end]
    U_5017 = np.array(data_all['CH5017'])[T_begin:T_end]
    U_5018 = np.array(data_all['CH5018'])[T_begin:T_end]
    U_5019 = np.array(data_all['CH5019'])[T_begin:T_end]
    U_5020 = np.array(data_all['CH5020'])[T_begin:T_end]

    data = []
    for i in range(len(U_5001)):  # 原数据矩阵
        data.append([U_5001[i], U_5002[i], U_5003[i], U_5004[i], U_5005[i], U_5006[i], U_5007[i], U_5008[i], U_5009[i],
                     U_5010[i], U_5011[i], U_5012[i], U_5013[i], U_5014[i], U_5015[i], U_5016[i], U_5017[i], U_5018[i],
                     U_5019[i], U_5020[i]])
    data = np.array(data)
    U_max = np.max(data[win:])
    U_min = np.min(data[win:])

    Vis_value = float((U_max + U_min) / 2)
    U_5001 = signal_stick(U_5001)  # 再次使用 signal_stick 处理
    data = []
    for i in range(len(U_5001)):  # 原数据矩阵
        data.append([U_5001[i], U_5002[i], U_5003[i], U_5004[i], U_5005[i], U_5006[i], U_5007[i], U_5008[i], U_5009[i],
                     U_5010[i], U_5011[i], U_5012[i], U_5013[i], U_5014[i], U_5015[i], U_5016[i], U_5017[i], U_5018[i],
                     U_5019[i], U_5020[i]])
    data = np.array(data)

    AVE = data.mean(axis=1)
    U_5001_corr = calc_corr_group(U_5001, AVE, win)
    U_5002_corr = calc_corr_group(U_5002, AVE, win)
    U_5003_corr = calc_corr_group(U_5003, AVE, win)
    U_5004_corr = calc_corr_group(U_5004, AVE, win)
    U_5005_corr = calc_corr_group(U_5005, AVE, win)
    U_5006_corr = calc_corr_group(U_5006, AVE, win)
    U_5007_corr = calc_corr_group(U_5007, AVE, win)
    U_5008_corr = calc_corr_group(U_5008, AVE, win)
    U_5009_corr = calc_corr_group(U_5009, AVE, win)
    U_5010_corr = calc_corr_group(U_5010, AVE, win)
    U_5011_corr = calc_corr_group(U_5011, AVE, win)
    U_5012_corr = calc_corr_group(U_5012, AVE, win)
    U_5013_corr = calc_corr_group(U_5013, AVE, win)
    U_5014_corr = calc_corr_group(U_5014, AVE, win)
    U_5015_corr = calc_corr_group(U_5015, AVE, win)
    U_5016_corr = calc_corr_group(U_5016, AVE, win)
    U_5017_corr = calc_corr_group(U_5017, AVE, win)
    U_5018_corr = calc_corr_group(U_5018, AVE, win)
    U_5019_corr = calc_corr_group(U_5019, AVE, win)
    U_5020_corr = calc_corr_group(U_5020, AVE, win)

    data_corr = []
    for i in range(len(U_5001) - win):  # 相关系数矩阵（0号时间索引对应原数据win号时间索引）
        data_corr.append(
            [U_5001_corr[i], U_5002_corr[i], U_5003_corr[i], U_5004_corr[i], U_5005_corr[i], U_5006_corr[i],
             U_5007_corr[i],
             U_5008_corr[i], U_5009_corr[i], U_5010_corr[i], U_5011_corr[i], U_5012_corr[i], U_5013_corr[i],
             U_5014_corr[i],
             U_5015_corr[i], U_5016_corr[i], U_5017_corr[i], U_5018_corr[i], U_5019_corr[i], U_5020_corr[i]])

    for i in range(len(data_corr)):
        for j in range(len(data_corr[0])):
            if data_corr[i][j] < 0:
                data_corr[i][j] = 0
    data_corr = np.array(data_corr)
    corr_min = np.min(data_corr)

    data_diff = []
    for i in range(win, len(U_5001)):  # 压差绝对值矩阵（0号时间索引对应原数据win号时间索引）
        data_diff.append(
            [abs(U_5001[i] - AVE[i]), abs(U_5002[i] - AVE[i]), abs(U_5003[i] - AVE[i]), abs(U_5004[i] - AVE[i]),
             abs(U_5005[i] - AVE[i]), abs(U_5006[i] - AVE[i]), abs(U_5007[i] - AVE[i]), abs(U_5008[i] - AVE[i]),
             abs(U_5009[i] - AVE[i]), abs(U_5010[i] - AVE[i]), abs(U_5011[i] - AVE[i]), abs(U_5012[i] - AVE[i]),
             abs(U_5013[i] - AVE[i]), abs(U_5014[i] - AVE[i]), abs(U_5015[i] - AVE[i]), abs(U_5016[i] - AVE[i]),
             abs(U_5017[i] - AVE[i]), abs(U_5018[i] - AVE[i]), abs(U_5019[i] - AVE[i]), abs(U_5020[i] - AVE[i])])
    data_diff = np.array(data_diff)
    diff_max = np.max(data_diff)

    data = data[win:]  # 由于计算了相关系数截取了时间窗，因此数据长度截短
    data_feature = np.array([data, data_corr, data_diff])

    return data_feature, U_max, U_min, corr_min, diff_max


def Data_Origin_Saved_Npy(data_feature, path, Condition, Fault, d_f, T_begin, T_end):
    """
    保存数据到 NPY 文件.
    Args:
        data_feature: 待保存的数据矩阵
        path: 数据源文件路径
        Condition: 工况词典
        Fault: 故障类型
        d_f: 选择的故障类型
        T_begin: 数据开始时间
        T_end: 数据结束时间
    """
    # 确定工况名
    d_c = next((c_name for c_name in Condition if c_name in path), None)

    # 根据不同故障类型调整文件名格式
    file_name = f"{d_c}_{d_f}_s0_05_{path[-8:-4]}"
    if d_f == "Vis":
        file_name = f"{d_c}_{d_f}_{T_begin}-{T_end}_{path[-8:-4]}"

    # 创建保存目录
    dir_path = f"../Npy/{d_c}/{d_f}"
    os.makedirs(dir_path, exist_ok=True)

    # 保存文件
    final_save_path = f"{dir_path}/{file_name}"
    np.save(final_save_path, data_feature)

    # 输出保存信息
    print("-" * 50)
    print(f"Data Saved Successfully! Path: {final_save_path}.npy")
    print(f"Total Length: {data_feature.shape[1]}")
    print("-" * 50)


def process_data(path_n, mode="Vis"):
    """
    处理数据，根据模式选择处理类型.
    Args:
        path_n: 数据路径
        mode: 处理模式，"Vis" 表示粘滞，"Noi" 表示噪声
    """
    len_d = Get_Data_Len(path_n)
    Condition = ["FUDS", "UDDS", "US06"]  # 工况词典
    Fault = ["Cor", "Isc", "Noi", "Nor", "Vis"]  # 故障词典

    if mode == "Vis":  # 粘滞故障模式
        # 一次性处理整个数据
        data_feature, U_max, U_min, corr_min, diff_max = Get_Data(path_n, 0, len_d)
        print(f"Data Get OK! U_max: {U_max}, U_min: {U_min}, corr_min: {corr_min}, diff_max: {diff_max}")

        # 保存数据为单个.npy文件
        Data_Origin_Saved_Npy(data_feature, path_n, Condition, Fault, "Vis", 0, len_d)

    elif mode == "Noi":  # 噪声处理模式
        data_feature, U_max, U_min, corr_min, diff_max = Get_Data(path_n, 0, len_d)
        print(f"Data Get OK! U_max: {U_max}, U_min: {U_min}, corr_min: {corr_min}, diff_max: {diff_max}")
        Data_Origin_Saved_Npy(data_feature, path_n, Condition, Fault, "Noi", 0, len_d)
    else:
        print("Invalid mode selected.")


if __name__ == '__main__':
    base_path = "../OriginV"

    # 遍历OriginV文件夹下所有子文件夹
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv") and "Vis" in root:  # 噪声模拟修改为 "Noi"
                file_path = os.path.join(root, file)
                # 粘滞调用
                process_data(file_path, mode="Vis")

                # 噪声调用
                # process_data(file_path, mode="Noi")
