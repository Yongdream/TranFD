import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as io
import pandas as pd
import os
import math
import torch

win = 300  # 计算相关系数的时间窗长度

def calc_corr(a, b):  # 计算相关系数
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return corr_factor

def calc_corr_group(a, AVE, win):  # 以时间窗截取的方式进行循环计算
    b = np.zeros_like(a[win:])  # 构造目标形状的0矩阵
    for i in range(win, len(a[:])):  # 时间戳从win开始进行截取
        b[i - win] = calc_corr(a[i - win:i], AVE[i - win:i])  # 对该时间戳及其之前长度为win的数据进行截取
    return b

def Normalize(data, U_max, U_min, corr_min, diff_max, pict_mode = False):
    data_copy = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    # if corr_min < 0:
    #     corr_min = -1
    # else:
    #     corr_min = 0
    corr_min = 0
    if pict_mode is False:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                data_copy[0, i, j] = ((data[0, i, j] - U_min) / (U_max - U_min))
                data_copy[1, i, j] = ((data[1, i, j] - corr_min) / (1 - corr_min))
                data_copy[2, i, j] = ((data[2, i, j] - 0) / (diff_max - 0))
    if pict_mode is True:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                data_copy[0, i, j] = ((data[0, i, j] - U_min) / (U_max - U_min)) * 255
                data_copy[1, i, j] = ((data[1, i, j] - corr_min) / (1 - corr_min)) * 255
                data_copy[2, i, j] = ((data[2, i, j] - 0) / (diff_max - 0)) * 255
    return data_copy

def Get_Data(path):  # 获取指定路径的原始数据、相关系数及压差绝对值矩阵，并输出相关系数最小值及压差绝对值最大值
    file = os.path.join(path)
    data_all = pd.read_csv(file).dropna()  # 读取csv文件，丢弃NaN数据
    U_5001 = np.array(data_all['CH5001'])[:]
    U_5002 = np.array(data_all['CH5002'])[:]
    U_5003 = np.array(data_all['CH5003'])[:]
    U_5004 = np.array(data_all['CH5004'])[:]
    U_5005 = np.array(data_all['CH5005'])[:]
    U_5006 = np.array(data_all['CH5006'])[:]
    U_5007 = np.array(data_all['CH5007'])[:]
    U_5008 = np.array(data_all['CH5008'])[:]
    U_5009 = np.array(data_all['CH5009'])[:]
    U_5010 = np.array(data_all['CH5010'])[:]
    U_5011 = np.array(data_all['CH5011'])[:]
    U_5012 = np.array(data_all['CH5012'])[:]
    U_5013 = np.array(data_all['CH5013'])[:]
    U_5014 = np.array(data_all['CH5014'])[:]
    U_5015 = np.array(data_all['CH5015'])[:]
    U_5016 = np.array(data_all['CH5016'])[:]
    U_5017 = np.array(data_all['CH5017'])[:]
    U_5018 = np.array(data_all['CH5018'])[:]
    U_5019 = np.array(data_all['CH5019'])[:]
    U_5020 = np.array(data_all['CH5020'])[:]
    data = []
    for i in range(0, len(U_5001)):  # 原数据矩阵
        data.append([U_5001[i], U_5002[i], U_5003[i], U_5004[i], U_5005[i], U_5006[i], U_5007[i], U_5008[i], U_5009[i],
                      U_5010[i], U_5011[i], U_5012[i], U_5013[i], U_5014[i], U_5015[i], U_5016[i], U_5017[i], U_5018[i],
                      U_5019[i], U_5020[i]])
    data = np.array(data)
    U_max = np.max(data[win:])
    U_min = np.min(data[win:])
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
    for i in range(0, len(U_5001) - win):  # 相关系数矩阵（0号时间索引对应原数据win号时间索引）
        data_corr.append([U_5001_corr[i], U_5002_corr[i], U_5003_corr[i], U_5004_corr[i], U_5005_corr[i], U_5006_corr[i], U_5007_corr[i],
                      U_5008_corr[i], U_5009_corr[i], U_5010_corr[i], U_5011_corr[i], U_5012_corr[i], U_5013_corr[i], U_5014_corr[i],
                      U_5015_corr[i], U_5016_corr[i], U_5017_corr[i], U_5018_corr[i], U_5019_corr[i], U_5020_corr[i]])
    data_corr = np.array(data_corr)
    corr_min = np.min(data_corr)
    data_diff = []
    for i in range(win, len(U_5001)):  # 压差绝对值矩阵（0号时间索引对应原数据win号时间索引）
        data_diff.append([abs(U_5001[i] - AVE[i]), abs(U_5002[i] - AVE[i]), abs(U_5003[i] - AVE[i]), abs(U_5004[i] - AVE[i]),
                          abs(U_5005[i] - AVE[i]), abs(U_5006[i] - AVE[i]), abs(U_5007[i] - AVE[i]), abs(U_5008[i] - AVE[i]),
                          abs(U_5009[i] - AVE[i]), abs(U_5010[i] - AVE[i]), abs(U_5011[i] - AVE[i]), abs(U_5012[i] - AVE[i]),
                          abs(U_5013[i] - AVE[i]), abs(U_5014[i] - AVE[i]), abs(U_5015[i] - AVE[i]), abs(U_5016[i] - AVE[i]),
                          abs(U_5017[i] - AVE[i]), abs(U_5018[i] - AVE[i]), abs(U_5019[i] - AVE[i]), abs(U_5020[i] - AVE[i])])
    data_diff = np.array(data_diff)
    diff_max = np.max(data_diff)
    data = data[win:]  # 由于计算了相关系数截取了时间窗，因此数据长度截短
    data_feature = np.array([data, data_corr, data_diff])

    All_list = []
    for i in range(0, 1000):
        All_list.append([U_5002[i], U_5001[i]])

    df1 = pd.DataFrame(data=All_list,
                       columns=['Fault cell', 'Normal cell'])
    df1.to_csv('../Result_csv/Low_SOC.csv', index=False)

    return data_feature, U_max, U_min, corr_min, diff_max

def Data_Feature_Max_or_Min(data):  # 从数据矩阵中提取标准化参数
    U_max = np.max(data[0])
    U_min = np.min(data[0])
    corr_min = np.min(data[1])
    diff_max = np.max(data[2])
    return U_max, U_min, corr_min, diff_max

def Data_Origin_Saved_Npy(data_feature, path, Condition, Fault):  # 数据矩阵/原始路径/工况词典/故障词典
    for _, c_name in enumerate(Condition):
        if c_name in path:
            d_c = c_name  # 工况名
    for index, f_name in enumerate(Fault):  # index可以用来给相应数据进行标签on-hot编码
        if f_name in path:
            d_f = f_name  # 故障名
    file_name = d_c + "_" + d_f + "_" + path[-8:-4]  # 原文件日期部分
    dir_path = "../Npy/" + d_c + "/" + d_f  # 根据工况名和故障类型进行文件夹命名
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # 创建文件夹
    final_save_path = dir_path + "/" + file_name  # 存储路径
    np.save(final_save_path, data_feature)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    print("Data Saved Successfully!")
    print("Data Saved to " + final_save_path + ".npy")
    print("Total Length:" + str(data_feature.shape[1]))
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")


if __name__ == '__main__':
    base_dir = "../Origin"

    # 遍历最底层的所有 CSV 文件
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):  # 只处理 CSV 文件
                path_n = os.path.join(root, file)

                data_feature, U_max, U_min, corr_min, diff_max = Get_Data(path_n)
                print("U_max:" + str(U_max) + " U_min:" + str(U_min) + " corr_min:" + str(corr_min) + " diff_max:" + str(diff_max))
                print("Data Get OK!")

                U_max, U_min, corr_min, diff_max = 4.3, 3.0, 0.0, 0.85  # 统一标准化参数
                data_sample = Normalize(data_feature, U_max, U_min, corr_min, diff_max)  # 标准化
                
                print("SAMPLE")
                sample_index = 1
                plt.plot(range(len(data_sample[0])), data_sample[0, :, sample_index], color='b', label='Voltage')
                plt.plot(range(len(data_sample[0])), data_sample[1, :, sample_index], color='g', label='Corr')
                plt.plot(range(len(data_sample[0])), data_sample[2, :, sample_index], color='r', label='Diff')
                plt.legend()
                plt.show()

                data = Normalize(data_feature, U_max, U_min, corr_min, diff_max, pict_mode=True)
                data_o = Normalize(data_feature, U_max, U_min, corr_min, diff_max)
                data = torch.from_numpy(data)
                data = data.permute(2, 1, 0)  # 序号/时间/通道
                data = np.array(data)
                np.random.shuffle(data)  # 对序号进行打乱
                pict = Image.fromarray(np.uint8(data), mode="RGB")
                pict.show()
                print("Image Generate OK!")
