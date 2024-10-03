import numpy as np
import os
import data_get

def Data_Cut_Saved_Npy(data_feature, path, Condition, Fault, start, win, filename):  # 数据矩阵/原始路径/工况词典/故障词典/起始时间/截取长度/截取步长
    for _, c_name in enumerate(Condition):  # 从文件名中读取出工况名和故障名以创建路径
        if c_name in path:
            d_c = c_name  # 工况名
        for index, f_name in enumerate(Fault):  # index可以用来给相应数据进行标签on-hot编码
            if f_name in path:
                d_f = f_name  # 故障名
    if d_f == "Vis":
        file_name = d_c + "_" + d_f + "_" + filename[9:-4] + "_" + str(start) + "-" + str(start + win)
    else:
        file_name = d_c + "_" + d_f + "_" + filename[-8:-4] + "_" + str(start) + "-" + str(start + win)
    dir_path = "../Cut_Npy/" + d_c + "/" + d_f  # 存储路径
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    final_save_path = dir_path + "/" + file_name
    np.save(final_save_path, data_feature)
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    print("Data Saved Successfully!")
    print("Data Saved to " + final_save_path + ".npy")
    print("Total Length:" + str(data_feature.shape[1]))
    # print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

def Read_Cut_and_Save(path, win, U_max, U_min, corr_min, diff_max, step, filename):
    data = np.load(path)  # 读取数据
    data = data_get.Normalize(data, U_max, U_min, corr_min, diff_max)  # 标准化
    Condition = ["FUDS", "UDDS", "US06"]  # 工况词典
    Fault = ["Cor", "Isc", "Noi", "Nor", "Vis"]  # 相关性/微短路/噪声/正常/粘滞
    for i in range(0, data.shape[1] - win, step):
        if "Vis" in path:
            data_cut = data[:, i:i + win, :]
            Data_Cut_Saved_Npy(data_cut, path, Condition, Fault, i, win, filename)
        else:
            if (0 < i < 300) or (1100 < i < 1300) or (2100 < i < 2300) or (3100 < i < 3300) or (4100 < i < 4300) or (5100 < i < 5300):
                data_cut = data[:, i:i + win, :]
                Data_Cut_Saved_Npy(data_cut, path, Condition, Fault, i, win, filename)

def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path)  # 这个可以删除单个文件，不能删除文件夹
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            # if file_name != 'wibot.log':
            tf = os.path.join(dir_path, file_name)
            del_files(tf)
    print('A file has been cleaned!')

if __name__ == '__main__':
    win = 300
    Condition_d = ["FUDS", "UDDS", "US06"]  # 工况词典
    Fault_d = ["Cor", "Isc", "Noi", "Nor", "Vis"]
    Condition_selected = "US06"
    Dst_path = "../Cut_Npy/" + Condition_selected + "/"
    del_files(Dst_path)  # 清除原有数据
    print("Clean ok!")
    U_max, U_min, corr_min, diff_max = 4.3, 3.0, 0.0, 0.85  # 统一标准化参数
    Origin_path = "../Npy/" + Condition_selected + "/"
    for fault_index, Fault_name in enumerate(Fault_d):
        dst_dir = Origin_path + Fault_name + "/"
        FILE_list = os.listdir(dst_dir)
        for filename_index, File_name in enumerate(FILE_list):
            path = dst_dir + File_name
            Read_Cut_and_Save(path, win, U_max, U_min, corr_min, diff_max, step=5, filename=File_name)
