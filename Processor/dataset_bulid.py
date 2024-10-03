import random
import numpy as np
import os
import shutil

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


def mycopyfile(srcfile, dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        fpath, fname = os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))

Condition_d = ["FUDS", "UDDS", "US06"]  # 工况词典
Fault_d = ["Cor", "Isc", "Noi", "Nor", "Vis"]
Condition_selected = "US06"  # 选择要处理的工况

Dst_path = "../Dataset/" + Condition_selected + "/"  # 基础数据及保存目录
del_files(Dst_path)  # 清除原有数据
print("Clean ok!")
Origin_path = "../Cut_Npy/" + Condition_selected + "/"  # 基础读取目录
for fault_index, Fault_name in enumerate(Fault_d):  # 遍历每种故障
    dst_dir = Origin_path + Fault_name + "/"  # 工况/故障文件夹目录
    FILE_list = os.listdir(dst_dir)  # 获取目录下的文件名列表
    File_count = len(FILE_list)  # 获取文件总数
    File_index_array = list(np.arange(File_count))  # 构造与文件总数长度一致的索引矩阵
    Train_count = int(File_count * 0.75)  # 每个类别取75%作为训练集，获取训练集中需要采样的次数
    print(Fault_name + ":" + str(Train_count))  # 显示每个故障的训练集文件数
    Train_index_array = list(random.sample(File_index_array, Train_count))  # 从原有索引矩阵中随机筛选出训练集文件对应的索引
    Val_index_array = list(np.delete(File_index_array, Train_index_array, 0))  # 删除训练集文件对应的索引以获得验证集文件对应的索引
    for i in range(0, len(Train_index_array)):  # 遍历训练集文件索引矩阵
        dst_file_name = FILE_list[Train_index_array[i]]  # 根据索引矩阵的数据从文件名列表中获取文件名
        dst_file_wholename = Origin_path + Fault_name + "/" + dst_file_name  # 构建原Npy文件的完整路径
        mycopyfile(dst_file_wholename, Dst_path + "Train/")  # 复制文件到训练集文件夹
    print("Successfully bulid dataset of %s for Training!" % Fault_name)
    for i in range(0, len(Val_index_array)):  # 遍历验证集文件索引矩阵
        dst_file_name = FILE_list[Val_index_array[i]]  # 根据索引矩阵的数据从文件名列表中获取文件名
        dst_file_wholename = Origin_path + Fault_name + "/" + dst_file_name  # 构建原Npy文件的完整路径
        mycopyfile(dst_file_wholename, Dst_path + "Val/")  # 复制文件到验证集文件夹
    print("Successfully bulid dataset of %s for Valiation!" % Fault_name)
