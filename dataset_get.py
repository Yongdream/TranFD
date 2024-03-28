import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomedDataSet(Dataset):
    def __init__(self, train=True, train_x=None, train_y=None, test_x=None, test_y=None, val=False, transform=None):
        self.train = train
        self.val = val
        self.transform = transform
        if self.train:
            self.dataset = train_x
            self.labels = train_y
        elif self.val:
            self.dataset = test_x
            self.labels = test_y
        else:
            self.dataset = test_x

    def __getitem__(self, index):
        if self.train:
            # return torch.Tensor(self.dataset[index].astype(float)).to(device), self.labels[index].to(device)
            return torch.Tensor(self.dataset[index]).to(device), self.labels[index].to(device)
        elif self.val:
            # return torch.Tensor(self.dataset[index].astype(float)).to(device), self.labels[index].to(device)
            return torch.Tensor(self.dataset[index]).to(device), self.labels[index].to(device)
        else:
            # return torch.Tensor(self.dataset[index].astype(float)).to(device)
            return torch.Tensor(self.dataset[index]).to(device)

    def __len__(self):
        return self.dataset.shape[0]

class CustomedDataSet_n(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
            return (self.data[index][0], self.data[index][1])

    def __len__(self):
        return self.data.shape[0]

def Generate_Dataset_from_Dir(path, condition):
    assert condition in ["FUDS", "UDDS", "US06"]  # 确保工况正确
    Fault = ["Cor", "Isc", "Noi", "Nor", "Vis"]  # 相关性/微短路/噪声/正常/粘滞
    # Fault = ["Cor", "Noi", "Nor", "Vis"]
    train_path = os.path.join(path, condition, 'Train')  # 训练集路径
    train_data = []  # 训练数据
    train_label = []  # 训练标签
    FILE_list = os.listdir(train_path)
    for _, file_name in enumerate(FILE_list):
        for fault_index, fault_name in enumerate(Fault):  # 对每个文件名进行核验，以故障词典对其进行验证
            if fault_name in file_name:
                current_label = np.zeros((1, len(Fault)))  # 创建原始标签向量
                current_label[0, fault_index] = 1  # one-hot编码
                current_data = np.load(os.path.join(train_path, file_name))  # 加载Npy文件数据
                train_data.append(current_data)  # 将当前加载数据加入训练数据列表
                train_label.append(current_label)  # 将当前标签加入训练标签列表
    val_path = os.path.join(path, condition, 'Val')  # 验证集路径
    val_data = []  # 验证数据
    val_label = []  # 验证标签
    FILE_list = os.listdir(val_path)
    for _, file_name in enumerate(FILE_list):
        for fault_index, fault_name in enumerate(Fault):  # 对每个文件名进行核验，以故障词典对其进行验证
            if fault_name in file_name:
                current_label = np.zeros((1, len(Fault)))  # 创建原始标签向量
                current_label[0, fault_index] = 1  # one-hot编码
                current_data = np.load(os.path.join(val_path, file_name))  # 加载Npy文件数据
                val_data.append(current_data)  # 将当前加载数据加入验证数据列表
                val_label.append(current_label)  # 将当前标签加入验证标签列表

    train_data, train_label, val_data, val_label = np.array(train_data), np.array(train_label), np.array(val_data), np.array(val_label)

    return train_data, train_label, val_data, val_label

if __name__ == '__main__':
    train_data, train_label, val_data, val_label = Generate_Dataset_from_Dir("./Dataset", "UDDS")
    train_data, train_label, val_data, val_label = torch.from_numpy(train_data), torch.from_numpy(train_label), torch.from_numpy(val_data), torch.from_numpy(val_label)
    train_data_a = []
    for i in range(0, len(train_data)):
        train_data_a.append([train_data[i], train_label[i]])
    train_data_a = np.array(train_data_a, dtype=object)
    val_data_a = []
    for i in range(0, len(val_data)):
        val_data_a.append([val_data[i], val_label[i]])
    val_data_a = np.array(val_data_a, dtype=object)
    print("Read OK!")
    train_set = CustomedDataSet(train_x=train_data, train_y=train_label)
    val_set = CustomedDataSet(test_x=val_data, test_y=val_label, train=False, val=True)
    train_set_a = CustomedDataSet_n(data=train_data_a)
    val_set_a = CustomedDataSet_n(data=val_data_a)
    print("Dataset OK!")
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False)
    train_loader_a = DataLoader(dataset=train_set_a, batch_size=32, shuffle=True)
    val_loader_a = DataLoader(dataset=val_set_a, batch_size=32, shuffle=False)
    print("DataLoader OK!")



