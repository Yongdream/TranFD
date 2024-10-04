from torch import nn
import numpy as np
import torch
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def initialize_weights(self):
    for m in self.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zeros_()

class SELF(nn.Module):
    def __init__(self,):
        super(SELF, self).__init__()
        self.name = 'SELF'

        self.cnn1 = nn.Conv2d(3, 16, 3)  # 消融：修改首个通道数即可
        # self.bn1 = nn.BatchNorm2d(num_features=16)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.cnn2 = nn.Conv2d(16, 32, 3)
        # self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pooling2 = nn.MaxPool2d(2, 2)
        # self.cnn3 = nn.Conv2d(32, 64, 3)

        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(7008, 1000)
        self.dropout1 = nn.Dropout(0.4)
        # self.act1 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(1000, 5)
        # self.act2 = nn.Sigmoid()
        self.act2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn1(x)
        # x = self.bn1(x)
        x = self.pooling1(x)
        x = self.cnn2(x)
        # x = self.bn2(x)
        x = self.pooling2(x)
        # x = self.cnn3(x)
        x = self.flat(x)  # 展平张量连接MLP
        # x = x.view(x.shape[0], -1)  # 使得批次处理成为可能
        # x = self.dropout1(x)
        x = self.lin1(x)
        # x = self.act1(x)
        # x = self.dropout2(x)
        x = self.lin2(x)
        # x = self.act2(x)
        x = x.view(x.shape[0], 1, -1)

        return x



# if __name__ == '__main__':
#     test = torch.randn(32, 1, 300, 20)
#     model = SELF()
#     output = model(test)
#     print(output.shape)