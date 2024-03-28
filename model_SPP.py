from torch import nn
import numpy as np
import torch
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
            math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


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
    def __init__(self, ):
        super(SELF, self).__init__()
        self.name = 'SELF'

        self.cnn1 = nn.Conv2d(3, 16, (10, 1))  # 对比实验修改首个通道数即可
        self.cnn2 = nn.Conv2d(16, 32, (10, 1))

        self.spp = SPPLayer(num_levels=3)

        self.lin1 = nn.Linear(448, 1000)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = self.spp(x)

        x = self.dropout1(x)
        x = self.lin1(x)
        x = self.dropout2(x)
        x = self.lin2(x)
        x = x.view(x.shape[0], 1, -1)

        return x

    def __init__(self, ):
        super(SELF, self).__init__()
        self.name = 'SELF'

        self.cnn1 = nn.Conv2d(1, 16, 3)  # 对比实验修改首个通道数即可
        self.cnn2 = nn.Conv2d(16, 32, 3)

        self.spp = SPPLayer(num_levels=3)

        self.lin1 = nn.Linear(448, 1000)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = self.spp(x)

        x = self.dropout1(x)
        x = self.lin1(x)
        x = self.dropout2(x)
        x = self.lin2(x)
        x = x.view(x.shape[0], 1, -1)

        return x


if __name__ == '__main__':
    test = torch.randn(32, 3, 300, 20)
    model = SELF()
    output = model(test)
    print(output.shape)