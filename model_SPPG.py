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
    def __init__(self, ):
        super(SELF, self).__init__()
        self.name = 'SELF'

        self.cnn1 = nn.Conv2d(3, 16, (15, 1))  # 对比实验修改首个通道数即可
        self.cnn2 = nn.Conv2d(16, 32, (15, 1))

        self.SPG_1 = nn.Conv2d(32, 4, (136, 1), stride=(136, 1))
        self.SPG_2 = nn.Conv2d(32, 4, (68, 1), stride=(68, 1))
        self.SPG_3 = nn.Conv2d(32, 4, (34, 1), stride=(34, 1))

        self.flat = nn.Flatten()


        self.lin1 = nn.Linear(1120, 1000)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        x1 = self.flat(self.SPG_1(x))
        x2 = self.flat(self.SPG_2(x))
        x3 = self.flat(self.SPG_3(x))


        x = torch.cat([x1, x2, x3], 1)

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