import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as io
import pandas as pd
import os
import math
import torch
import data_get

### UDDS DSOC故障
# path = "./Npy/UDDS/Cor/UDDS_Cor_0627.npy"
# path = "./Npy/UDDS/Cor/UDDS_Cor_0628.npy"
# path = "./Npy/UDDS/Cor/UDDS_Cor_0629.npy"
### FUDS DSOC故障
# path = "./Npy/FUDS/Cor/FUDS_Cor_0630.npy"
# path = "./Npy/FUDS/Cor/FUDS_Cor_0715.npy"
# path = "./Npy/FUDS/Cor/FUDS_Cor_0718.npy"
### UDDS ISC故障
# path = "./Npy/UDDS/Isc/UDDS_Isc_0718.npy"  # 1ohmISC
# path = "./Npy/UDDS/Isc/UDDS_Isc_0628.npy"  # 5ohmISC
# path = "./Npy/UDDS/Isc/UDDS_Isc_0630.npy"  # 10ohmISC
### FUDS ISC故障
# path = "./Npy/FUDS/Isc/FUDS_Isc_0718.npy"  # 1ohmISC
# path = "./Npy/FUDS/Isc/FUDS_Isc_0714.npy"  # 5ohmISC
path = "./Npy/FUDS/Isc/FUDS_Isc_0715.npy"  # 10ohmISC
### US06 ISC故障
# path = "./Npy/US06/Isc/US06_Isc_0907.npy"  # 1ohmISC
### UDDS 传感器噪声
# path = "./Npy/UDDS/Noi/UDDS_Noi_s0_05_0328.npy"  # Sigma0.05
# path = "./Npy/UDDS/Noi/UDDS_Noi_s0_1_0706.npy"  # Sigma0.1
# path = "./Npy/UDDS/Noi/UDDS_Noi_s0_2_0711.npy"  # Sigma0.2
### FUDS 传感器噪声
# path = "./Npy/FUDS/Noi/FUDS_Noi_s0_05_0704.npy"  # Sigma0.05
# path = "./Npy/FUDS/Noi/FUDS_Noi_s0_1_0707.npy"  # Sigma0.1
# path = "./Npy/FUDS/Noi/FUDS_Noi_s0_2_0712.npy"  # Sigma0.2
### UDDS 正常
# path = "./Npy/UDDS/Nor/UDDS_Nor_0704.npy"
### FUDS 正常
# path = "./Npy/FUDS/Nor/FUDS_Nor_0706.npy"
### UDDS 传感器粘滞
# path = "./Npy/UDDS/Vis/UDDS_Vis_300-900_0411.npy"
### FUDS 传感器粘滞
# path = "./Npy/FUDS/Vis/FUDS_Vis_300-900_0705.npy"

###突发型
# path = "./Npy/FUDS/Vis/FUDS_Vis_300-900_0705.npy"

data = np.load(path)[:, 4100:4400, :]
# data = np.load(path)[:, -300:, :]

U_max, U_min, corr_min, diff_max = 4.3, 3.0, 0.0, 0.85
data = data_get.Normalize(data, U_max, U_min, corr_min, diff_max)

ab_index = 2
All_list = []
for i in range(0, 300):
    All_list.append([data[0, i, ab_index], data[1, i, ab_index], data[2, i, ab_index]])

df1 = pd.DataFrame(data=All_list,
                   columns=['Voltage', 'Correlation', 'VD'])
# df1.to_csv('./Visual_csv_and_pict/Vis.csv', index=False)

plt.plot(range(0, 300), data[0, :, ab_index], color='b', label='Voltage')
plt.plot(range(0, 300), data[1, :, ab_index], color='g', label='Corr')
plt.plot(range(0, 300), data[2, :, ab_index], color='r', label='Diff')
plt.legend()
plt.show()

data = torch.from_numpy(data)
data = np.array(data.permute(2, 1, 0))
np.random.shuffle(data)

pict = Image.fromarray(np.uint8(255 * np.array(data)), mode="RGB")  # 调试
pict.show()
# pict.save("./Fault_Visual/US06_ISC_1Ohm_1.png")
# pict.save("./Visual_csv_and_pict/Vis.png")