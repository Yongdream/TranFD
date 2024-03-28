import torch
from torch import nn
import numpy as np
import model_Conv
import model_Vit
import dataset_get
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import tqdm

all = torch.load("./Model/Default_Conv.pt")
a = all.popitem(last=False)
b = all.popitem(last=False)
c = all.popitem(last=False)
d = all.popitem(last=False)
print("OK")