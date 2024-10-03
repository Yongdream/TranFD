import torch

all = torch.load("./Model/Default_Conv.pt")
a = all.popitem(last=False)
b = all.popitem(last=False)
c = all.popitem(last=False)
d = all.popitem(last=False)
print("OK")