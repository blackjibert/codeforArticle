import torch
from model.base_model import Net
print(123456)
model = torch.jit.trace(Net(), torch.zeros([1, 1, 28, 28], dtype=torch.float))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(123, device)
model.to(device) # 移动模型到cuda