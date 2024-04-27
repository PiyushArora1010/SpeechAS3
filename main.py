import torch
from module.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'check/xlsr2_300m.pt'

model = Model(path, device)

stateDic = torch.load('check/LA_model.pth')
del stateDic['args'], stateDic['cfg']
model.load_state_dict(stateDic)