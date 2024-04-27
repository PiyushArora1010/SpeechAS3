import torch
from torchaudio.pipelines import WAV2VEC2_XLSR_300M

# Load the model
model = WAV2VEC2_XLSR_300M.get_model()

state_dic = torch.load("check/Best_LA_model_for_DF.pth")
print(state_dic.keys())