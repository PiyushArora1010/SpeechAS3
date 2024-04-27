import torch

from module.metrics import AUC_and_EER
from module.model import load_model
from module.data import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data = AudioDataset('Dataset_Speech_Assignment', cut = 4000)
    
    model = load_model(device)

    dataloader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=4, shuffle=True)
    
    auc, eer = AUC_and_EER(dataloader, model, device)
    print(f"AUC: {auc}, EER: {eer}")