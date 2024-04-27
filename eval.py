import argparse

import torch

from module.metrics import AUC_and_EER
from module.model import load_model
from module.data import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Evaluate the model')

parser.add_argument('--path', type=str, default='model.pth', help='Path to the model')
parser.add_argument('--dataset', type=str, default='Dataset_Speech_Assignment', help='Path to the dataset')
parser.add_argument('--cut', type=int, default=4000, help='Cut the audio to this length')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')

args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":
    if 'for' in args['dataset']:
        data = AudioDataset(args['dataset'], cut = args['cut'], typ='testing')
    else:
        data = AudioDataset(args['dataset'], cut = args['cut'])
    model = load_model(device, args['path'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
    auc, eer = AUC_and_EER(dataloader, model, device)
    print(f"AUC: {auc}, EER: {eer}")
