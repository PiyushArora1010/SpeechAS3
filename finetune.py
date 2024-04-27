import argparse

import torch

from module.metrics import AUC_and_EER
from module.model import load_model
from module.data import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='fine-tune the model')

parser.add_argument('--path', type=str, default='model.pth', help='Path to the model')
parser.add_argument('--dataset', type=str, default='Dataset_Speech_Assignment', help='Path to the dataset')
parser.add_argument('--cut', type=int, default=4000, help='Cut the audio to this length')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')

args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":
    traindata = AudioDataset(args['dataset'], cut = args['cut'], typ='training')
    valdata = AudioDataset(args['dataset'], cut = args['cut'], typ='validation')
    testdata = AudioDataset(args['dataset'], cut = args['cut'], typ='testing')
    model = load_model(device, args['path']).train()

    traindataloader = torch.utils.data.DataLoader(traindata, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
    valdataloader = torch.utils.data.DataLoader(valdata, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
    testdataloader = torch.utils.data.DataLoader(testdata, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    auc_best = -1
    for epoch in range(args['epochs']):
        model.train()
        for (x, y) in traindataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        auc, eer = AUC_and_EER(valdataloader, model, device)
        if auc > auc_best:
            auc_best = auc
            torch.save(model.state_dict(), 'model.pth')
        print(f"Epoch {epoch+1}/{args['epochs']} VAL AUC: {auc}, VAL EER: {eer}")
    
    model.load_state_dict(torch.load('model.pth'))
    auc, eer = AUC_and_EER(testdataloader, model, device)
    print(f"TEST AUC: {auc}, TEST EER: {eer}")