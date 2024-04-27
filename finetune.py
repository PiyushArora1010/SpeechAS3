import torch

from module.metrics import AUC_and_EER
from module.model import load_model
from module.data import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-5,
}

if __name__ == "__main__":
    data = AudioDataset('Dataset_Speech_Assignment', cut = 4000)
    model = load_model(device).train()
    dataloader = torch.utils.data.DataLoader(data, batch_size=args['batch_size'], num_workers=4, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args['epochs']):
        model.train()
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        auc, eer = AUC_and_EER(dataloader, model, device)
        print(f"Epoch {epoch+1}/{args['epochs']} AUC: {auc}, EER: {eer}")
    torch.save(model.state_dict(), 'model.pth')