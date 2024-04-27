import os
from torch.utils.data import Dataset
from torch import Tensor
import torchaudio
import numpy as np

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class AudioDataset(Dataset):
    def __init__(self, root_dir, cut=64600):
        self.root_dir = root_dir
        self.cut = cut
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        classes = ['Real', 'Fake']

        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if file_path[-3:] == "wav":
                    self.samples.append((file_path, idx))
                elif file_path[:-3] == "mp3":
                    self.samples.append((file_path, idx))
                else:
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file_path, label = self.samples[idx]
        X, fs = torchaudio.load(file_path)
        X = np.array(X[0])

        X_pad = pad(X,self.cut)
        audio = Tensor(X_pad)
        return audio, label