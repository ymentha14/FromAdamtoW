from pathlib import Path
from scipy.io import wavfile
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from speechpy.feature import mfcc
import pytorch_helper as ph
import wget
import zipfile
import os
import shutil

import helper


# url for the dataset
url = 'http://emodb.bilderbar.info/download/download.zip'
target_path = Path("./data/speech")
data_path = target_path.joinpath("wav")


# useful dics to convert labels from german to english
from torch.utils.data.dataset import Subset

DE2EN = {
    "W": "A",  # Wut-Anger
    "L": "B",  # Langeweile-Bordom
    "E": "D",  # Ekel-Disgust
    "A": "F",  # Angst-Fear
    "F": "H",  # Freude-Happiness
    "T": "S",  # Traueer-Sadness
    "N": "N",
}  # Neutral

DE2NUM = {item[0]: num for item, num in zip(DE2EN.items(), range(len(DE2EN)))}


class SpeechModel(nn.Module):
    """
    CNN classifier: inspired from "Emotion Recognition from Speech" (Kannan Venkataramanan,Haresh Rengaraj Rajamohan,2019)
    https://www.researchgate.net/publication/338138024_Emotion_Recognition_from_Speech

    Attributes:
        convblock[i] (nn.Sequential) : various convolutional blocks
        linblock (nn.Sequential): output layer
    Methods:
        forward : regular forward overriding
    """

    def __init__(self):
        super(SpeechModel, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=13), nn.BatchNorm2d(8), nn.ReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=13),
            nn.BatchNorm2d(8),
            nn.Dropout(0.33),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=13),
            nn.BatchNorm2d(8),
            nn.Dropout(0.33),
            nn.ReLU(),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.Dropout(0.33),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.linblock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1456, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),
        )

    def forward(self, x):
        x = self.convblock1(x.float())
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.linblock(x)
        return x


class SpeechDataset(Dataset):
    """
    Dataset to load wav files from EmoDB 
    """

    def __init__(self, data_root):
        """
        Args:
            data_root: path to the wav files directory
        """
        self.samples = []
        data_root = Path(data_root)
        data, sfs, targets, file_names = [], [], [], []
        for i, file in enumerate(data_root.iterdir()):
            sf, audio_data = wavfile.read(file)
            data.append(audio_data)
            sfs.append(sf)
            target = DE2NUM[file.name[5].capitalize()]
            targets.append(target)
            file_names.append(file.name)

        data = zeropadd(data, mode="mean")
        file_names = np.array(file_names)
        sfs = np.array(sfs)
        targets = np.array(targets)
        order = np.argsort(file_names)
        data = data[order]
        self.targets = targets[order]
        self.filenames = file_names[order]
        assert all([i == sfs[0] for i in sfs])
        self.sfs = sfs[0]
        self.data = get_mfcc(data, self.sfs)
        assert len(self.data) == len(self.filenames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        name = self.filenames[idx]
        return X, y


def zeropadd(data, mode="max"):
    """
    zero padds the audio files
    
    Args:
        mode(str):'max' or 'mean' if set to max, zero padds all the files to the max length of these files. Otherwise zero padds/cuts to the mean size.
        data(np.array): audio data
    Returns:
        data_padded(int): same data as in the arg, but zeropadded
    """
    if mode == "max":
        new_len = max([x.shape[0] for x in data])
    else:
        new_len = int(np.round(np.mean([x.shape[0] for x in data]) * 1.5))

    def padd(x):
        diff = abs(new_len - x.shape[0])
        shift = diff % 2
        diff //= 2
        if x.shape[0] < new_len:
            return np.pad(x, (diff, diff + shift), "constant")
        else:
            return x[diff : -(diff + shift)]

    data_padded = np.zeros((len(data), new_len))
    for i, x in enumerate(data):
        data_padded[i] = padd(x)
    return data_padded


def get_mfcc(data, sfs):
    """
    load the wav data
    Args:
        data(np.array): audio files
        sfs(np.array(int)): frequencies of the audio data
    Returns:
        (np.array): mel-frequency cepstrum of the audio data
    """
    if isinstance(sfs, (int, np.int64)):
        sfs = [sfs for i in range(len(data))]
    ret = np.array([mfcc(x, sf, num_cepstral=39) for x, sf in zip(data, sfs)])
    return np.expand_dims(ret, axis=1)


def get_model():
    """
    return the pytorch model that performs decently on the EmoDB dataset
    """
    # double necessary to work with the mfcc features
    return SpeechModel

def get_scoring_function():
    """
    Returns the function that computes the score, given the model and the data (as a torch DataLoader).
    In case of images_cls the scoring function is the accuracy (correct / total).
    Returns:
        score_func: (model: nn.Module, data: torch.utils.data.DataLoader) -> float
    """

    def accuracy(model: nn.Module, data: torch.utils.data.DataLoader):
        device = ph.get_device()
        model.eval()  # Define we are going to evaluate the model! No idea why, Pytorch stuff
        model.to(device=device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        return 100.0 * correct / total

    return accuracy

def get_full_dataset(sample_size):
    """
    Return a DataLoader for the training data.
    Args:
        sample_size: int, take a sample of smaller size in order to train faster. None: take all sample
    Returns:
        dataset: of type DataLoader
    """
    download()
    full_dataset = SpeechDataset(str(data_path))

    if sample_size is not None:
        # If we want a smaller subset, we just sample a subset of the given size.
        full_dataset = helper.get_sample(sample_size)

    return full_dataset



def download():
    """
    download the data from EMoDB website
    """
    target_path.mkdir(parents=True,exist_ok=True)
    zip_path = target_path.joinpath("download.zip")

    if data_path.exists():
        return None

    if (not zip_path.exists()):
        print("No existing zip file found: start downloading")
        wget.download(url, str(target_path))

    with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
        print("Extracting zip file..")
        zip_ref.extractall(str(target_path))
        
    assert(data_path.exists())
    shutil.rmtree(str(target_path.joinpath("lablaut")))
    shutil.rmtree(str(target_path.joinpath("labsilb")))
    shutil.rmtree(str(target_path.joinpath("silb")))
    os.remove(str(target_path.joinpath("erkennung.txt")))
    os.remove(str(target_path.joinpath("erklaerung.txt")))
    print("Download successful.")


if __name__ == "__main__":
    model = get_model()
    dataloader = get_data()
    nb_epochs = 4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("Start training Model")
    for e in range(nb_epochs):
        print("Epoch:{}/{}".format(e, nb_epochs))
        for X_batch, y_batch in dataloader:
            output_batch = model(X_batch)
            loss = criterion(output_batch, y_batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()
