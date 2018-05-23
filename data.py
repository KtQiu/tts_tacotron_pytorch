import os
import re
import unicodedata
import collections
# import codecs
import numpy as np
import pandas as pd
import librosa
# from util import *
import util

import torch
# import torch.nn as nn
from torch.utils.data import Dataset

from hparam import tacotron_hparams


class LJDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, wav_dir):
        self.text_file = pd.read_csv(csv_file, sep='|', header=None)
        self.wav_dir = wav_dir

    def load_wav(self, wav_name):
        return librosa.load(wav_name, sr=tacotron_hparams["sample_rate"])

    def __len__(self):
        return len(self.text_file)

    def __getitem__(self, idx):
        char2idx, _ = load_vocab()
        wav_name = os.path.join(self.wav_dir, self.text_file.ix[idx, 0]) + '.wav'
        text = text_normalize(self.text_file.ix[idx, 1]) + 'E' # add End token
        text = [char2idx[char] for char in text] # char2idx
        text = np.asarray(text, dtype=np.int32) # translate to ndarray
        wav = np.asarray(self.load_wav(wav_name)[0], dtype=np.float32)
        sample = {'text': text, 'wav': wav}
        return sample


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(tacotron_hparams["vocab"]), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(tacotron_hparams["vocab"])}
    idx2char = {idx: char for idx, char in enumerate(tacotron_hparams["vocab"])}
    return char2idx, idx2char


def get_dataset():
    return LJDataset(
        os.path.join(tacotron_hparams["data_path"], 'metadata.csv'),
        os.path.join(tacotron_hparams["data_path"], 'wavs'))



def fetch_batch(batch):
    
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        keys = list()

        text = [d['text'] for d in batch]
        wav = [d['wav'] for d in batch]

        # PAD sequences with largest length of the batch
        text = util._prepare_data(text).astype(np.int32)
        wav = util._prepare_data(wav)

        magnitude = np.array([util.spectrogram(w) for w in wav])
        mel = np.array([util.melspectrogram(w) for w in wav])
        timesteps = mel.shape[-1]

        # PAD with zeros that can be divided by outputs per step
        if timesteps % tacotron_hparams["outputs_per_step"] != 0:
            magnitude = util._pad_per_step(magnitude)
            mel = util._pad_per_step(mel)

        return text, magnitude, mel

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

