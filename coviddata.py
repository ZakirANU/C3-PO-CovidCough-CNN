import os
from typing import List
import pandas as pd
import librosa
import librosa.display
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import soundfile
import pickle
from pathlib import Path
from copy import deepcopy

from audiomentations import Compose, TimeStretch, PitchShift, Shift, Trim, Gain, PolarityInversion
from sklearn.model_selection import train_test_split
from utils.segmentation import segment_cough


class CovidData:
    """
    data passed into CovidData must have structure:
    data = {
        filename: List[str], 
        signal: List[np.ndarray],
        sr: List[int]
        label: List[int]
    }   
    """
    
    def __init__(self, data: pd.DataFrame):
        # Deep copy, global variable will not be effected.
        self.data = pd.DataFrame(data = deepcopy(data.values), columns = data.columns)

    def deepcopy(self):
        return CovidData(self.data)

    def resample(self, target_sr):
        print("Resampling...")
        new_signals = []
        for signal, orig_sr in tqdm(
                zip(self.data["signal"], self.data["sr"]), 
                total = len(self.data)):
            new_signals.append(librosa.resample(signal, orig_sr = orig_sr, target_sr = target_sr))
        self.data["signal"] = new_signals
        self.data["sr"] = [target_sr for i in range(len(self.data))]


    def balance(self):
        print("Balancing...")
        n_mino = self.data["label"].value_counts().min()

        self.data = self.data.groupby("label").sample(n = n_mino, random_state = 7)


    def balance_multiset(self, n_set):
        print("Balancing...")
        label_count = self.data["label"].value_counts()
        n_mino = label_count.min()
        n_majo = label_count.max()
        label_mino = label_count.argmin()
        label_majo = label_count.argmax()
        assert n_set * n_mino <= n_majo, "Do not have enough majority data."

        minority = self.data.loc[self.data["label"] == label_mino]
        majority = self.data.loc[self.data["label"] == label_majo]

        majority = majority.sample(n = n_set * n_mino, random_state = 7, replace = False)
        balanced_datasets = []
        for i in range(n_set):
            majority_i = majority[i * n_mino: (i + 1) * n_mino]
            balanced_i = pd.concat([majority_i, minority])
            balanced_datasets.append(CovidData(balanced_i))

        return balanced_datasets

    
    def normalise(self):
        print("Normalising...")
        new_signals = []
        for signal in tqdm(self.data["signal"]):
            signal_norm = signal / np.abs(signal).max()
            new_signals.append(signal_norm)
        self.data["signal"] = new_signals


    def segment(self):
        print("Segmenting...")
        data = {
            "filename": [],
            "signal": [],
            "sr": [],
            "label": []
        }

        for _, row in tqdm(self.data.iterrows(), total = len(self.data)):
            cough_segments, cough_mask = segment_cough(row.signal, row.sr, cough_padding=0)

            for i, seg in enumerate(cough_segments):
                data["signal"].append(seg)
                data["filename"].append(row.filename)
                data["sr"].append(row.sr)
                data["label"].append(row.label) 

        self.data = pd.DataFrame(data)

    def augment(self, n = 1, effect = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.4, p=0.9),
            PitchShift(min_semitones=-2, max_semitones=4, p=1),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.8),
            Trim(p=1),
            Gain(p=1),
            PolarityInversion(p=0.8)
        ])):
        
        print("Augmenting...")
        data_augmented = {
            "filename": [],
            "signal": [],
            "sr": [],
            "label": []
        }
        
        for _, row in tqdm(self.data.iterrows(), total = len(self.data)):
            for i in range(n):
                augmented = effect(row.signal, row.sr)
                data_augmented["signal"].append(augmented)
                data_augmented["filename"].append(row.filename)
                data_augmented["sr"].append(row.sr)
                data_augmented["label"].append(row.label) 

        df_augmented = pd.DataFrame(data_augmented)
        self.data = pd.concat(self.data, df_augmented)


    def split(self, frac):
        data_A = {
            "filename": [],
            "signal": [],
            "sr": [],
            "label": []
        }
        data_B = {
            "filename": [],
            "signal": [],
            "sr": [],
            "label": []
        }
        labels = self.data["label"].unique()
        for label in labels:
            data_of_label = self.data.loc[self.data["label"] == label]
            m = round(len(data_of_label) * frac)
            count = 0
            filenames = data_of_label["filename"].unique()
            np.random.shuffle(filenames)
            for filename in filenames:
                data_to_append = data_A if count < m else data_B
                for _, row in data_of_label.loc[data_of_label["filename"] == filename].iterrows():
                    data_to_append["filename"].append(row.filename)
                    data_to_append["signal"].append(row.signal)
                    data_to_append["sr"].append(row.sr)
                    data_to_append["label"].append(row.label)
                    count += 1
        
        cdata_A = CovidData(pd.DataFrame(data_A))
        cdata_B = CovidData(pd.DataFrame(data_B))

        return cdata_A, cdata_B
    

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self)


    def load(load_path):
        with open(load_path, "rb") as f:
            obj = pickle.load(f)

        return obj


    def stat(self):
        print(self.data[["label", "filename"]].groupby("label").count().rename(columns = {"filename": "number"}))



class MultiCovidData:
    def __init__(self, data_list: List[CovidData]):
        self.data_list = data_list


    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self)


    def load(load_path):
        with open(load_path, "rb") as f:
            obj = pickle.load(f)

        return obj