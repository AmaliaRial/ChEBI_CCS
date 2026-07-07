from __future__ import annotations

import json
import numpy as np


class AdductOneHotEncoder:
    def __init__(self) -> None:
        self.converter: dict[str, int] = {} #creates an empty storer, for the mapping, for each unique adduct type each is a number
        self._is_fit = False #marks the encoder as not fitted yet

    def fit(self, adducts: np.ndarray) -> None:
        unique = sorted({str(a).strip() for a in adducts}) #converts evey adduct to a string, removes spaces at the beggining and end
        self.converter = {adduct: idx for idx, adduct in enumerate(unique)}#creates the dictionary that maps each adduct to a numerical position
        self._is_fit = True #now its fitted and ready to transform into vetcors

    def transform(self, adducts: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Adduct encoder must be fit first")

        encoded = np.zeros((len(adducts), len(self.converter)), dtype=np.float32) #creates a matrix full of zeros
        for i, adduct in enumerate(adducts):
            key = str(adduct).strip()
            if key in self.converter:
                encoded[i, self.converter[key]] = 1.0 #places a 1 in the vector position of said adduct
        return encoded #a one hot encoded matrix is exported

    def save_encoder(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.converter, f)

    def load_encoder(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.converter = json.load(f)
        self._is_fit = True
