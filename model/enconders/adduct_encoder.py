from __future__ import annotations

import json
import numpy as np


class AdductOneHotEncoder:
    def __init__(self) -> None:
        self.converter: dict[str, int] = {}
        self._is_fit = False

    def fit(self, adducts: np.ndarray) -> None:
        unique = sorted({str(a).strip() for a in adducts})
        self.converter = {adduct: idx for idx, adduct in enumerate(unique)}
        self._is_fit = True

    def transform(self, adducts: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Adduct encoder must be fit first")

        encoded = np.zeros((len(adducts), len(self.converter)), dtype=np.float32)
        for i, adduct in enumerate(adducts):
            key = str(adduct).strip()
            if key in self.converter:
                encoded[i, self.converter[key]] = 1.0
        return encoded

    def save_encoder(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.converter, f)

    def load_encoder(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.converter = json.load(f)
        self._is_fit = True
