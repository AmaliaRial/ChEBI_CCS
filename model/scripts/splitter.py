from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(
	df: pd.DataFrame,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Hace un único split: 80% train y 20% test"""
	train_df, test_df = train_test_split(
		df,
		test_size=test_size,
		random_state=random_state,
		shuffle=True,
	)
	return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split(
	input_csv: str | Path,
	train_csv: str | Path,
	test_csv: str | Path,
	test_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	df = pd.read_csv(input_csv)
	train_df, test_df = split_train_test(df, test_size=test_size, random_state=random_state)

	Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
	Path(test_csv).parent.mkdir(parents=True, exist_ok=True)
	train_df.to_csv(train_csv, index=False)
	test_df.to_csv(test_csv, index=False)

	return train_df, test_df
