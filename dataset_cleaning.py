import pandas as pd
from pathlib import Path

# allccs: AllCCS ID,Name,Structure,Formula,Type,Adduct,m/z,CCS,Confidence level,Update date,InChI, (structure es SMILES)

# Obtener todos los archivos CSV de la carpeta data/raw_datasets
# csv_files = Path('data/raw_datasets').glob('*.csv')

# # Importar todos los CSV en un diccionario
# datasets = {}
# for file in csv_files:
#     datasets[file.stem] = pd.read_csv(file)

import pandas as pd

# Importar el archivo CSV
df = pd.read_csv("data/raw_datasets/METLIN-CCS-Lipids_descriptors.csv")

# Mostrar solo la columna "nStructures"
print(df['nStructures'])
