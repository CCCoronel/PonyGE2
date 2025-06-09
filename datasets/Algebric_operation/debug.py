import pandas as pd

# Carrega o arquivo
df = pd.read_csv("Algebric_operation/Train.txt", sep='\s+', header=None)

# Mostra número de linhas e colunas
print("Número de linhas e colunas:", df.shape)

# Opcional: mostra as primeiras linhas
print(df.head())