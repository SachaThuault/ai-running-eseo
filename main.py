import pandas as pd

# read
df = pd.read_parquet('data/run_ww_2019_m.parquet')
print(f"Nom des colonnes: {df.columns}")
print(f"Nombre de lignes: {len(df)}")
print(df.distance.describe())
print(df.athlete)
print(df.loc[df.athlete == 37594])
