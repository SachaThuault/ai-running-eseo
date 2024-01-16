import pandas as pd

# read daily
df = pd.read_parquet('../../data/run_ww_2019_d.parquet')
df2 = pd.read_parquet('../../data/run_ww_2020_d.parquet')

# répartition des genres 2019
print("---------------------------------------------------")
gender_counts = df.groupby(['age_group', 'gender', 'datetime']).size().unstack()
print("Répartition des genres en 2019 : ")
print(gender_counts.mean(axis=1))

# répartition des genres 2020
print("---------------------------------------------------")
gender_counts2 = df2.groupby(['age_group', 'gender', 'datetime']).size().unstack()
print("Répartition des genres en 2020 : ")
print(gender_counts2.mean(axis=1))

# calculs totaux
print("----------------------------------------------------")
print("nombre de femmes en 2019 : ", 3864 + 4629 + 396)
print("nombre d'hommes en 2019 : ", 8378 + 16972 + 2173)
print("total athlètes : ", 3864 + 4629 + 396 + 8378 + 16972 + 2173)

# répartition entre les continents
print("----------------------------------------------------")
continent_count = df.groupby(['country']).size().sort_values(ascending=False)
print("Répartition par pays : ")
print(continent_count / 365)
