import pandas as pd

# read daily
df = pd.read_parquet('../../data/run_ww_2019_d.parquet')
df2 = pd.read_parquet('../../data/run_ww_2020_d.parquet')

print("----------------COLONNE ATHLETE 2019----------------")
print(df.athlete)
print("----------------COLONNE ATHLETE 2020----------------")
print(df2.athlete)

print("----------------------------------------------------")

gender_counts = df.groupby(['age_group', 'gender']).size().unstack()
print("répartition des genres en 2019 : ")
print(gender_counts)
print("----------------------------------------------------")

gender_counts2 = df2.groupby(['age_group', 'gender']).size().unstack()
print("répartition des genres en 2020 : ")
print(gender_counts2)
print("----------------------------------------------------")


# répartition entre les continents
#
