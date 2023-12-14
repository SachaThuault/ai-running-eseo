import pandas as pd

# read daily
df = pd.read_parquet('../../data/run_ww_2019_d.parquet')
df2 = pd.read_parquet('../../data/run_ww_2020_d.parquet')
count=0
count2=0

print("1-------------------------------------")
print(f"Nom des colonnes : {df.columns}")
print(f"Nom des colonnes df2 : {df2.columns}")
# print("2-------------------------------------")
# print(f"Nombre de lignes: {len(df)}")
# print("3-------------------------------------")
# print(df.distance.describe())
# print("4-------------------------------------")
# print(df.athlete)
# print("5-------------------------------------")
# print(len(df.loc[df.athlete == 37594]))
# print(df.loc[df.athlete == 37594])
# print("6-------------------------------------")
# print(df.loc[df.distance > 42].duration)
# print("7-------------------------------------")
# print(df.loc[df.distance == 42].duration.describe())
print("7-------------------------------------")
print(df.loc[df.age_group == '55 +'])
# print(df.age_group)
# if df.age_group == '55+':
#     print(df.athlete)

# for train in df:
#     # print("train : " + train.distance)
#     print("train : " + train)
#     if train.loc[train.athlete == 37594]:
#         if train.loc[df.distance == 0]:
#                 count += 1



# Créer des listes vides pour chaque colonne
datetimes = []
athletes = []
distances = []
durations = []
genders = []
age_groups = []
countries = []
majors = []

# Filtrer les données pour un coureur spécifique (par exemple, athlete = 37594)

# coureur_specifique = df.loc[df.athlete == 215456] athlete vide ?
# coureur_specifique = df.loc[df.athlete == 21547] 2000
# coureur_specifique = df.loc[df.athlete == 21549] 2500
# coureur_specifique = df.loc[df.athlete == 21550] 500 km / an
# coureur_specifique = df.loc[df.athlete == 21551] 500 35 - 54
coureur_specifique = df.loc[df.athlete == 10] # 6 10 26 28 35 13326635 13326665 13326676 13326695 13326700

# Parcourir les lignes résultantes et ajouter les valeurs aux listes correspondantes
for index, row in coureur_specifique.iterrows():
    datetimes.append(row['datetime'])
    athletes.append(row['athlete'])
    distances.append(row['distance'])
    durations.append(row['duration'])
    genders.append(row['gender'])
    age_groups.append(row['age_group'])
    countries.append(row['country'])
    majors.append(row['major'])

# Maintenant, vous avez chaque colonne dans une liste
# Vous pouvez accéder aux données spécifiques du coureur à partir de ces listes
for i in distances:
    if i == 0:
        count += 1


print("jours de repos dans l'année : " + str(count))
print("moyenne de nombre de jour par semaine : " + str(count/54))
print("gender : " + str(genders))
print("age_groups : " + str(age_groups))
print("countries : " + str(countries))
print("majors : " + str(majors))
print("volume total : " + str(sum(distances)))


















