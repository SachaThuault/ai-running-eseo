import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_parquet('../../data/run_ww_2019_d.parquet')
df2 = pd.read_parquet('../../data/run_ww_2020_d.parquet')
#
# # Assurez-vous d'avoir la colonne 'age_group' au format de catégorie pour garantir un ordre correct sur l'axe y
# df['age_group'] = df['age_group'].astype('category')
#
# # Définir le style de seaborn pour des graphiques plus agréables
# sns.set(style="whitegrid")
#
# # Créer un diagramme de dispersion avec age_group et datetime
# plt.figure(figsize=(12, 8))
# sns.scatterplot(x='gender', y='age_group', data=df, hue='age_group', palette='viridis', s=100)
#
# # Ajouter des titres et des légendes
# plt.title('Clusters en fonction de age_group et gender')
# plt.xlabel('gender')
# plt.ylabel('Age Group')
# plt.legend(title='Age Group')
#
# # Afficher le graphique
# plt.show()





# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# X = df[['duration', 'datetime']].copy()
#
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
#
# plt.plot(range(1, 11), wcss)
# plt.title('Selecting the Numbeer of Clusters using the Elbow Method')
# plt.xlabel('Clusters')
# plt.ylabel('WCSS')
# plt.show()
#
#
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Assurez-vous que la colonne 'datetime' est au format datetime
# df['datetime'] = pd.to_datetime(df['datetime'])
#
# # Convertissez la colonne 'datetime' en timestamp Unix (représentation numérique)
# df['timestamp'] = (df['datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#
# # Créez un nouvel ensemble de données avec les colonnes 'duration' et 'timestamp'
# X = df[['duration', 'timestamp']].copy()
#
# # Initialisez une liste pour stocker les valeurs de WCSS
# wcss = []
#
# # Exécutez K-means pour différentes valeurs de clusters
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
#
# # Tracez le coude (elbow) du graphe
# plt.plot(range(1, 11), wcss)
# plt.title('Selecting the Number of Clusters using the Elbow Method')
# plt.xlabel('Clusters')
# plt.ylabel('WCSS')
# plt.show()



# --------- Analyse pour un seul coureur


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

coureur_specifique = df.loc[df.athlete == 37594]
coureur_specifique2 = df2.loc[df2.athlete == 37594]

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

for index, row in coureur_specifique2.iterrows():
    datetimes.append(row['datetime'])
    athletes.append(row['athlete'])
    distances.append(row['distance'])
    durations.append(row['duration'])
    genders.append(row['gender'])
    age_groups.append(row['age_group'])
    countries.append(row['country'])
    majors.append(row['major'])

import matplotlib.pyplot as plt

# Assurez-vous d'avoir la colonne 'age_group' au format de catégorie pour garantir un ordre correct sur l'axe y
# df['age_group'] = df['age_group'].astype('category')

# Créer un diagramme de dispersion avec age_group et datetime
plt.figure(figsize=(12, 8))

# Utiliser une boucle pour créer un point pour chaque catégorie d'âge
# for datetime, group_df in df.loc[df.distance]:
#     plt.scatter(df.datetime, df.distance, label=df.loc[df.datetime], s=100)
#
# for age_group, group_df in df.groupby('age_group'):
#     plt.scatter(group_df['datetime'], group_df['age_group'], label=age_group, s=100)

plt.scatter(datetimes, distances, label="datetime / distance", s=100)


 # Ajouter des titres et des légendes
plt.title('Analyse entrainement en fonction de datetime et distance sur 2 ans')
plt.xlabel('datetime')
plt.ylabel('distance')
plt.legend(title='Analyse entrainement en fonction de datetime et distance sur 2 ans')

 # Afficher le graphique
plt.show()

plt.scatter(durations, distances, label="duration / distance", s=100)


 # Ajouter des titres et des légendes
plt.title('Analyse entrainement en fonction de duration et distance sur 2 ans')
plt.xlabel('duration')
plt.ylabel('distance')
plt.legend(title='Analyse entrainement en fonction de duration et distance sur 2 ans')

 # Afficher le graphique
plt.show()
