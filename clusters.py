import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Charger le dataset depuis le fichier CSV
file_path = 'dataset_v1.csv'
dataset = pd.read_csv(file_path, delimiter=';')

# Renommer les colonnes pour éviter les problèmes avec les parenthèses
dataset = dataset.rename(columns={
    'distance_(m)': 'distance',
    'elapsed_time_(s)': 'elapsed_time',
    'elevation_gain_(m)': 'elevation_gain',
    'average_heart_rate_(bpm)': 'average_heart_rate'
})

# Sélectionner les colonnes pertinentes sans utiliser selected_columns
data = dataset[['distance', 'elapsed_time', 'elevation_gain', 'average_heart_rate']]

# Normalisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Application de K-means avec, par exemple, 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
dataset['cluster'] = kmeans.fit_predict(data_scaled)

# Afficher les résultats
print(dataset[['athlete', 'cluster']])
