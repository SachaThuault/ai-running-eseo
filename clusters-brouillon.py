import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = 'dataset_v1.csv'
dataset = pd.read_csv(file_path, delimiter=';')

dataset = dataset.rename(columns={
    'distance_(m)': 'distance',
    'elapsed_time_(s)': 'elapsed_time',
    'elevation_gain_(m)': 'elevation_gain',
    'average_heart_rate_(bpm)': 'average_heart_rate'
})

data = dataset[['distance', 'elapsed_time', 'elevation_gain', 'average_heart_rate']]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
dataset['cluster'] = kmeans.fit_predict(data_scaled)

print(dataset[['athlete', 'cluster']])
