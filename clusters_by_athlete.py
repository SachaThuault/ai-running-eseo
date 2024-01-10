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

grouped_data = dataset.groupby('athlete').agg({
    'distance': 'mean',
    'elapsed_time': 'mean',
    'elevation_gain': 'mean',
    'average_heart_rate': 'mean'
}).reset_index()

data = grouped_data[['distance', 'elapsed_time', 'elevation_gain', 'average_heart_rate']]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
grouped_data['cluster'] = kmeans.fit_predict(data_scaled)

print(grouped_data[['athlete', 'cluster']])
