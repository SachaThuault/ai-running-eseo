import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

start_time = time.time()

df = pd.read_parquet('./data/run_ww_2020_d.parquet')

distance_moy = df.distance.mean()
duration_moy = df.duration.mean()
print(f"average speed km/h = {(distance_moy / duration_moy) * 60}")

selected_features = ['athlete', 'distance', 'age_group']
X = df[selected_features]
y = df['duration']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('age_group', OneHotEncoder(), ['age_group'])
    ],
    remainder='passthrough'
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

scaler = StandardScaler()
X_train_preprocessed = scaler.fit_transform(X_train_preprocessed)
X_val_preprocessed = scaler.transform(X_val_preprocessed)
X_test_preprocessed = scaler.transform(X_test_preprocessed)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train_preprocessed.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_preprocessed, y_train, epochs=15, batch_size=32, validation_data=(X_val_preprocessed, y_val), verbose=2)

y_test_pred_nn = model.predict(X_test_preprocessed)

mse_nn = mean_squared_error(y_test, y_test_pred_nn)
r2_nn = r2_score(y_test, y_test_pred_nn)
print(f'Mean Squared Error (MSE) on test set: {mse_nn}')
print(f'R-squared (R2) on test set: {r2_nn}')

plt.scatter(X_test['distance'], y_test, color='black', label='True values (Neural Network)')
plt.scatter(X_test['distance'], y_test_pred_nn, color='purple', linewidth=1, label='Predicted values (Neural Network)')
plt.xlabel('Distance (km)')
plt.ylabel('Duration (minute)')
plt.legend()
plt.show()

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time : {execution_time} secondes")
