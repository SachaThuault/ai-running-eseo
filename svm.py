import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import time

df = pd.read_parquet('./data/run_ww_2019_m.parquet')
df = df.sample(frac=0.10, random_state=42)
start_time = time.time()
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

svm_model = make_pipeline(preprocessor, SVR(kernel='linear'))

svm_model.fit(X_train, y_train)

y_val_pred_svm = svm_model.predict(X_val)

mse_svm = mean_squared_error(y_val, y_val_pred_svm)
r2_svm = r2_score(y_val, y_val_pred_svm)
print(f'Mean Squared Error (MSE) on validation set: {mse_svm}')
print(f'R-squared (R2) on validation set: {r2_svm}')

# Enregistrez le temps de fin
end_time = time.time()

execution_time = end_time - start_time

print(f"Temps d'exécution total : {execution_time} secondes")
# Visualiser les résultats pour SVM
plt.scatter(X_test['distance'], y_test, color='black', label='True values (SVM)')
plt.scatter(X_test['distance'], svm_model.predict(X_test), color='purple', linewidth=1,
            label=f'Predicted values (SVM)')
plt.xlabel('Distance (km)')
plt.ylabel('Duration (minute)')
plt.legend()
plt.show()
