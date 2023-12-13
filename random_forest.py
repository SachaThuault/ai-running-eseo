import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Charger les données
df = pd.read_parquet('../data/run_ww_2019_m.parquet')

# Calculer la moyenne
distance_moy = df.distance.mean()
duration_moy = df.duration.mean()
print(f"average speed km/h = {(distance_moy / duration_moy) * 60}")

# Sélectionner les colonnes pertinentes
selected_features = ['athlete', 'duration', 'age_group']
X = df[selected_features]
y = df['distance']

# Diviser les données en ensembles d'entraînement (train), de validation (validation) et de test (test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Créer un préprocesseur pour gérer les variables catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('age_group', OneHotEncoder(), ['age_group'])
    ],
    remainder='passthrough'
)


# Créer le pipeline avec le préprocesseur et le modèle de forêt aléatoire
rf_model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100, random_state=42))

# Adapter le modèle
rf_model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de validation
y_val_pred_rf = rf_model.predict(X_val)

# Évaluer les performances sur l'ensemble de validation
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
r2_rf = r2_score(y_val, y_val_pred_rf)
print(f'Mean Squared Error (MSE) on validation set: {mse_rf}')
print(f'R-squared (R2) on validation set: {r2_rf}')

# Visualiser les résultats pour la forêt aléatoire
plt.scatter(X_test['duration'], y_test, color='black', label='True values (Random Forest)')
plt.scatter(X_test['duration'], rf_model.predict(X_test), color='green', linewidth=1,
            label=f'Predicted values (Random Forest)')
plt.xlabel('Duration')
plt.ylabel('Distance')
plt.legend()
plt.show()
