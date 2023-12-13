import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Charger les données
df = pd.read_parquet('../data/run_ww_2019_m.parquet')

# Sélectionner les colonnes pertinentes
selected_features = ['athlete', 'duration']
X = df[selected_features]
y = df['distance']

# Diviser les données en ensembles d'entraînement (train), de validation (validation) et de test (test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


for degree in range(1, 10):
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de validation
    y_val_pred_poly = polyreg.predict(X_val)

    # Évaluer les performances sur l'ensemble de validation
    mse = mean_squared_error(y_val, y_val_pred_poly)
    r2 = r2_score(y_val, y_val_pred_poly)
    print(f'---------------DEGREE {degree}------------------')
    print(f'Mean Squared Error (MSE) on validation set: {mse}')
    print(f'R-squared (R2) on validation set: {r2}')


    # Visualiser les résultats pour la régression polynomiale
    plt.scatter(X_test['duration'], y_test, color='black', label='True values (Polynomial Regression)')
    plt.scatter(X_test['duration'], polyreg.predict(X_test), color='blue', linewidth=1, label=f'Predicted values (Polynomial Regression) deg: {degree}')
    plt.xlabel('Duration')
    plt.ylabel('Distance')
    plt.legend()
    plt.show()

