import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.linear_model import Ridge

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

for degree in range(1, 6):
    start_time = time.time()
    polyreg = make_pipeline(preprocessor, PolynomialFeatures(degree), Ridge(alpha=1.0))

    polyreg.fit(X_train, y_train)

    y_val_pred_poly = polyreg.predict(X_val)

    mse = mean_squared_error(y_val, y_val_pred_poly)
    r2 = r2_score(y_val, y_val_pred_poly)
    print(f'---------------DEGREE {degree}------------------')
    print(f'Mean Squared Error (MSE) on validation set: {mse}')
    print(f'R-squared (R2) on validation set: {r2}')

    plt.scatter(X_test['distance'], y_test, color='black', label='True values (Polynomial Regression)')
    plt.scatter(X_test['distance'], polyreg.predict(X_test), color='blue', linewidth=1,
                label=f'Predicted values (Polynomial Regression) deg: {degree}')
    plt.xlabel('Distance (km)')
    plt.ylabel('Duration (minute)')
    plt.legend()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time}')
    plt.show()

print(f"Temps d'exécution total : {execution_time} secondes")
