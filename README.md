# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: Pandas, NumPy, and DecisionTreeRegressor from scikit-learn.

2.Load the weather dataset and select input features such as humidity, pressure, wind speed, illumination, and CO₂.

3.Handle missing values in input features and target variables using mean values.

4.Split the dataset into training and testing sets for Pollution, Temperature, and Energy prediction.

5.Create and train separate Decision Tree Regressor models for PM2.5, temperature, and TSR prediction.

6.Predict test results, calculate RMSE and R² score, and display prediction accuracy for all models.

## Program:

Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.

Developed by: Harish P

RegisterNumber:  212225040115
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("weather.csv")

# =========================
# 1. Pollution Prediction (PM2.5)
# =========================

x = data[['hum', 'pressure', 'wind_speed', 'illumination', 'co2']]

x = x.fillna(x.mean())
y = data['pm2_5'].fillna(data['pm2_5'].mean())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42, max_depth=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

rmse_pollution = np.sqrt(mean_squared_error(y_test, y_pred))
r2_pollution = r2_score(y_test, y_pred)
accuracy_pollution = r2_pollution * 100

print("🏭 Pollution Prediction (PM2.5)")
print("Accuracy (%):", accuracy_pollution)




print("RMSE:", rmse_pollution)
print("R2 Score:", r2_pollution)


# =========================
# 2. Temperature Prediction
# =========================

ytemp = data['tem'].fillna(data['tem'].mean())

x_train, x_test, y_train, y_test = train_test_split(x, ytemp, test_size=0.2, random_state=42)

model1 = DecisionTreeRegressor(random_state=42, max_depth=5)
model1.fit(x_train, y_train)

temp_pred = model1.predict(x_test)

rmse_temp = np.sqrt(mean_squared_error(y_test, temp_pred))
r2_temp = r2_score(y_test, temp_pred)
accuracy_temp = r2_temp * 100
print("\n🌡️ Temperature Prediction")
print("Accuracy (%):", accuracy_temp)
print("RMSE:", rmse_temp)
print("R2 Score:", r2_temp)


# =========================
# 3. Energy Prediction (TSR)
# =========================

yenergy = data['tsr'].fillna(data['tsr'].mean())

x_train, x_test, y_train, y_test = train_test_split(x, yenergy, test_size=0.2, random_state=42)

model2 = DecisionTreeRegressor(random_state=42, max_depth=5)
model2.fit(x_train, y_train)

energy_pred = model2.predict(x_test)

rmse_energy = np.sqrt(mean_squared_error(y_test, energy_pred))
r2_energy = r2_score(y_test, energy_pred)
accuracy_energy = r2_energy * 100
print("\n⚡ Energy Prediction (TSR)")
print("Accuracy (%):", accuracy_energy)


print("RMSE:", rmse_energy)
print("R2 Score:", r2_energy)
```

## Output:
<img width="478" height="312" alt="image" src="https://github.com/user-attachments/assets/67bfff14-8ec9-4e97-af46-a70d984e644e" />


## Result:

Thus, a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm is written and verified using Python Programming.
