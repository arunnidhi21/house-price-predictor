# House Price Predictor
# Uses Linear Regression to predict house prices based on features

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset (you can replace with a CSV)
data = {
    'area_sqft': [500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500],
    'bedrooms':  [1,   2,    2,    3,    3,    4,    4,    5,    5,    6],
    'age_years': [10,  5,    8,    3,    6,    2,    1,    4,    7,    2],
    'price_lakh':[20,  35,   45,   55,   70,   90,   100,  130,  160,  200]
}

df = pd.DataFrame(data)

X = df[['area_sqft', 'bedrooms', 'age_years']]
y = df['price_lakh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== House Price Predictor ===")
print(f"R² Score     : {r2_score(y_test, y_pred):.2f}")
print(f"RMSE         : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} lakh")

# Predict a new house
new_house = pd.DataFrame({'area_sqft': [1600], 'bedrooms': [3], 'age_years': [4]})
predicted = model.predict(new_house)[0]
print(f"\nPredicted price for 1600 sqft, 3 BHK, 4yr old: ₹{predicted:.2f} lakh")

# Plot
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("house_price_plot.png")
plt.show()
