import numpy as np
from sklearn.linear_model import LinearRegression

# House size in square feet
X = np.array([[500], [800], [1000], [1200], [1500]])

# House prices
y = np.array([1500000, 2000000, 2500000, 3000000, 3500000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for 1100 sq ft house
prediction = model.predict([[1100]])

print("Predicted House Price:", prediction)
