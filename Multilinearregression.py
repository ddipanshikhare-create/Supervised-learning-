from sklearn.linear_model import LinearRegression
import numpy as np

# Input data (size, rooms)
X = np.array([[1000,2],
              [1500,3],
              [2000,4],
              [2500,5]])

# Output (price)
y = np.array([200000,300000,400000,500000])

# Model
model = LinearRegression()

# Train model
model.fit(X,y)

# Predict price
pred = model.predict([[1800,3]])

print("Predicted Price:", pred)
