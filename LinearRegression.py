import numpy as np
from sklearn.linear_model import LinearRegression

# Input data (hours studied)
X = np.array([[1], [2], [3], [4], [5]])

# Output data (marks)
y = np.array([40, 50, 60, 70, 80])

# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict marks for 6 hours of study
prediction = model.predict([[6]])

print("Predicted Marks:", prediction)
