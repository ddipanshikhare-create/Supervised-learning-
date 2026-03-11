from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Input data (hours studied)
X = np.array([[1],[2],[3],[4],[5]])

# Output data (marks)
y = np.array([40,50,60,70,80])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict result
prediction = model.predict([[6]])

print("Predicted Marks:", prediction)
