from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 50, 65, 75, 90])

model = LinearRegression()

#training
model.fit(X, y)

# predicting
x_new = np.array([[6]])
predicted_score = model.predict(x_new)
print("Predicted score for 6 hours study:", predicted_score[0])

plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(6, predicted_score, color='green', label='Predicted Point')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.title("Linear Regression Example")
plt.show()
