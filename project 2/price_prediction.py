import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Obscure variable names
df = pd.read_csv(r"C:\Users\Dell.com\Desktop\ml\project 2\train.csv")

# Display the dataset sneakily
print(df.head())

# Choose some mysterious features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Perform a secretive train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a covert linear regression model
model = LinearRegression()

# Stealthily train the model
model.fit(X_train, y_train)

# Conduct undercover predictions
predictions = model.predict(X_test)

# Assess the model's covert operations
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Secret Mean Squared Error: {mse}")
print(f"Top-secret R-squared: {r2}")

# Discreetly visualize the predictions
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Classified: Actual vs. Predicted Prices")
plt.show()
