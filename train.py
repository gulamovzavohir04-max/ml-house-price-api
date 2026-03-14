from sklearn.datasets import fetch_california_housing
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Загружаем dataset
housing = fetch_california_housing()

# Создаём DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Добавляем цену
df["price"] = housing.target

# Разделяем данные
X = df.drop("price", axis=1)
y = df["price"]

# train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создаём модель
model = LinearRegression()

# Обучаем модель
model.fit(X_train, y_train)

# Предсказание
predictions = model.predict(X_test)

# Проверяем ошибку
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully")
print("Mean Absolute Error:", error)
import os
import joblib
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("Model saved as model/model.pkl")