# 📌 Step 1: Upload File
from google.colab import files
uploaded = files.upload()

# 📌 Step 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

# 📌 Step 3: Load Dataset
df = pd.read_csv(list(uploaded.keys())[0])

# Ubah ke datetime format, jika error → jadi NaT
df['pickup_time'] = pd.to_datetime(df['pickup_time'], errors='coerce')
df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')

# Hapus baris yang error parsing datetime
df = df.dropna(subset=['pickup_time', 'delivery_time'])

# 📌 Step 4: Feature Engineering
df['delivery_duration'] = (df['delivery_time'] - df['pickup_time']).dt.total_seconds() / 3600
df['day_of_week'] = df['pickup_time'].dt.dayofweek
df['hour_of_day'] = df['pickup_time'].dt.hour

# Drop kolom yang tidak digunakan untuk modeling
df = df.drop(columns=['shipment_id', 'pickup_time', 'delivery_time'])

# Encode kolom kategorikal
df = pd.get_dummies(df, columns=['origin', 'destination'], drop_first=True)

# 📌 Step 5: Define Features & Target
X = df.drop(columns=['delivery_duration'])
y = df['delivery_duration']

# 📌 Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 7: Train XGBoost Model
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# 📌 Step 8: Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE: {mae:.2f} jam")
print(f"📊 RMSE: {rmse:.2f} jam")
print(f"📊 R² Score: {r2:.2f}")

# 📌 Step 9: Feature Importance
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=10)
plt.title("Top 10 Feature Importance (XGBoost)")
plt.show()