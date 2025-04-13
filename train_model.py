import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np

# Load Excel
df = pd.read_excel("AllCrops.xlsx", sheet_name="Sheet1")
df = df.dropna(subset=["Date", "District Name", "Market Name", "Commodity", "Season", "Modal Price"])

# Parse date and extract features
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# Label Encoding
columns_to_encode = ["District Name", "Market Name", "Commodity", "Season"]
mappings = {}
for col in columns_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Features and target
features = ["District Name", "Market Name", "Commodity", "Season", "Year", "Month", "Day", "DayOfWeek", "WeekOfYear"]
X = df[features]
y = df["Modal Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Export model and label mappings
joblib.dump(model, "model.pkl")
joblib.dump(mappings, "label_mappings.pkl")

print(f"✅ Model trained and saved.")
print(f"📈 RMSE: {rmse:.2f}")
print(f"📊 R2 Score: {r2:.2f}")
