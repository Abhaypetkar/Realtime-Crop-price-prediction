import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load Excel
df = pd.read_excel("AllCrops.xlsx", sheet_name="Sheet1")
df = df.dropna(subset=["Date", "District Name", "Market Name", "Commodity", "Season", "Modal Price"])

# Label encode required columns
columns_to_encode = ["District Name", "Market Name", "Commodity", "Season"]
mappings = {}
for col in columns_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Features and target
X = df[["District Name", "Market Name", "Commodity", "Season"]]
y = df["Modal Price"]

# Model
model = RandomForestRegressor(n_estimators=50)
model.fit(X, y)

# Export both
joblib.dump(model, "model.pkl")
joblib.dump(mappings, "label_mappings.pkl")

print("✅ model.pkl and label_mappings.pkl exported.")
