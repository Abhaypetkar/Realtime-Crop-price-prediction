from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and mappings
model = joblib.load("model.pkl")
mappings = joblib.load("label_mappings.pkl")

# Load dataset for historical dashboard
df = pd.read_excel("AllCrops.xlsx", sheet_name="Sheet1")
df = df.dropna(subset=["Date", "District Name", "Market Name", "Commodity", "Modal Price"])
df["Date"] = pd.to_datetime(df["Date"])

@app.route("/")
def home():
    return "✅ Agrokalp AI API is running."

@app.route("/mappings")
def get_mappings():
    label_lists = {key: list(val.keys()) for key, val in mappings.items()}
    return jsonify(label_lists)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        district = mappings["District Name"][data["district"]]
        market = mappings["Market Name"][data["market"]]
        commodity = mappings["Commodity"][data["commodity"]]
        season = mappings["Season"][data["season"]]

        date_str = data["date"]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        year = date.year
        month = date.month
        day = date.day
        day_of_week = date.weekday()
        week_of_year = int(date.strftime("%U"))

        features = [[district, market, commodity, season, year, month, day, day_of_week, week_of_year]]
        prediction = model.predict(features)

        return jsonify({"predicted_price": float(round(prediction[0], 2))})
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Historical Price Dashboard APIs

@app.route("/options")
def get_options():
    return jsonify({
        "commodities": sorted(df["Commodity"].unique()),
        "districts": sorted(df["District Name"].unique()),
        "markets": sorted(df["Market Name"].unique())
    })

@app.route("/historical-prices", methods=["POST"])
def historical_prices():
    data = request.json
    commodity = data.get("commodity")
    district = data.get("district")
    market = data.get("market")

    filtered = df.copy()
    if commodity:
        filtered = filtered[filtered["Commodity"].str.lower() == commodity.lower()]
    if district:
        filtered = filtered[filtered["District Name"].str.lower() == district.lower()]
    if market:
        filtered = filtered[filtered["Market Name"].str.lower() == market.lower()]

    grouped = (
        filtered.groupby("Date")["Modal Price"]
        .mean()
        .reset_index()
        .sort_values("Date")
    )
    grouped["Modal Price"] = grouped["Modal Price"].astype(float)

    return jsonify(grouped.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
