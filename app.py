from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and label mappings
model = joblib.load("model.pkl")
mappings = joblib.load("label_mappings.pkl")

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
        # Encode categorical values
        district = mappings["District Name"][data["district"]]
        market = mappings["Market Name"][data["market"]]
        commodity = mappings["Commodity"][data["commodity"]]
        season = mappings["Season"][data["season"]]

        # Parse and extract date features
        date_str = data["date"]
        date = datetime.strptime(date_str, "%Y-%m-%d")
        year = date.year
        month = date.month
        day = date.day
        day_of_week = date.weekday()      # 0 = Monday, 6 = Sunday
        week_of_year = int(date.strftime("%U"))

        # Prepare input for prediction
        features = [[district, market, commodity, season, year, month, day, day_of_week, week_of_year]]
        prediction = model.predict(features)

        return jsonify({"predicted_price": float(round(prediction[0], 2))})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
