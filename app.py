from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
mappings = joblib.load("label_mappings.pkl")

@app.route("/")
def home():
    return "Agrokalp AI API is running."

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

        prediction = model.predict([[district, market, commodity, season]])
        return jsonify({"predicted_price": round(prediction[0], 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
