from flask import Flask, request, jsonify, render_template
from chatbot import get_response, predict_disease
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    bot_reply = get_response(user_message)
    return jsonify({"response": bot_reply})

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    lang = request.form.get("lang", "en")
    if not file:
        return jsonify({"response": "No file uploaded."})

    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    prediction = predict_disease(filepath, lang)
    return jsonify({"response": prediction})

if __name__ == "__main__":
    app.run(debug=True)
