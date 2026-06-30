import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend.predictor import FaceMaskPredictor

app = Flask(
    __name__,
    static_folder="../frontend",
    static_url_path=""
)

CORS(app)

print("Loading CNN Model...")
predictor = FaceMaskPredictor()
print("CNN Loaded Successfully!")


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(app.static_folder, "style.css")


@app.route("/script.js")
def script():
    return send_from_directory(app.static_folder, "script.js")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error":"No image"}),400

    file=request.files["image"]

    temp = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".jpg"
    )

    temp_path = temp.name

    temp.close()   # <-- This closes the file

    file.save(temp_path)

    result = predictor.predict(temp_path)

    os.remove(temp_path)

    return jsonify(result)


if __name__ == "__main__":

    print("====================================")
    print("Scratch CNN Server Started")
    print("http://127.0.0.1:5000")
    print("====================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )