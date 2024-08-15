from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import backend
import cv2
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for localhost:3000

@app.route("/", methods=["GET"])
def hello_world():
    if request.method == "POST":
        print("hello world")
    return "<p>Hello, World!</p>"

@app.route("/detect", methods=["POST", "GET"])
def detect():
    if request.method == "POST":
        try:
            print(request.json)  # Log the incoming JSON to inspect it
            data = request.json["image"]
            print("Base64 data received.")
        except KeyError:
            return jsonify({"success": False, "error": "Field 'image' does not exist"}), 400

        try:
            file_bytes = np.asarray(bytearray(BytesIO(base64.b64decode(data)).read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image decoding failed. The image data may be corrupt.")
            print("Image decoded successfully.")
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid data: {str(e)}"}), 400

        try:
            output = backend.run_process(img)
            print("Processing completed successfully.")
        except Exception as e:
            print("Error:", e)
            return jsonify({"success": False, "error": str(e)}), 400

        # Return success response with output data
        return jsonify({"success": True, "output": output}), 200

    else:
        return "<p>Hello, Detector!</p>"

@app.route("/test", methods=["GET"])
def test():
    testPath = "./data/die.jpg"
    with open(testPath, "rb") as image_file:
        data = base64.b64encode(image_file.read()).decode('utf-8')

    print("Data : ", data)

    r = requests.post("http://127.0.0.1:3000/detect", json={'image': data})

    if r.status_code == 200:
        return "<p>server and backend is working properly</p>"
    else:
        return "<p>check the backend</p>"

# Vercel serverless functions need to expose the application
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
