from flask import Flask, request, Response
from flask_cors import CORS
from flask import jsonify
import requests
import json
import backend
import cv2
import numpy as np
import base64
from io import BytesIO

class Server:
    def __init__(self, url):
        self.url = url
        self.app = Flask(__name__)
        CORS(self.app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for localhost:3000
        self.app.add_url_rule("/", "home", self.hello_world, methods=["GET"])
        self.app.add_url_rule("/detect", "detect", self.detect, methods=["POST", "GET"])
        self.app.add_url_rule("/test", "test", self.test, methods=["GET"])

    def hello_world(self):
        if request.method == "POST":
            print("hello world")
        return "<p>Hello, World!</p>"

    def detect(self):
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


    def run(self):
        try:
            self.app.run(host="0.0.0.0",  # Allows access from any network interface
                         port=int(self.url.split(":")[-1][:-1:]),
                         threaded=True,
                         debug=True)
            print("\033[0;35m" + f"\nlisten(GET): {url}" + "\n\033[0m")
        except OSError:
            print("bruh moment")
            exit()

    def test(self):
        testPath = "./data/die.jpg"
        with open(testPath, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode('utf-8')
        
        print("Data : ",data)

        r = requests.post("http://127.0.0.1:3000/detect", json={'image': data})

        if r.status_code == 200:
            return "<p>server and backend is working properly</p>"
        else:
            return "<p>check the backend</p>"

if __name__ == "__main__":
    url = "http://localhost:3000/"
    server = Server(url)
    server.run()
    print("server is closed")
