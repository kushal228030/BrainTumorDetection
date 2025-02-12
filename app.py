from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "brain_tumor.h5"  # Update the path if needed
model = keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize((299, 299))  # Resize to match model input size
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Route to predict tumor type (POST)
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(image)

        # Make prediction
        prediction = model.predict(img_array)
        confidences = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

        # Sort and return results
        sorted_confidences = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True))
        return jsonify({"predictions": sorted_confidences})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get class names (GET)
@app.route("/class_names", methods=["GET"])
def get_class_names():
    return jsonify({"class_names": class_names})

# Route to check server status (GET)
@app.route("/status", methods=["GET"])
def server_status():
    return jsonify({"status": "Server is running!"})

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Brain Tumor Classification API is running!"})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
