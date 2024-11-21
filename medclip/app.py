from flask import Flask, request, jsonify
import os
import torch
from PIL import Image
from datastore_retrieval import get_retrieved_info_for_image

app = Flask(__name__)

# Define default paths (can be set through environment variables if needed)
DEFAULT_INDEX_PATH = os.getenv("INDEX_PATH", "/workspace/datastore/kg_nle_index")
DEFAULT_CAPTIONS_PATH = os.getenv("CAPTIONS_PATH", "/workspace/datastore/kg_nle_index_captions.json")
TEMP_IMAGE_PATH = "/workspace/temp_uploaded_image.png"  # Temporary file storage path inside the container

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400

        file = request.files['image']

        # Ensure the file is a .png file
        if file.filename == '' or not file.filename.lower().endswith('.jpg'):
            return jsonify({"error": "Only .jpg files are allowed."}), 400

        # Save the uploaded image temporarily for processing
        file.save(TEMP_IMAGE_PATH)

        # Update paths dynamically or use default ones
        index_path = os.getenv("INDEX_PATH", DEFAULT_INDEX_PATH)
        captions_path = os.getenv("CAPTIONS_PATH", DEFAULT_CAPTIONS_PATH)

        # Assuming you have the model and feature extractor set up
        retrieved_info = get_retrieved_info_for_image(
            image_path=TEMP_IMAGE_PATH,
            index_path=index_path,
            captions_path=captions_path,
            k=7  # Number of neighbors to retrieve
        )

        # Example: Return retrieved captions and other relevant information
        return jsonify(retrieved_info)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
