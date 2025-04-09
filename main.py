from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load precomputed image embeddings and routes
image_embeddings = None
routes = None
try:
    with open("image_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        image_embeddings = data["embeddings"]
        # Correct shape from (n, 1, 512) to (n, 512)
        if image_embeddings.ndim == 3 and image_embeddings.shape[1] == 1:
            image_embeddings = image_embeddings.squeeze(1)
        elif image_embeddings.ndim != 2:
            raise ValueError(f"Invalid shape for image_embeddings: {image_embeddings.shape}")
        routes = data["routes"]
    logging.info(f"Successfully loaded image_embeddings.pkl with shape: {image_embeddings.shape}")
except FileNotFoundError:
    logging.error("Error: image_embeddings.pkl not found.")
    raise
except pickle.UnpicklingError as e:
    logging.error(f"Error unpickling image_embeddings.pkl: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error loading image_embeddings.pkl: {e}")
    raise

# Load precomputed uploaded embedding (for testing)
uploaded_embedding = None
try:
    with open("uploaded_embedding.pkl", "rb") as f:
        uploaded_embedding = pickle.load(f)
        # Ensure 2D shape (1, 512)
        if uploaded_embedding.ndim == 1:
            uploaded_embedding = uploaded_embedding.reshape(1, -1)
        elif uploaded_embedding.ndim == 3 and uploaded_embedding.shape[1] == 1:
            uploaded_embedding = uploaded_embedding.squeeze(1)
        elif uploaded_embedding.shape != (1, 512):
            raise ValueError(f"Invalid shape for uploaded_embedding: {uploaded_embedding.shape}")
    logging.info(f"Successfully loaded uploaded_embedding.pkl with shape: {uploaded_embedding.shape}")
except FileNotFoundError:
    logging.warning("Warning: uploaded_embedding.pkl not found. Using default.")
    uploaded_embedding = np.zeros((1, 512))  # Default embedding if file is missing
except pickle.UnpicklingError as e:
    logging.error(f"Error unpickling uploaded_embedding.pkl: {e}. Using default.")
    uploaded_embedding = np.zeros((1, 512))  # Fallback on unpickling error
except Exception as e:
    logging.error(f"Unexpected error loading uploaded_embedding.pkl: {e}. Using default.")
    uploaded_embedding = np.zeros((1, 512))  # Generic fallback

pattern_responses = {
    "help": "Sure! I can help you. Ask about products, return policy, or upload jewelry image!",
    "return policy": "You can return items within 15 days with original packaging.",
    "bye": "Goodbye! Feel free to come back anytime!",
}

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").lower().strip()
    reply = pattern_responses.get(message, "Try asking for help or upload an image!")
    return jsonify({"reply": reply})

@app.route("/api/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    if file and uploaded_embedding is not None and image_embeddings is not None:
        # Save the uploaded file (for reference, though not used for embedding here)
        filepath = os.path.join('/tmp', file.filename)
        file.save(filepath)
        os.remove(filepath)

        # Use precomputed uploaded_embedding (for testing)
        logging.info(f"uploaded_embedding shape: {uploaded_embedding.shape}, image_embeddings shape: {image_embeddings.shape}")
        similarity = cosine_similarity([uploaded_embedding], image_embeddings)
        best_match_idx = np.argmax(similarity)
        best_route = routes[best_match_idx] if similarity[0][best_match_idx] > 0.7 else None

        if best_route:
            return jsonify({"reply": "Found a matching product!", "route": best_route})
        return jsonify({"reply": "No matching product found.", "route": ""})

    return jsonify({"error": "Invalid file or embeddings not loaded"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4986)
