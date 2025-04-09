from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load precomputed image embeddings and routes
with open("image_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    image_embeddings = data["embeddings"]  # 2D NumPy array
    routes = data["routes"]  # List of routes

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").lower().strip()
    pattern_responses = {
        "help": "Sure! I can help you. Ask about products, return policy, or upload jewelry image!",
        "return policy": "You can return items within 15 days with original packaging.",
        "bye": "Goodbye! Feel free to come back anytime!",
    }
    reply = pattern_responses.get(message, "Try asking for help or upload an image!")
    return jsonify({"reply": reply})

@app.route("/api/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    if file:
        # Placeholder: Load precomputed embedding for the uploaded image
        # This should be done offline and passed (e.g., via a form field or separate API)
        filepath = os.path.join('/tmp', file.filename)
        file.save(filepath)
        with open("uploaded_embedding.pkl", "rb") as f:  # Precomputed offline
            uploaded_embedding = pickle.load(f)
        os.remove(filepath)

        # Compute similarity
        similarity = cosine_similarity([uploaded_embedding], image_embeddings)
        best_match_idx = np.argmax(similarity)
        best_route = routes[best_match_idx] if similarity[0][best_match_idx] > 0.7 else None

        if best_route:
            return jsonify({"reply": "Found a matching product!", "route": best_route})
        return jsonify({"reply": "No matching product found.", "route": ""})

    return jsonify({"error": "Invalid file"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4986)
