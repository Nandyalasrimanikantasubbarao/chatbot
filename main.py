import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from clip import load 
import torch

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load precomputed embeddings
with open("image_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    image_embeddings = data["embeddings"]
    routes = data["routes"]

# ✅ Load the CLIP model ONCE here
device = "cpu"
model, preprocess = load("ViT-B/32", device=device)

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
    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).cpu().numpy()

    similarity = cosine_similarity(image_features, image_embeddings)
    best_match_idx = np.argmax(similarity)
    best_route = routes[best_match_idx]

    return jsonify({"reply": "Found a matching product!", "route": best_route})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
