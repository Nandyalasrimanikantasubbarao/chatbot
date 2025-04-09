import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Load precomputed embeddings
with open("image_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    image_embeddings = data["embeddings"]  # shape: (n, 512)
    routes = data["routes"]                # list of routes (e.g., "/product/123")

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
    from clip import load 
    import torch

    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    device = "cpu"
    model, preprocess = load("ViT-B/32", device=device)

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).cpu().numpy()

    similarity = cosine_similarity(image_features, image_embeddings)
    best_match_idx = np.argmax(similarity)
    best_route = routes[best_match_idx]

    return jsonify({"reply": "Found a matching product!", "route": best_route})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4988))  # <- critical!
    app.run(host="0.0.0.0", port=port)
