import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import clip
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity  # ✅ IMPORT THIS

app = Flask(__name__)
CORS(app)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load precomputed image embeddings
with open("image_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    image_embeddings = data["embeddings"]   # shape: (n, 512)
    routes = data["routes"]                 # list of routes (e.g., "/products/diamond_ring")

# Rule-based chatbot logic
pattern_responses = {
    "help": "Sure! I can help you. You can ask me about our products, return policy, or even upload a jewelry image to find similar items.",
    "how are you": "I'm just a bot, but I'm always ready to help you find the perfect jewelry!",
    "bye": "Goodbye! Feel free to come back anytime if you need help.",
    "return policy": "Our return policy allows returns within 15 days of purchase with original packaging.",
    "find jewelry by image": "Please upload an image and I’ll try to find the most similar product we have!"
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').lower().strip()

    reply = pattern_responses.get(message, "I'm not sure how to respond to that. Try asking for help or upload an image!")

    return jsonify({'reply': reply})

@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features = image_features.cpu().numpy()  # shape (1, 512)

    # ✅ Compute similarity
    similarities = cosine_similarity(image_features, image_embeddings)  # both 512-d
    best_match_index = np.argmax(similarities)

    # ✅ Get route
    best_route = routes[best_match_index]

    return jsonify({
        "reply": "Here’s a similar product we found!",
        "route": best_route
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
