from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load precomputed image embeddings
with open("image_embeddings.pkl", "rb") as f:
    image_embeddings = pickle.load(f)  # { "product_id": embedding (1D np.array) }

# Rule-based chatbot responses
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


@app.route('/api/upload', methods=['POST'])
def upload():
    try:
        file = request.files['image']
        img = Image.open(file).convert("RGB")

        # Make sure to match this processing with how you created your embeddings
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_vector = img_array.transpose(2, 0, 1).flatten().reshape(1, -1)  # shape: (1, N)

        # Compare with stored embeddings
        best_match = None
        best_score = -1

        for product_id, embedding in image_embeddings.items():
            embedding = embedding.reshape(1, -1)  # Ensure shape matches
            score = cosine_similarity(img_vector, embedding)[0][0]
            if score > best_score:
                best_score = score
                best_match = product_id

        product_route = f"/product/{best_match}"
        return jsonify({'route': product_route, 'reply': 'Found a matching product! Click below to view it:'})

    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
