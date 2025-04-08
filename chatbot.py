
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import torch
import clip
import os
from io import BytesIO
import re
import logging
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path='./env') 

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  

PRODUCT_API_URL = os.getenv("PRODUCT_API_URL")


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


product_embeddings = []

def get_image_embedding(image_path_or_url):
    try:
 
        if image_path_or_url.startswith(('http://', 'https://')):
            logger.debug(f"Processing URL: {image_path_or_url}")
            response = requests.get(image_path_or_url, timeout=10, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            logger.debug(f"Processing local file: {image_path_or_url}")
            if not os.path.exists(image_path_or_url):
                raise Exception(f"File not found: {image_path_or_url}")
            image = Image.open(image_path_or_url).convert("RGB")
        

        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        logger.error(f"Error processing image {image_path_or_url}: {e}")
        return None

def fetch_and_compute_product_embeddings():
    global product_embeddings
    product_embeddings = []
    try:
        response = requests.get(PRODUCT_API_URL, timeout=10)
        response.raise_for_status()
        logger.debug(f"Raw response from /api/products: {response.text[:500]}...")  
        data = json.loads(response.text)  
        if not isinstance(data, dict) or 'products' not in data:
            logger.error(f"Expected a dictionary with 'products' key, got {type(data)}")
            return
        products = data['products'] 
        if not isinstance(products, list):
            logger.error(f"Expected a list of products, got {type(products)}")
            return
        for product in products:
            image_url = product.get("imageUrl") or product.get("image")
            if not image_url:
                logger.warning(f"Skipping product {product.get('_id', 'Unknown ID')} due to missing image URL")
                continue
            embedding = get_image_embedding(image_url)
            if embedding is not None:
                product_embeddings.append((product, embedding))
        logger.info(f"Computed embeddings for {len(product_embeddings)} products")
    except Exception as e:
        logger.error(f"Error fetching products from PRODUCT_API_URL: {e}")


fetch_and_compute_product_embeddings()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').lower()

    if "find jewelry by image" in message:
        return jsonify({"reply": "Sure! Please upload an image of the jewelry you are looking for."})

    chatbot_rules = [
        {"pattern": r"(hello|hi|hey)", "response": "Hello! How can I assist you today?"},
        {"pattern": r"help", "response": "I'm here to help! What do you need assistance with?"},
        {"pattern": r"how are you", "response": "I'm doing great, thank you! How about you?"},
        {"pattern": r"(bye|goodbye)", "response": "Goodbye! Have a great day!"},
        {"pattern": r".*", "response": "Sorry, I didn't understand that. Can you please rephrase?"}
    ]

    for rule in chatbot_rules:
        if re.search(rule['pattern'], message, re.IGNORECASE):
            return jsonify({"reply": rule['response']})

    return jsonify({"reply": "I don't understand"})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        logger.error("No image part in request")
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        logger.error("No selected image")
        return jsonify({"error": "No selected image"}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Uploaded file saved to {filepath}")

        # Get embedding for uploaded image (treat as local file)
        uploaded_emb = get_image_embedding(filepath)
        os.remove(filepath)  # Clean up after processing

        if uploaded_emb is None:
            logger.error(f"Failed to process uploaded image from {filepath}")
            return jsonify({"error": "Failed to process uploaded image"}), 400

        # Find the best match
        best_match = None
        highest_similarity = -1

        for product, emb in product_embeddings:
            similarity = torch.cosine_similarity(uploaded_emb, emb).item()
            logger.debug(f"Comparing with {product.get('name', 'Unknown')}: Similarity = {similarity:.4f}")
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = product

        if best_match and highest_similarity > 0.7:  # Threshold lowered to 0.7
            route = f"/product/{str(best_match['_id'])}"
            logger.info(f"Match found: {best_match['name']} with similarity {highest_similarity:.2f}")
            return jsonify({
                "reply": f"We found a match: {best_match['name']} (Similarity: {highest_similarity:.2f})",
                "route": route
            })
        else:
            logger.info(f"No match found. Highest similarity: {highest_similarity:.2f}")
            return jsonify({"reply": "No similar jewelry found", "route": ""})

    logger.error("Invalid file")
    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
