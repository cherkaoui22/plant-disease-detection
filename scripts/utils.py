"""
scripts/utils.py — PlantCare AI
Chargement du modèle et inférence.
"""
import os, hashlib
import numpy as np
from PIL import Image

DISEASES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", n)
    for n in ["best_model.keras","mobilenetv2.keras","cnn_scratch.keras",
              "resnet50.keras","best_model.h5","mobilenetv2.h5",
              "cnn_scratch.h5","resnet50.h5"]
]

def load_model(path=None):
    try:
        import tensorflow as tf
    except ImportError:
        return None
    for p in ([path] if path else _MODEL_CANDIDATES):
        if p and os.path.exists(p):
            try:
                return tf.keras.models.load_model(p)
            except Exception as e:
                print(f"[utils] Cannot load {p}: {e}")
    return None

def preprocess_image(image, target_size=(224,224)):
    img = image.convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, image):
    """Returns (class_idx, confidence_0_100, probabilities_array)"""
    if model is None:
        # Demo mode: use image pixel hash for variety per image
        arr = np.array(image.convert("RGB").resize((32,32)))
        seed = int(hashlib.md5(arr.tobytes()).hexdigest(), 16) % (2**32)
        rng  = np.random.default_rng(seed)
        probs = rng.dirichlet(np.ones(len(DISEASES)) * 0.3)
        idx  = int(np.argmax(probs))
        return idx, float(probs[idx]) * 100, probs

    arr   = preprocess_image(image)
    preds = model.predict(arr, verbose=0)
    probs = preds[0]
    idx   = int(np.argmax(probs))
    return idx, float(np.max(probs)) * 100, probs

def format_disease_name(raw):
    """'Tomato___Early_blight' → 'Tomate — Early Blight'"""
    if "___" in raw:
        plant, disease = raw.split("___", 1)
    else:
        return raw.replace("_"," ").title()
    plant   = plant.replace("_"," ").replace("(","").replace(")","").strip().title()
    disease = disease.replace("_"," ").strip().title()
    return f"{plant} — {disease}" if disease else plant
