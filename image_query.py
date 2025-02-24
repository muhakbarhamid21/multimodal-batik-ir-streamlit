# image_query.py
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

def letterbox_image(img, target_size=(224, 224), fill_color=(0, 0, 0)):

    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(img.shape) == 3:
        canvas = np.full((target_h, target_w, 3), fill_color, dtype=resized.dtype)
    else:
        canvas = np.full((target_h, target_w), fill_color[0] if isinstance(fill_color, tuple) else fill_color, dtype=resized.dtype)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_image(image):
    image = image.convert("L")
    
    img_np = np.array(image)
    
    letterboxed = letterbox_image(img_np, target_size=(224, 224), fill_color=0)
    
    normalized = letterboxed.astype("float32") / 255.0
    
    if len(normalized.shape) == 2:
        normalized = np.stack((normalized,)*3, axis=-1)
    
    return normalized

def load_resnet50_model():
    # Pastikan model sudah disimpan dengan format .keras atau SavedModel
    model = load_model("models/resnet50-globalaveragepooling-256dense.h5", compile=False)
    return model

def get_image_embedding(image, model):
    preprocessed_image = preprocess_image(image)
    img_input = np.expand_dims(preprocessed_image, axis=0)  # shape: (1,224,224,3)
    features = model.predict(img_input)
    # features shape: (1,256)
    embedding = features[0]
    return embedding
