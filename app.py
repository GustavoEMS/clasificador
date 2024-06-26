from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__, static_folder='static')

# Cargar el modelo y las clases
model = load_model('modelo.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        img_path = f"./{file.filename}"
        file.save(img_path)
        img_array = prepare_image(img_path)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        os.remove(img_path)  # Eliminar la imagen después de la predicción
        return jsonify({'class': predicted_class})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
