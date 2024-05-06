import sys
import subprocess


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    'Flask',
    'keras',
    'tensorflow',
    'numpy'
]

for package in required_packages:
    install(package)

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io
from keras.applications.mobilenet_v2 import preprocess_input
import webbrowser
import threading


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])



app = Flask(__name__)

# Load the Keras model
model = load_model('models/disease.h5')
model.load_weights('models/model_weights.h5')

def open_browser():
    # Open the default web browser
    webbrowser.open_new_tab('http://127.0.0.1:5000')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_width, img_height = 128, 128  # Adjust according to your model input shape

    # Read the image file as bytes
    img_bytes = img_file.read()

    # Convert the bytes to an image object using io.BytesIO
    img = image.load_img(io.BytesIO(img_bytes),
                         target_size=(img_width, img_height))  # Resize image as required
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocess the input image

    # Perform prediction
    pred = model.predict(img)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(pred)

    # List of disease names
    disease_names = [
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    # Get the name of the predicted disease
    predicted_disease = disease_names[predicted_class_index]

    return jsonify({'prediction': predicted_disease})


if __name__ == '__main__':
    threading.Thread ( target = open_browser ).start ( )
    app.run(debug=True)
