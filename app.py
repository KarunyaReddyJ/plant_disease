from flask import Flask, request, render_template, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('trained_plant_disease_model.keras')

def model_predict(img_path, model):
    # Load the image with the same size as the training images
    img = load_img(img_path, target_size=(128, 128))
    img = img_to_array(img)
    img = np.array([img])
    
    # Make the prediction
    prediction = model.predict(img)
    return prediction

@app.route('/', methods=['GET'])
def index():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']
        
        # Save the file to ./uploads directory
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)

        # Find the class with the highest probability
        ind = np.argmax(prediction)
        class_names = ['Healthy', 'Powdery', 'Rust']  # Replace with your actual class names
        predicted_class = class_names[ind]

        # Return image URL and prediction
        
        return jsonify({
            'prediction': predicted_class
        })

if __name__ == '__main__':
    app.run(debug=True)
