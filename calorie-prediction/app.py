from flask import Flask, request, render_template
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as model_file:
    classifier, regressor, label_encoder = pickle.load(model_file)

# Set image size
IMAGE_SIZE = (128, 128)

# Define path for saving uploaded images
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to preprocess images
def preprocess_image(image_path):
    img = imread(image_path)
    img_resized = resize(img, IMAGE_SIZE, anti_aliasing=True)
    img_resized = img_resized / 255.0  # Normalize
    return img_resized.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the uploaded image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            
            # Preprocess the image
            image_data = preprocess_image(img_path)
            
            # Get the predicted label (fruit)
            predicted_label = classifier.predict([image_data])
            fruit_name = label_encoder.inverse_transform(predicted_label)[0]
            
            # Get the predicted calorie count
            predicted_calories = regressor.predict([image_data])[0]
            
            # Render the result
            return render_template('index.html', 
                                   fruit=fruit_name, 
                                   calories=predicted_calories, 
                                   image_url=img_path)
    
    return render_template('index.html', fruit=None, calories=None, image_url=None)

if __name__ == '__main__':
    app.run(debug=True)
