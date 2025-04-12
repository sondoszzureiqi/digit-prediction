import pickle
from PIL import Image
import numpy as np  
from flask import Flask, request, render_template 
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploads'

# Running the flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():  
    
    # File upload
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocessing image
    img = Image.open(file_path)
    img = img.convert('1')  # Convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.flatten().reshape(-1, 784)

    # Model prediction
    prediction = model.predict(img) 
    return render_template('index.html', prediction_text='The digit is {}'.format(prediction))

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0')

