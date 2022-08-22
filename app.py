from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import os

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='F:\\Cat-vs-Dog-Classification\\model.h5'

# Load your trained model
model = load_model(MODEL_PATH)





def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(100, 100))

    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)/255.0
    x = x.reshape(1,100,100,3)

    preds = model.predict(x)
    if preds[0]<=0.5:
        preds="The Image Classified is Cat"
    else:
        preds="The Image Classified is Dog"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'F:\\Cat-vs-Dog-Classification\\uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
