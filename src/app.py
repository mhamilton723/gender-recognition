from __future__ import division, print_function

import io
import os
from os.path import join
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from gevent.wsgi import WSGIServer
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# Define a flask app
app = Flask(__name__,
            static_folder=join("..", "website", "static"),
            template_folder=join("..", "website", "templates"))

# Model saved with Keras model.save()
MODEL_PATH = os.path.join("..", "saved_models", "run_1.model")

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()  # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
print('Model loaded. Check http://localhost:5000/')
width = model.layers[0].batch_input_shape[1]
height = model.layers[0].batch_input_shape[2]


def prepare_image(image, target):
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.

    # return the processed image
    return image


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        image = request.files['file'].read()
        image = Image.open(io.BytesIO(image))

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(width, height))

        # classify the input image and then initialize the list
        # of predictions to return to the client
        male_prob = model.predict(image)[0][0] * 100.
        if male_prob >= 50:
            result = "Male ({}%)".format(round(male_prob,2))
        else:
            result = "Female ({}%)".format(round(100-male_prob, 2))
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
