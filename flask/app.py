#!/usr/bin/python3
from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.models import load_model
import json
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Preload our rice model
print('Loading rice model...')
rice_graph = tf.compat.v1.get_default_graph()


def predict_rice(features):
    with rice_graph.as_default():
        rice_model = load_model('./model/model.h5')

        sample = {
            "Soil_Moisture": features[0],
            "Temperature": features[1],
            "Humidity": features[2],
            "Time": features[3]
        }

        input_dict = {name: tf.convert_to_tensor(
            [value]) for name, value in sample.items()}
        prediction = rice_model.predict(input_dict, steps=50)

        #prediction = rice_model.predict(np.array(features))
        print('prediction: ', prediction[0, 0])
        return np.float64(prediction[0, 0])


@app.route('/rice/predict', methods=['POST'])
def predict_rice_ctrl():
    features = request.get_json()['features']
    print(features[0])
    return jsonify({'prediction': predict_rice(features)})


@app.route("/")
def home_view():
    return "<h1>Hello World</h1>"


if __name__ == '__main__':
    app.run(host='0.0.0.0')
