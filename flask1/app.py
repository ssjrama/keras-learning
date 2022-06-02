from flask import Flask, jsonify, request
from keras.models import load_model
import json
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Preload our diabetes model
print('Loading diabetes model...')
diabetes_graph = tf.compat.v1.get_default_graph()


def predict_diabetes(features):
    with diabetes_graph.as_default():
        diabetes_model = load_model('./model/model.h5')

        sample = {
            "Soil_Moisture": features[0],
            "Temperature": features[1],
            "Humidity": features[2],
            "Time": features[3]
        }

        input_dict = {name: tf.convert_to_tensor(
            [value]) for name, value in sample.items()}
        prediction = diabetes_model.predict(input_dict, steps=50)

        #prediction = diabetes_model.predict(np.array(features))
        print('prediction: ', prediction[0, 0])
        return np.float64(prediction[0, 0])


@app.route('/diabetes/predict', methods=['POST'])
def predict_diabetes_ctrl():
    features = request.get_json()['features']
    print(features[0])
    return jsonify({'prediction': predict_diabetes(features)})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
