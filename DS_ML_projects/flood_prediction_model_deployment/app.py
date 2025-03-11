from flask import Flask, request, jsonify, render_template #type: ignore
import numpy as np
import tensorflow as tf #type:ignore
import os
from model import load_model, preprocess_input

app = Flask(__name__)

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        precipitation = np.array(data['precipitation']).reshape(1,-1)
        image = np.array(data['image']).reshape(1,128,128,6)

        processed_precip, processed_image = preprocess_input(precipitation, image)
        prediction = model.predict([processed_precip, processed_image])

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host = '0.0.0.0', port =port, debug=True)