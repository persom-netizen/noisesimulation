from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os
from utils.hog import extract_hog

app = Flask(__name__)

model = joblib.load("model/svm_model.pkl")

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    img = cv2.imread(path)
    
    features = extract_hog(img)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    return render_template('result.html',
                           prediction=prediction,
                           image_path=path)


if __name__ == "__main__":
    app.run(debug=True)