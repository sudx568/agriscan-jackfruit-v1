import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model you are training right now
model = tf.keras.models.load_model('jackfruit_model.keras')

def predict_quality(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 0.0 is BAD, 1.0 is GOOD
    score = model.predict(img_array)[0][0]
    
    if score < 0.5:
        # This is the "Bad" side of the scale
        label = "BAD JACKFRUIT"
        # Calculate how 'Bad' it is (e.g., score 0.1 means 90% Bad)
        conf = f"{(1 - score) * 100:.2f}%"
    else:
        # This is the "Good" side
        label = "GOOD JACKFRUIT"
        conf = f"{score * 100:.2f}%"
        
    return label, conf

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            label, conf = predict_quality(filepath)
            return render_template('index.html', label=label, conf=conf, img=filepath)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)