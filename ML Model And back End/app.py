from unittest import result
import cv2,os
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename


model_path='ML Model And back End\model.h5'
model=load_model(model_path)
app = Flask(__name__)



diseases_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

labels=['nv','mel','bkl','bcc','akiec','vasc','df']

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']
        sfname = 'uploads/'+str(secure_filename(f.filename))
        f.save(sfname) 
        f=cv2.imread(sfname)
        f=cv2.resize(f,(32,32))
        f=f.reshape(-1,32,32,3)
        p=model.predict([f])
        x=np.argmax(p)
        os.remove(sfname)
        return render_template('index.html',result= diseases_dict[labels[x]])
       
   



if __name__ == "__main__":
    app.run(debug=True)